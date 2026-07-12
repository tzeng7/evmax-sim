"""Walk-forward validation for the NCAAB / NCAAW opponent-adjusted efficiency stack.

Strictly point-in-time: predictions for game G use only games before G.

  - elo / form start EMPTY and are walk-forward updated game by game.
    Elo is swept over a (K, HOME_ADVANTAGE) grid — this doubles as the
    MODEL-2 NCAAW Elo calibration harness.
  - The efficiency state is REGENERATED every --reseed-days during the walk
    (mirrors the weekly live re-seed): at each cutoff the season's cached
    games with date < cutoff are re-solved through the opponent-adjustment
    solver. Prior-season data is NEVER used (mirrors the live staleness
    guard, which silences prior-season state in-season).
  - Efficiency margins are recorded per SHRINK_K value; the Φ transform's
    SCORE_STDEV is swept offline (margins don't depend on it).
  - The possession sim shares SHRINK_K with efficiency (as in live).

Protocol: sweep every choice (elo K/HA, shrink k, stdev, blend weights) on
--train-season ONLY; evaluate the frozen winner on --holdout-season. The
gate: new blend must beat the current blend on HOLDOUT Brier, and each new
model must not be strictly harmful standalone.

Usage:
    python scripts/backtest_ncaab_efficiency.py --league mens
    python scripts/backtest_ncaab_efficiency.py --league womens
    python scripts/backtest_ncaab_efficiency.py --league mens --refresh
    python scripts/backtest_ncaab_efficiency.py --league mens --verify-trank

Requires the day-file cache from scripts/seed_ncaab_efficiency.py (fetches
any gaps automatically).
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from evmax.agents.models._college_efficiency import (
    MIN_D1_APPEARANCES,
    dean_oliver_possessions,
    filter_d1_games,
    opponent_adjust,
    project_matchup,
    shrink_team_stats,
    simulate_matchup_vectorized,
)
from evmax.agents.models.form_agent import FormModelAgent
from evmax.backtest.metrics import accuracy_score, brier_score
from evmax.backtest.sources.espn_walkforward import _form_predict, _form_update
from scripts.seed_ncaab_efficiency import LEAGUE_CONFIG, load_season_games, to_solver_rows

CACHE_ROOT = _REPO_ROOT / "data" / "backtest" / "college_efficiency"

# --- Sweep grids (train season only) ---------------------------------------

SHRINK_KS = [6.0, 10.0, 14.0]
SCORE_STDEVS = [9.5, 10.5, 11.5, 12.5]

ELO_GRIDS: dict[str, list[tuple[float, float]]] = {
    # (K, HOME_ADVANTAGE_ELO). Men's grid brackets the current live config
    # (20, 70); women's grid is the MODEL-2 calibration sweep and includes the
    # de-facto current config (20, 0 — no ncaaw entry in HOME_ADVANTAGE_ELO).
    "mens": [(k, ha) for k in (15.0, 20.0, 25.0, 30.0) for ha in (50.0, 70.0, 90.0)],
    "womens": [(k, ha) for k in (15.0, 20.0, 25.0, 30.0, 35.0)
               for ha in (0.0, 40.0, 60.0, 80.0, 100.0)],
}

# The blend running in production today (categories.yaml + ensemble class
# weights; poisson is agent-gated to soccer so the effective ncaab blend is
# elo+form; ncaaw's documented blend is form-only but generic elo does fire
# with uncalibrated constants, so both baselines are evaluated).
CURRENT_ELO_CFG: dict[str, tuple[float, float]] = {
    "mens": (20.0, 70.0),
    "womens": (20.0, 0.0),
}

WEIGHT_GRID = {
    "elo": [0.0, 0.10, 0.20, 0.30, 0.40],
    "form": [0.0, 0.10, 0.20, 0.30],
    "eff": [0.10, 0.20, 0.30, 0.40, 0.50],
    "sim": [0.0, 0.10, 0.20, 0.30, 0.40],
}

BACKTEST_N_SIMS = 4000

POSS_CLIP = {"mens": (55.0, 90.0), "womens": (55.0, 92.0)}


@dataclass
class GameRec:
    season: int
    game_date: date
    home_id: str
    away_id: str
    home_won: bool
    neutral: bool
    elo: dict = field(default_factory=dict)        # (K, HA) -> P(home)
    form: Optional[float] = None
    eff_margin: dict = field(default_factory=dict)  # shrink_k -> projected margin
    sim_prob: dict = field(default_factory=dict)    # shrink_k -> P(home)


# ---------------------------------------------------------------------------
# Point-in-time efficiency state
# ---------------------------------------------------------------------------


def build_pit_state(
    season_games: list[dict],
    cutoff: date,
    season_start: date,
) -> Optional[dict]:
    """Solve the opponent adjustment from this season's games strictly before cutoff."""
    before = [g for g in season_games if date.fromisoformat(g["date"]) < cutoff]
    if not before:
        return None
    weeks = max(1, (cutoff - season_start).days // 7)
    min_app = min(MIN_D1_APPEARANCES, max(2, weeks))
    rows = to_solver_rows(before)
    d1_rows, _ = filter_d1_games(rows, min_app)
    if len(d1_rows) < 50:
        return None
    solved = opponent_adjust(d1_rows)

    # tov_pct per team for the sim (D1 games only).
    d1_pairs = {(r["home_id"], r["away_id"]) for r in d1_rows}
    tov_tot: dict[str, list[float]] = {}
    for g in before:
        h, a = g["home"], g["away"]
        if (h["id"], a["id"]) not in d1_pairs:
            continue
        for own in (h, a):
            poss = dean_oliver_possessions(own["fga"], own["fta"], own["tov"], own["oreb"])
            if poss <= 0:
                continue
            acc = tov_tot.setdefault(own["id"], [0.0, 0.0])
            acc[0] += own["tov"]
            acc[1] += poss

    teams = {}
    for tid, s in solved["teams"].items():
        acc = tov_tot.get(tid)
        teams[tid] = {
            "ortg": s["adj_o"], "drtg": s["adj_d"], "pace": s["pace"], "gp": s["gp"],
            "tov_pct": (acc[0] / acc[1]) if acc and acc[1] > 0 else None,
        }
    tovs = [t["tov_pct"] for t in teams.values() if t["tov_pct"]]
    league = {
        "league_avg_ortg": solved["league_avg_eff"],
        "league_avg_drtg": solved["league_avg_eff"],
        "league_avg_pace": solved["league_avg_pace"],
        "league_avg_tov_pct": (sum(tovs) / len(tovs)) if tovs else 0.17,
    }
    return {"teams": teams, "league": league, "hca_eff": solved["hca_eff"]}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def run_walk(league: str, seasons: list[int], reseed_days: int) -> list[GameRec]:
    elo_grid = ELO_GRIDS[league]
    if CURRENT_ELO_CFG[league] not in elo_grid:
        elo_grid = elo_grid + [CURRENT_ELO_CFG[league]]
    sector = LEAGUE_CONFIG[league]["sector"]
    poss_clip = POSS_CLIP[league]

    # elo state per config: cfg -> {team_id: rating}
    elo_ratings: dict[tuple, dict[str, float]] = {cfg: {} for cfg in elo_grid}
    elo_counts: dict[str, int] = {}

    form = FormModelAgent()
    form._state = {}

    recs: list[GameRec] = []
    min_games_gate = 4

    for season in seasons:
        games = load_season_games(league, season, fetch_missing=True)
        print(f"[{league} {season}] {len(games)} games")
        season_start = date(season, 11, 1)
        pit_state: Optional[dict] = None
        last_seed: Optional[date] = None

        for g in games:
            gd = date.fromisoformat(g["date"])
            h_id, a_id = g["home"]["id"], g["away"]["id"]
            h_name = g["home"]["display"].lower()
            a_name = g["away"]["display"].lower()
            home_won = g["home"]["score"] > g["away"]["score"]

            if last_seed is None or (gd - last_seed).days >= reseed_days:
                pit_state = build_pit_state(games, cutoff=gd, season_start=season_start)
                last_seed = gd

            rec = GameRec(
                season=season, game_date=gd, home_id=h_id, away_id=a_id,
                home_won=home_won, neutral=g["neutral"],
            )

            # --- Elo (per config) ---
            for cfg in elo_grid:
                ratings = elo_ratings[cfg]
                if h_id in ratings or a_id in ratings:
                    k, ha = cfg
                    eh = ratings.get(h_id, 1500.0) + ha
                    ea = ratings.get(a_id, 1500.0)
                    rec.elo[cfg] = 1.0 / (1.0 + 10 ** ((ea - eh) / 400.0))

            # --- Form ---
            rec.form = _form_predict(form, sector, h_name, a_name, gd)

            # --- Efficiency + sim (per shrink_k) ---
            if pit_state is not None:
                sh_raw = pit_state["teams"].get(h_id)
                sa_raw = pit_state["teams"].get(a_id)
                if (sh_raw and sa_raw
                        and sh_raw["gp"] >= min_games_gate
                        and sa_raw["gp"] >= min_games_gate):
                    league_avgs = pit_state["league"]
                    hca = pit_state["hca_eff"]
                    for sk in SHRINK_KS:
                        proj = project_matchup(
                            sh_raw, sa_raw, league_avgs,
                            hca_eff=hca, score_stdev=10.5, shrink_k=sk,
                        )
                        rec.eff_margin[sk] = proj["margin"]
                        rng = np.random.default_rng(
                            seed=hash(f"{sector}_{h_id}_{a_id}_{gd}_{sk}") & 0xFFFFFFFF
                        )
                        sim = simulate_matchup_vectorized(
                            proj["stats_home"], proj["stats_away"], league_avgs,
                            hca_eff=hca, n_sims=BACKTEST_N_SIMS, rng=rng,
                            poss_clip=poss_clip,
                        )
                        rec.sim_prob[sk] = sim["prob_home"]

            recs.append(rec)

            # --- Updates (after prediction) ---
            for cfg, ratings in elo_ratings.items():
                k, ha = cfg
                eh = ratings.get(h_id, 1500.0)
                ea = ratings.get(a_id, 1500.0)
                expected_h = 1.0 / (1.0 + 10 ** ((ea - (eh + ha)) / 400.0))
                delta = k * ((1.0 if home_won else 0.0) - expected_h)
                ratings[h_id] = eh + delta
                ratings[a_id] = ea - delta
            _form_update(form, sector, h_name, a_name, home_won, gd)

    return recs


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def eff_prob(rec: GameRec, shrink_k: float, stdev: float) -> Optional[float]:
    m = rec.eff_margin.get(shrink_k)
    if m is None:
        return None
    return max(0.02, min(0.98, _phi(m / stdev)))


def _metrics(pairs: list[tuple[float, bool]]) -> dict:
    if not pairs:
        return {"n": 0, "brier": float("nan"), "acc": float("nan")}
    ps = [p for p, _ in pairs]
    ys = [y for _, y in pairs]
    return {"n": len(ps), "brier": brier_score(ps, ys), "acc": accuracy_score(ps, ys)}


def blend_prob(
    rec: GameRec,
    weights: dict[str, float],
    elo_cfg: tuple,
    shrink_k: float,
    stdev: float,
) -> Optional[float]:
    pieces = [
        (rec.elo.get(elo_cfg), weights.get("elo", 0.0)),
        (rec.form, weights.get("form", 0.0)),
        (eff_prob(rec, shrink_k, stdev), weights.get("eff", 0.0)),
        (rec.sim_prob.get(shrink_k), weights.get("sim", 0.0)),
    ]
    tw = sum(w for p, w in pieces if p is not None and w > 0)
    if tw <= 0:
        return None
    return sum(p * w for p, w in pieces if p is not None and w > 0) / tw


def evaluate_blend(
    recs: list[GameRec],
    weights: dict[str, float],
    elo_cfg: tuple,
    shrink_k: float,
    stdev: float,
) -> dict:
    pairs = []
    for r in recs:
        p = blend_prob(r, weights, elo_cfg, shrink_k, stdev)
        if p is not None:
            pairs.append((p, r.home_won))
    return _metrics(pairs)


def eval_pool(recs: list[GameRec], season: int, shrink_k: float) -> list[GameRec]:
    """Comparable eval pool: season games where the efficiency stack fired."""
    return [r for r in recs if r.season == season and shrink_k in r.eff_margin]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--league", choices=list(LEAGUE_CONFIG), default="mens")
    ap.add_argument("--train-season", type=int, default=2024)
    ap.add_argument("--holdout-season", type=int, default=2025)
    ap.add_argument("--reseed-days", type=int, default=7)
    ap.add_argument("--refresh", action="store_true", help="ignore the walk pickle cache")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--verify-trank", action="store_true",
                    help="sanity-correlate final adjusted ratings vs barttorvik T-Rank (mens only)")
    args = ap.parse_args()

    league = args.league
    seasons = [args.train_season, args.holdout_season]
    cache = CACHE_ROOT / f"{league}_walkforward_{seasons[0]}_{seasons[1]}_r{args.reseed_days}.pkl"

    if cache.exists() and not args.refresh:
        print(f"[cache] loading {cache}")
        recs: list[GameRec] = pickle.loads(cache.read_bytes())
    else:
        recs = run_walk(league, seasons, args.reseed_days)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(recs))
        print(f"[cache] wrote {cache} ({len(recs)} games)")

    train_s, hold_s = args.train_season, args.holdout_season
    elo_grid = list(recs[0].elo.keys()) if recs else []
    elo_grid = sorted({cfg for r in recs for cfg in r.elo})
    current_cfg = tuple(CURRENT_ELO_CFG[league])

    # ---- 1. Elo (K, HA) sweep on TRAIN ------------------------------------
    print("\n" + "=" * 78)
    print(f"ELO (K, HOME_ADV) sweep — {league} — TRAIN {train_s}-{train_s+1}")
    print("=" * 78)
    elo_rows = []
    for cfg in elo_grid:
        pairs = [(r.elo[cfg], r.home_won) for r in recs if r.season == train_s and cfg in r.elo]
        m = _metrics(pairs)
        elo_rows.append((m["brier"], cfg, m))
    elo_rows.sort(key=lambda x: x[0])
    for brier, cfg, m in elo_rows[:8]:
        tag = "  <- current" if cfg == current_cfg else ""
        print(f"  K={cfg[0]:>5.1f} HA={cfg[1]:>5.1f}  n={m['n']:>5}  Brier={brier:.4f}  Acc={m['acc']*100:.1f}%{tag}")
    cur_m = next((m for b, cfg, m in elo_rows if cfg == current_cfg), None)
    if cur_m:
        print(f"  current cfg K={current_cfg[0]} HA={current_cfg[1]}: Brier={cur_m['brier']:.4f}")
    best_elo_cfg = elo_rows[0][1]
    print(f"  chosen: K={best_elo_cfg[0]} HA={best_elo_cfg[1]}")

    # ---- 2. Efficiency (shrink_k, stdev) sweep on TRAIN --------------------
    print("\n" + "=" * 78)
    print(f"EFFICIENCY (shrink_k, score_stdev) sweep — TRAIN {train_s}-{train_s+1}")
    print("=" * 78)
    eff_rows = []
    for sk, sd in product(SHRINK_KS, SCORE_STDEVS):
        pairs = []
        for r in recs:
            if r.season != train_s:
                continue
            p = eff_prob(r, sk, sd)
            if p is not None:
                pairs.append((p, r.home_won))
        m = _metrics(pairs)
        eff_rows.append((m["brier"], sk, sd, m))
    eff_rows.sort(key=lambda x: x[0])
    for brier, sk, sd, m in eff_rows[:8]:
        print(f"  k={sk:>4.0f} stdev={sd:>4.1f}  n={m['n']:>5}  Brier={brier:.4f}  Acc={m['acc']*100:.1f}%")
    best_sk, best_sd = eff_rows[0][1], eff_rows[0][2]
    print(f"  chosen: shrink_k={best_sk} score_stdev={best_sd}")

    # sim standalone at chosen shrink_k (train)
    sim_train = _metrics([(r.sim_prob[best_sk], r.home_won)
                          for r in recs if r.season == train_s and best_sk in r.sim_prob])
    print(f"  sim standalone (k={best_sk}) train: n={sim_train['n']} Brier={sim_train['brier']:.4f}")

    # ---- 3. Blend weight sweep on TRAIN ------------------------------------
    print("\n" + "=" * 78)
    print(f"BLEND weight sweep — TRAIN {train_s}-{train_s+1} (eval pool: efficiency fired)")
    print("=" * 78)
    train_pool = eval_pool(recs, train_s, best_sk)
    combos = []
    for we, wf, wef, ws in product(WEIGHT_GRID["elo"], WEIGHT_GRID["form"],
                                   WEIGHT_GRID["eff"], WEIGHT_GRID["sim"]):
        s = we + wf + wef + ws
        if s <= 0 or s < 0.5 or s > 1.5:
            continue
        w = {"elo": we / s, "form": wf / s, "eff": wef / s, "sim": ws / s}
        m = evaluate_blend(train_pool, w, best_elo_cfg, best_sk, best_sd)
        if m["n"]:
            combos.append((m["brier"], w, m))
    combos.sort(key=lambda x: x[0])
    for brier, w, m in combos[: args.top]:
        ws = " ".join(f"{k}={v:.2f}" for k, v in w.items())
        print(f"  Brier={brier:.4f}  Acc={m['acc']*100:.1f}%  n={m['n']}   {ws}")
    best_w = combos[0][1]

    # Baselines on train pool
    cur_mens = {"elo": 0.35 / 0.60, "form": 0.25 / 0.60, "eff": 0.0, "sim": 0.0}
    cur_womens_doc = {"elo": 0.0, "form": 1.0, "eff": 0.0, "sim": 0.0}
    baselines = {"current (elo+form, live cfg)": (cur_mens, current_cfg)}
    if league == "womens":
        baselines["current-doc (form only)"] = (cur_womens_doc, current_cfg)
    for name, (w, cfg) in baselines.items():
        m = evaluate_blend(train_pool, w, cfg, best_sk, best_sd)
        print(f"  [baseline] {name}: Brier={m['brier']:.4f} Acc={m['acc']*100:.1f}% n={m['n']}")

    # ---- 4. HOLDOUT gate ----------------------------------------------------
    print("\n" + "=" * 78)
    print(f"HOLDOUT {hold_s}-{hold_s+1} — frozen train choices "
          f"(elo K={best_elo_cfg[0]} HA={best_elo_cfg[1]}, k={best_sk}, stdev={best_sd})")
    print("=" * 78)
    hold_pool = eval_pool(recs, hold_s, best_sk)
    print(f"  eval pool: {len(hold_pool)} games (efficiency fired)")

    results = {}
    for name, w, cfg in [
        ("NEW blend", best_w, best_elo_cfg),
        *[(f"baseline: {n}", w, c) for n, (w, c) in baselines.items()],
    ]:
        m = evaluate_blend(hold_pool, w, cfg, best_sk, best_sd)
        results[name] = m
        print(f"  {name:38s} n={m['n']:>5}  Brier={m['brier']:.4f}  Acc={m['acc']*100:.1f}%")

    print("\n  standalone models on holdout pool:")
    stand = {
        f"elo (chosen K={best_elo_cfg[0]}, HA={best_elo_cfg[1]})":
            [(r.elo[best_elo_cfg], r.home_won) for r in hold_pool if best_elo_cfg in r.elo],
        f"elo (current K={current_cfg[0]}, HA={current_cfg[1]})":
            [(r.elo[current_cfg], r.home_won) for r in hold_pool if current_cfg in r.elo],
        "form": [(r.form, r.home_won) for r in hold_pool if r.form is not None],
        f"efficiency (k={best_sk}, sd={best_sd})":
            [(eff_prob(r, best_sk, best_sd), r.home_won) for r in hold_pool
             if eff_prob(r, best_sk, best_sd) is not None],
        f"possession_sim (k={best_sk})":
            [(r.sim_prob[best_sk], r.home_won) for r in hold_pool if best_sk in r.sim_prob],
    }
    for name, pairs in stand.items():
        m = _metrics(pairs)
        print(f"    {name:42s} n={m['n']:>5}  Brier={m['brier']:.4f}  Acc={m['acc']*100:.1f}%")

    # Leave-one-out on holdout for each NEW model
    print("\n  leave-one-out (holdout, NEW blend):")
    for drop in ("eff", "sim", "elo", "form"):
        if best_w.get(drop, 0.0) <= 0:
            continue
        w2 = {k: (0.0 if k == drop else v) for k, v in best_w.items()}
        s = sum(w2.values())
        if s <= 0:
            continue
        w2 = {k: v / s for k, v in w2.items()}
        m = evaluate_blend(hold_pool, w2, best_elo_cfg, best_sk, best_sd)
        d = results["NEW blend"]["brier"] - m["brier"]
        print(f"    without {drop:5s}: Brier={m['brier']:.4f} "
              f"(Δ vs full = {d:+.4f}; negative = model was helping)")

    # Gate verdict
    print("\n" + "=" * 78)
    new_b = results["NEW blend"]["brier"]
    verdicts = []
    for name, m in results.items():
        if name == "NEW blend":
            continue
        delta = m["brier"] - new_b
        verdicts.append(delta > 0)
        print(f"  GATE vs {name}: ΔBrier={delta:+.4f} "
              f"({'PASS' if delta > 0 else 'FAIL'} — positive = new blend better)")
    print(f"  OVERALL: {'VALIDATED' if all(verdicts) else 'REJECTED'}")
    print("=" * 78)

    # ---- 5. Optional barttorvik sanity check --------------------------------
    if args.verify_trank and league == "mens":
        _verify_trank(recs, hold_s)

    return 0


def _verify_trank(recs: list[GameRec], season: int) -> None:
    """Correlate end-of-season adjusted ratings vs barttorvik T-Rank (best effort)."""
    try:
        import httpx
        from scripts.seed_ncaab_efficiency import build_state
        games = load_season_games("mens", season, fetch_missing=False)
        state = build_state(games, season, "ncaab")
        r = httpx.get(
            f"https://barttorvik.com/trank.php?json=1&year={season + 1}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        # row format: list, [0]=adjoe? — detect name + adjoe/adjde positions
        ours, theirs_o, theirs_d = [], [], []
        by_name = {}
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            # T-Rank rows: [rank?, team, conf, ..., adjoe, adjde, ...] — the
            # team name is the first string field; adjoe/adjde the first two
            # floats > 60 < 140.
            name = next((x for x in row if isinstance(x, str) and len(x) > 3), None)
            floats = [x for x in row if isinstance(x, (int, float)) and 60 < float(x) < 140]
            if name and len(floats) >= 2:
                by_name[name.lower()] = (float(floats[0]), float(floats[1]))
        matched = 0
        for key, t in state["teams"].items():
            cand = by_name.get(t["full_name"]) or by_name.get(key)
            if not cand:
                continue
            matched += 1
            ours.append(t["ortg"] - t["drtg"])
            theirs_o.append(cand[0])
            theirs_d.append(cand[1])
        if matched < 50:
            print(f"[trank] only {matched} teams matched — skipping correlation")
            return
        ours_a = np.asarray(ours)
        theirs_net = np.asarray(theirs_o) - np.asarray(theirs_d)
        corr = float(np.corrcoef(ours_a, theirs_net)[0, 1])
        print(f"[trank] matched {matched} teams; net-rating correlation vs T-Rank: {corr:.3f}")
        print("[trank] (>= ~0.9 indicates the opponent-adjustment solver is sound)")
    except Exception as e:  # noqa: BLE001 — optional check, never fail the run
        print(f"[trank] verification skipped ({e})")


if __name__ == "__main__":
    raise SystemExit(main())
