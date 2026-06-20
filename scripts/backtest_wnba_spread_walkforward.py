"""TRUE walk-forward backtest of the WNBA spread (cover) model — no lookahead.

The sibling scripts/backtest_wnba_totals_walkforward.py does this for over/under;
this is the spread/cover analogue. Until now WNBA spread had NO walk-forward at
all — the only spread backtest in the repo (run_spread_backtest in
espn_walkforward.py) is NBA-only and loads TODAY's efficiency state, so it leaks
future box scores and never compares against the real market.

For each 2026 WNBA game in chronological order this script:
  1. reconstructs the efficiency state from ONLY the box scores of games played
     strictly before this game's date (Dean Oliver four-factors, reusing the exact
     functions from seed_wnba_efficiency.py so the state is identical to production),
  2. injects that as-of-date state into WNBAPossessionSimAgent,
  3. prices P(cover) at every archived Kalshi spread strike and at Pinnacle's
     spread line, using the SAME call the live EV-gap path uses
     (sim.cover_probability(eid, -strike, yes_is_underdog=<yes is the away team>)),
  4. resolves against the ESPN final margin (the title's "wins by over X" strike is
     authoritative — the archived `line` column is unreliable: some snapshots
     pre-date the floor_strike parser fix and are a full point too long),
  5. THEN folds this game into the running totals for the next prediction.

This mirrors live behaviour: production re-seeds the efficiency state from the
current season (`seed_wnba_efficiency.py --year 2026`), so on a June game day the
model has 2026-to-date stats only. Early-season games where either team has
< MIN_GAMES are skipped by the sim's own gate — exactly as they would be live.

We compare three priced probabilities on identical resolved outcomes:
  - model    : sim.cover_probability  (margin sim mean + σ=12.5 normal CDF)
  - kalshi   : the archived YES (=cover) ask at that strike
  - pinnacle : devigged true_prob that the favourite covers its line

If the model's Brier is not below Kalshi's / Pinnacle's, the model is behind the
market and spread should stay in shadow (same conclusion the totals walk-forward
reached — see the project_wnba_totals_no_edge memory).

Run:  .venv/bin/python scripts/backtest_wnba_spread_walkforward.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
# Import the worktree's evmax + scripts FIRST, ahead of any editable-installed
# checkout on sys.path (the .venv may point at a different evmax tree).
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import seed_wnba_efficiency as seed  # noqa: E402  (reuse exact production formulas)
from evmax.agents.models.wnba_possession_sim_agent import (  # noqa: E402
    MIN_GAMES,
    SPREAD_SIGMA,
    WNBAPossessionSimAgent,
)
from evmax.backtest.metrics import (  # noqa: E402
    accuracy_score,
    brier_score,
    calibration_bins,
)
from evmax.matching.normalizer import NameNormalizer  # noqa: E402
from evmax.models.odds import SharpBook, SharpOdds  # noqa: E402
from evmax.models_ml.spread_distribution import SpreadDistributionModel  # noqa: E402

ARCHIVE = REPO / "data" / "archive.db"
CACHE = REPO / "data" / "backtest" / "wnba_totals_walkforward_games.json"
MONTHS = ["202605", "202606"]
# Kalshi WNBA spread titles read "New York wins by over 4.5 points?" — the strike
# (4.5) is the cover threshold and is authoritative (see module docstring).
STRIKE_RE = re.compile(r"wins by over (\d+(?:\.\d+)?) points")


# ---------------------------------------------------------------------------
# Box-score game log (shared on-disk cache with the totals walk-forward)
# ---------------------------------------------------------------------------
def fetch_game_log(months: list[str] | None = None,
                   cache: Path | None = None) -> list[dict]:
    """Chronological list of completed WNBA games with box-score stats.

    Identical structure / cache to backtest_wnba_totals_walkforward.fetch_game_log
    so the two backtests share one ESPN pull. Each item:
      {date, home_name, away_name, home_score, away_score, home_stats, away_stats}
    """
    months = months or MONTHS
    cache = cache or CACHE
    if cache.exists():
        return json.loads(cache.read_text())

    games: list[dict] = []
    with httpx.Client() as client:
        for month in months:
            events = seed._fetch_month_events(client, month)
            print(f"  {month}: {len(events)} completed games")
            for e in events:
                comp = e["competitions"][0]
                teams = comp.get("competitors", [])
                home_t = next((t for t in teams if t.get("homeAway") == "home"), None)
                away_t = next((t for t in teams if t.get("homeAway") == "away"), None)
                if not home_t or not away_t:
                    continue
                summary = seed._fetch_summary(client, e.get("id", ""))
                time.sleep(seed.REQUEST_DELAY)
                if not summary:
                    continue
                bs_teams = summary.get("boxscore", {}).get("teams", [])
                if len(bs_teams) != 2:
                    continue
                bs_home = next((t for t in bs_teams if t.get("homeAway") == "home"), None)
                bs_away = next((t for t in bs_teams if t.get("homeAway") == "away"), None)
                hs = seed._extract_team_stats(bs_home) if bs_home else None
                as_ = seed._extract_team_stats(bs_away) if bs_away else None
                if not hs or not as_:
                    continue
                games.append({
                    "date": e["date"][:10],
                    "home_name": home_t["team"].get("displayName", ""),
                    "away_name": away_t["team"].get("displayName", ""),
                    "home_score": int(home_t.get("score", 0) or 0),
                    "away_score": int(away_t.get("score", 0) or 0),
                    "home_stats": hs,
                    "away_stats": as_,
                })
    games.sort(key=lambda g: g["date"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(games, indent=2))
    return games


def build_state(totals: dict, year: int) -> dict:
    """Mirror seed.seed()'s state assembly from a running totals dict."""
    team_stats = {k: seed._derive_advanced(t) for k, t in totals.items()}
    team_stats = {
        k: v for k, v in team_stats.items()
        if v and k in seed.REAL_WNBA_TEAMS
    }
    if not team_stats:
        return {}
    n = len(team_stats)
    return {
        "league_avg_ortg": round(sum(v["ortg"] for v in team_stats.values()) / n, 2),
        "league_avg_drtg": round(sum(v["drtg"] for v in team_stats.values()) / n, 2),
        "league_avg_pace": round(sum(v["pace"] for v in team_stats.values()) / n, 2),
        "teams": team_stats,
        # Make the agent's daily cache hit + pass state_is_stale_for_today:
        "fetched_at": date.today().isoformat(),
        "source_season": str(year),
    }


# ---------------------------------------------------------------------------
# Archived market loaders (Kalshi spread strikes + Pinnacle spread line)
# ---------------------------------------------------------------------------
def load_archive_spreads() -> dict:
    """(date_iso, yes_canonical) -> {strike: {entry, close}}.

    Keyed by the YES side because the archived team_home/team_away short codes
    don't normalize cleanly, but yes_team always does (ny->liberty, lv->aces).
    The game itself is recovered later by matching yes_canonical into the ESPN
    game-log for that date.

    `entry` = the FIRST archived YES ask for the contract (our scan/entry price);
    `close` = the LAST archived YES ask before the snapshot stream ends (the
    pre-tipoff close proxy — Kalshi never truly closes, T-30 is the convention).
    kalshi CLV = close - entry. Brier/EV pricing uses `close` to stay consistent
    with the totals loader's "last snapshot wins".
    """
    con = sqlite3.connect(ARCHIVE)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT yes_team, title, yes_price, date(event_date) AS d, fetched_at
        FROM archived_kalshi_markets
        WHERE sector='wnba' AND market_type='spread'
              AND yes_team IS NOT NULL AND title LIKE '%wins by over%'
        ORDER BY fetched_at ASC
        """
    ).fetchall()
    con.close()

    norm = NameNormalizer("wnba")
    out: dict = {}
    for r in rows:  # rows are ascending by fetched_at
        m = STRIKE_RE.search(r["title"] or "")
        if not m:
            continue
        strike = round(float(m.group(1)), 1)
        yes_c = norm.normalize(r["yes_team"])
        if not yes_c:
            continue
        px = float(r["yes_price"])
        rungs = out.setdefault((r["d"], yes_c), {})
        if strike not in rungs:
            rungs[strike] = {"entry": px, "close": px}  # first snapshot = entry
        else:
            rungs[strike]["close"] = px                 # keep overwriting => last = close
    return out


def load_pinnacle_spreads() -> dict:
    """(date_iso, frozenset(teams)) -> {a, b, line, prob_a}.

    `line` is Pinnacle's spread from outcome_a's perspective (negative = a is the
    favourite); prob_a is the devigged probability that a covers that line.
    """
    con = sqlite3.connect(ARCHIVE)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT event_id, outcome_a_label, outcome_b_label, spread_line,
               true_prob_a, date(event_date) AS d
        FROM archived_sharp_odds
        WHERE sector='wnba' AND spread_line IS NOT NULL AND true_prob_a IS NOT NULL
        """
    ).fetchall()
    con.close()

    norm = NameNormalizer("wnba")
    out: dict = {}
    for r in rows:
        a = norm.normalize(r["outcome_a_label"] or "")
        b = norm.normalize(r["outcome_b_label"] or "")
        if not a or not b:
            continue
        key = (r["d"], frozenset({a, b}))
        out[key] = {
            "a": a, "b": b,
            "line": round(float(r["spread_line"]), 1),
            "prob_a": float(r["true_prob_a"]),
        }
    return out


def _lookup_pm1(d_iso: str, teams, table: dict):
    """Look up a (date, key) entry tolerating a ±1-day ESPN/Kalshi shift."""
    k = (d_iso, teams)
    if k in table:
        return table[k]
    d = date.fromisoformat(d_iso)
    for off in (-1, 1):
        kk = ((d + timedelta(days=off)).isoformat(), teams)
        if kk in table:
            return table[kk]
    return None


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def main() -> None:
    arch = load_archive_spreads()
    pinn = load_pinnacle_spreads()
    games = fetch_game_log()
    print(f"game log={len(games)}  archive yes-sides={len(arch)}  "
          f"pinnacle games={len(pinn)}")

    norm = NameNormalizer("wnba")
    sim = WNBAPossessionSimAgent()
    spread_model = SpreadDistributionModel()
    totals: dict = {}  # canonical key -> seed.TeamSeasonTotals

    # full ladder: every archived strike (correlated within a game)
    m_full, k_full, y_full = [], [], []
    # main line: one strike per game — the Kalshi contract whose YES price is
    # nearest 0.5 (the market's pick'em line), chosen model-independently.
    m_main, k_main, y_main = [], [], []
    # pinnacle head-to-head at Pinnacle's own line
    p_model, p_pin, y_pin = [], [], []

    # PRODUCTION-BLEND check: the actual live spread price is
    #   SpreadDistributionModel.predict (Pinnacle CDF re-priced at the strike)
    #   blended with sim.cover_probability at sim_weight.
    # OLD = w=0.35 (previous default); NEW = w=0.0 (shipped). Only the strikes the
    # live SpreadDistributionModel would actually price (within 1σ of Pinnacle's
    # line, Pinnacle anchor present) enter here — exactly the live universe.
    po_full, pn_full, kpf_full, ypf_full = [], [], [], []   # full ladder
    sim_full: list[float] = []                               # aligned sim cover prob
    po_main, pn_main, kp_main, ypf_main = [], [], [], []     # main line

    # CLV AUDIT: for every strike the live path would BET (+EV ≥ 2% at the entry
    # price under a given pricing regime), record one row. This separates "did we
    # get a good price" (CLV) from "were our probs accurate" (Brier).
    #   kalshi_clv = kalshi_close - kalshi_entry   (real CLV — Kalshi's own move)
    #   pinn_drift = blended_entry_prob - kalshi_entry  (edge we bet on; the repo
    #                warns this is positive BY CONSTRUCTION — selection, not edge)
    EV_GATE = 0.02
    bets = {"NEW": [], "OLD": []}  # each row: dict(clv, drift, went, entry, p, strike, is_main)
    all_clv: list[float] = []   # CONTROL: kalshi CLV over EVERY priced strike
    fade_clv: list[float] = []  # CONTROL: strikes we'd FADE (NEW EV <= -2%)

    n_seen = predicted = gated = no_lines = 0

    for g in games:
        n_seen += 1
        d_iso = g["date"]
        year = int(d_iso[:4])
        home = norm.normalize(g["home_name"])
        away = norm.normalize(g["away_name"])
        margin_home = g["home_score"] - g["away_score"]  # +ve => home won

        state = build_state(totals, year)
        teams_fs = frozenset({home, away})

        if state and home and away:
            sim._efficiency_data = state  # inject as-of-date, leak-free state
            eid = f"wf_{home}_{away}_{d_iso}"
            s = sim.simulate_matchup(home, away, event_id=eid)
            if s is None:
                gated += 1  # < MIN_GAMES or team missing from state
            else:
                # Pinnacle spread anchor for this game (drives the production
                # market price + blend, exactly as the live path requires one).
                pin = _lookup_pm1(d_iso, teams_fs, pinn)
                sharp = None
                if pin is not None and pin["a"] in teams_fs and pin["b"] in teams_fs:
                    sharp = SharpOdds(
                        event_id=eid, book=SharpBook.pinnacle, sector="wnba",
                        outcome_a_label=pin["a"], outcome_b_label=pin["b"],
                        outcome_a_decimal=1.9, outcome_b_decimal=1.9,
                        true_prob_a=pin["prob_a"], true_prob_b=1.0 - pin["prob_a"],
                        spread_line=pin["line"],
                    )

                # ----- Kalshi strikes on both YES sides -----
                game_rungs: list[dict] = []
                for yes_c, is_home in ((home, True), (away, False)):
                    rungs = _lookup_pm1(d_iso, yes_c, arch)
                    if not rungs:
                        continue
                    yes_margin = margin_home if is_home else -margin_home
                    yes_is_b = (sharp is not None and yes_c == pin["b"])
                    for strike, px in rungs.items():
                        kalshi_p = px["close"]   # close = Brier/EV pricing baseline
                        entry_p = px["entry"]    # first snapshot = our entry price
                        # SAME call as the live EV-gap spread path:
                        mp = sim.cover_probability(
                            eid, -strike, yes_is_underdog=(not is_home),
                        )
                        if mp is None:
                            continue
                        went = yes_margin > strike
                        m_full.append(mp); k_full.append(kalshi_p); y_full.append(went)

                        # Production blend at OLD (0.35) vs NEW (0.0) sim weight,
                        # using the real SpreadDistributionModel as the anchor.
                        prod_old = prod_new = None
                        if sharp is not None:
                            sp = spread_model.predict(
                                sharp_odds=sharp, target_line=-strike,
                                sector="wnba", yes_is_underdog=yes_is_b,
                            )
                            if sp is not None:
                                prod_new = sp.true_prob                      # w=0 (shipped)
                                prod_old = 0.65 * sp.true_prob + 0.35 * mp   # w=0.35 (old)
                                po_full.append(prod_old); pn_full.append(prod_new)
                                kpf_full.append(kalshi_p); ypf_full.append(went)
                                sim_full.append(mp)  # aligned sim cover prob

                                # --- CLV AUDIT: would we BET this at entry? ---
                                # EV at entry = blended_true_prob / entry_price - 1
                                # (YES costs entry_price, pays 1). Gate at +2%.
                                if entry_p > 0:
                                    kalshi_clv = px["close"] - entry_p
                                    all_clv.append(kalshi_clv)  # control universe
                                    if prod_new / entry_p - 1.0 <= -EV_GATE:
                                        fade_clv.append(kalshi_clv)
                                    for regime, prob in (("NEW", prod_new), ("OLD", prod_old)):
                                        ev = prob / entry_p - 1.0
                                        if ev >= EV_GATE:
                                            bets[regime].append({
                                                "clv": kalshi_clv,
                                                "drift": prob - entry_p,
                                                "went": went, "entry": entry_p,
                                                "prob": prob, "strike": strike,
                                            })
                        game_rungs.append({
                            "sim": mp, "k": kalshi_p, "went": went,
                            "po": prod_old, "pn": prod_new,
                        })

                if game_rungs:
                    predicted += 1
                    # main line = market pick'em (kalshi price nearest 0.5)
                    best = min(game_rungs, key=lambda r: abs(r["k"] - 0.5))
                    m_main.append(best["sim"]); k_main.append(best["k"]); y_main.append(best["went"])
                    if best["pn"] is not None:
                        po_main.append(best["po"]); pn_main.append(best["pn"])
                        kp_main.append(best["k"]); ypf_main.append(best["went"])
                else:
                    no_lines += 1

                # ----- Pinnacle head-to-head -----
                if pin is not None:
                    a_is_home = (pin["a"] == home)
                    if pin["a"] in teams_fs and pin["b"] in teams_fs:
                        line = pin["line"]                 # a-perspective spread
                        strike = -line                     # a covers iff margin_a > strike
                        mp = sim.cover_probability(
                            eid, -strike, yes_is_underdog=(not a_is_home),
                        )
                        if mp is not None:
                            margin_a = margin_home if a_is_home else -margin_home
                            went = margin_a > strike
                            p_model.append(mp); p_pin.append(pin["prob_a"]); y_pin.append(went)
        elif not state:
            no_lines += 0  # early season: no state yet

        # ----- OBSERVE: fold this game into running totals -----
        seed._accumulate_game(
            totals,
            g["home_name"], g["home_stats"], g["home_score"],
            g["away_name"], g["away_stats"], g["away_score"],
        )

    print(f"\ngames={n_seen}  predicted={predicted}  "
          f"gated(<{MIN_GAMES}gp/missing)={gated}  σ={SPREAD_SIGMA}")

    def report(tag, model, market, actual, market_name="kalshi"):
        if not model:
            print(f"\n[{tag}] no data"); return
        rate = sum(actual) / len(actual)
        print(f"\n=== {tag} (n={len(model)}, cover-rate={rate:.3f}) ===")
        print(f"  model        Brier={brier_score(model, actual):.4f}  "
              f"acc={accuracy_score(model, actual):.3f}  mean_p={sum(model)/len(model):.3f}")
        print(f"  {market_name:<12} Brier={brier_score(market, actual):.4f}  "
              f"acc={accuracy_score(market, actual):.3f}  mean_p={sum(market)/len(market):.3f}")

    report("FULL LADDER — all archived strikes", m_full, k_full, y_full)
    report("MAIN LINE — one per game (market pick'em)", m_main, k_main, y_main)
    report("PINNACLE LINE — head-to-head", p_model, p_pin, y_pin, market_name="pinnacle")

    # ----- Blend sweep: how much sim should the live path mix into the market? -----
    # Live ev_gap_agent blends sharp_true_prob (market-anchored) with the sim's
    # cover_probability at sim_weight=0.35. Sweep that weight to find the Brier-
    # optimal mix. w=0 is pure market; w=1 is pure sim.
    def blend_sweep(tag, model, market, actual):
        if not model:
            return
        print(f"\n=== BLEND SWEEP {tag} (n={len(model)}) — Brier at sim weight w ===")
        best_w, best_b = 0.0, 1.0
        for w in (0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 1.0):
            blended = [w * mm + (1 - w) * kk for mm, kk in zip(model, market)]
            b = brier_score(blended, actual)
            mark = ""
            if b < best_b:
                best_b, best_w = b, w
            if abs(w - 0.35) < 1e-9:
                mark = "  <- live default"
            print(f"  w={w:.2f}  Brier={b:.4f}{mark}")
        print(f"  optimal w={best_w:.2f}  Brier={best_b:.4f}")

    blend_sweep("vs PINNACLE", p_model, p_pin, y_pin)
    blend_sweep("vs KALSHI (main line)", m_main, k_main, y_main)

    # ----- Production-blend A/B: OLD (sim_weight 0.35) vs NEW (0.0 = shipped) -----
    # This is the literal live price (SpreadDistributionModel anchor + sim blend),
    # not a proxy. NEW == pure market anchor.
    def ab(tag, old, new, kalshi, actual):
        if not new:
            print(f"\n[PROD BLEND {tag}] no data"); return
        rate = sum(actual) / len(actual)
        # Climatology = always predict the realized base rate (no information).
        # Brier reduces to r(1-r); the Brier Skill Score is 1 - Brier/clim.
        clim = rate * (1.0 - rate)
        def bss(probs):
            return 1.0 - brier_score(probs, actual) / clim if clim else 0.0
        print(f"\n=== PROD BLEND {tag} (n={len(new)}, cover-rate={rate:.3f}) ===")
        print(f"  climatology  Brier={clim:.4f}  (predict {rate:.3f} every time — zero skill)")
        print(f"  OLD w=0.35   Brier={brier_score(old, actual):.4f}  "
              f"acc={accuracy_score(old, actual):.3f}  BSS={bss(old):+.3f}")
        print(f"  NEW w=0.00   Brier={brier_score(new, actual):.4f}  "
              f"acc={accuracy_score(new, actual):.3f}  BSS={bss(new):+.3f}   (shipped)")
        print(f"  kalshi       Brier={brier_score(kalshi, actual):.4f}  "
              f"acc={accuracy_score(kalshi, actual):.3f}  BSS={bss(kalshi):+.3f}")
        d = brier_score(old, actual) - brier_score(new, actual)
        print(f"  ΔBrier (OLD−NEW) = {d:+.4f}  "
              f"({'NEW better' if d > 0 else 'OLD better' if d < 0 else 'tie'})")

    ab("FULL LADDER", po_full, pn_full, kpf_full, ypf_full)
    ab("MAIN LINE", po_main, pn_main, kp_main, ypf_main)

    # -------------------------------------------------------------------------
    # BIAS vs NOISE — is the sim wrong because it's MISCALIBRATED (a fixable
    # systematic shift) or because it carries NO incremental signal (drop it)?
    # -------------------------------------------------------------------------
    def diagnose(sim, mkt, y):
        import numpy as np
        if len(sim) < 10:
            print("\n[BIAS/NOISE] too few rows"); return
        sim = np.array(sim, float); mkt = np.array(mkt, float); y = np.array(y, float)
        # 1) BIAS: how far each estimator's mean prob sits from the realized rate.
        print(f"\n=== BIAS vs NOISE (n={len(y)}, realized cover={y.mean():.3f}) ===")
        print(f"  mean prob:  sim={sim.mean():.3f}  market={mkt.mean():.3f}  "
              f"realized={y.mean():.3f}")
        print(f"  bias:       sim={sim.mean()-y.mean():+.3f}  "
              f"market={mkt.mean()-y.mean():+.3f}   (sim systematically off?)")
        # 2) DISCRIMINATION: correlation of each prob with the realized outcome.
        def corr(a, b):
            if a.std() == 0 or b.std() == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        print(f"  corr w/ outcome:  sim={corr(sim, y):+.3f}  market={corr(mkt, y):+.3f}")
        # 3) INCREMENTAL SIGNAL: does the sim explain the market's RESIDUAL?
        #    resid = realized − market_prob. If the sim has info the market lacks,
        #    sim should correlate with resid. ~0 ⇒ pure noise relative to market.
        resid = y - mkt
        print(f"  corr(sim, market residual) = {corr(sim, resid):+.3f}   "
              f"<- ~0 ⇒ no info beyond market; <0 ⇒ actively misleading")
        # 4) DEBIAS TEST: recenter the sim to be bias-free, then re-blend. If a
        #    debiased sim STILL doesn't beat market-alone, the failure is noise,
        #    not bias (in-sample debias = best case for the sim, so this is a
        #    generous test).
        sim_db = np.clip(sim - (sim.mean() - y.mean()), 0.01, 0.99)
        from evmax.backtest.metrics import brier_score as bs
        b_mkt = bs(list(mkt), list(y))
        b_blend_raw = bs(list(0.65 * mkt + 0.35 * sim), list(y))
        b_blend_db = bs(list(0.65 * mkt + 0.35 * sim_db), list(y))
        print(f"  Brier:  market-alone={b_mkt:.4f}   "
              f"+raw sim(.35)={b_blend_raw:.4f}   +DEBIASED sim(.35)={b_blend_db:.4f}")
        verdict = ("bias is the whole story — a debiased sim helps"
                   if b_blend_db < b_mkt - 1e-4 else
                   "noise too — even a debiased (best-case) sim does NOT beat market")
        print(f"  -> {verdict}")

    diagnose(sim_full, pn_full, ypf_full)

    # -------------------------------------------------------------------------
    # CLV AUDIT — "good price" (CLV) vs "accurate prob" (Brier) on the bet set
    # -------------------------------------------------------------------------
    def clv_report(tag, rows):
        if not rows:
            print(f"\n[CLV {tag}] no qualifying bets"); return
        n = len(rows)
        clv = [r["clv"] for r in rows]
        drift = [r["drift"] for r in rows]
        won = [r["went"] for r in rows]
        probs = [r["prob"] for r in rows]
        mean_clv = sum(clv) / n
        pct_pos = sum(1 for c in clv if c > 0) / n
        mean_drift = sum(drift) / n
        # Brier of the bets we placed, against realized cover:
        brier = brier_score(probs, won)
        win = sum(won) / n
        print(f"\n=== CLV {tag} (n={n} bets @ EV≥{EV_GATE:.0%}) ===")
        print(f"  kalshi CLV    mean={mean_clv*100:+.2f}pp  %positive={pct_pos:.0%}   "
              f"<- REAL signal (Kalshi entry->close)")
        print(f"  pinn drift    mean={mean_drift*100:+.2f}pp   "
              f"<- positive by construction (selection, NOT edge)")
        print(f"  outcome       win-rate={win:.3f}   bet-prob mean={sum(probs)/n:.3f}")
        print(f"  bet Brier     {brier:.4f}  (vs realized cover of the bets placed)")

    clv_report("NEW pricing (w=0, shipped)", bets["NEW"])
    clv_report("OLD pricing (w=0.35)", bets["OLD"])

    # FALSIFICATION: if our +EV bets show CLV but the control universe doesn't,
    # the CLV comes from SELECTION (real), not a market-wide YES-side drift.
    def _m(xs):
        return (sum(xs) / len(xs) * 100) if xs else 0.0
    sel = [r["clv"] for r in bets["NEW"]]
    print(f"\n=== CLV CONTROL (falsification) ===")
    print(f"  bet set (+EV≥2%)   n={len(sel):<4} mean CLV={_m(sel):+.2f}pp")
    print(f"  ALL priced strikes n={len(all_clv):<4} mean CLV={_m(all_clv):+.2f}pp  "
          f"<- should be ~0 if edge is selection")
    print(f"  FADE set (EV≤−2%)  n={len(fade_clv):<4} mean CLV={_m(fade_clv):+.2f}pp  "
          f"<- should be NEGATIVE (mirror)")

    # Where does the CLV live — contested (entry 0.35-0.65) vs lopsided strikes?
    def split_clv(tag, rows):
        if not rows:
            return
        con = [r for r in rows if 0.35 <= r["entry"] <= 0.65]
        lop = [r for r in rows if not (0.35 <= r["entry"] <= 0.65)]
        def line(label, rs):
            if not rs:
                print(f"    {label:<22} n=0"); return
            mc = sum(r["clv"] for r in rs) / len(rs)
            pb = brier_score([r["prob"] for r in rs], [r["went"] for r in rs])
            print(f"    {label:<22} n={len(rs):<3} CLV={mc*100:+.2f}pp  "
                  f"Brier={pb:.4f}  win={sum(r['went'] for r in rs)/len(rs):.3f}")
        print(f"\n  -- {tag}: CLV by strike type --")
        line("contested (.35-.65)", con)
        line("lopsided", lop)

    split_clv("NEW pricing", bets["NEW"])

    print("\n=== model calibration (full ladder) ===")
    for b in calibration_bins(m_full, y_full):
        if b.n:
            print(f"  p[{b.prob_low:.1f}-{b.prob_high:.1f}]  pred={b.mean_pred:.3f}  "
                  f"emp={b.actual_rate:.3f}  n={b.n}")


if __name__ == "__main__":
    main()
