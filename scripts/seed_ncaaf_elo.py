#!/usr/bin/env python3
"""Warm-seed generic Elo for NCAAF with a regressed carry-forward prior.

WHY THIS EXISTS
---------------
Generic `elo` carries ensemble weight 0.25 for ncaaf, but the sector has never
had an `elo_state.json["ncaaf"]` entry — it started every season from an EMPTY
state (flat 1500). Two costs follow:

  1. Cold start = DARK early. `EloModelAgent.predict_pair` gates a team below the
     ensemble's 0.45 confidence gate until it has >=5 lifetime games (confidence
     0.30 at 0 games, 0.45 at 1-4 games, both <= 0.45 → dropped). So for the
     first ~5 games of every team, generic elo contributes NOTHING, and the
     early-season NCAAF blend collapses to ncaaf_efficiency + sharp. The strong
     CFB-calibrated elo (MODEL-2: K=40 / home_adv=60, holdout Brier 0.1836) does
     not enter the blend until ~week 5.
  2. A cold start also THROWS AWAY durable team quality. Alabama and a MAC
     bottom-feeder both start at 1500 in week 1.

The standard fix (FiveThirtyEight/SP+/FPI preseason priors): take each team's
final Elo from the prior season and REGRESS it toward the league mean, so the
preseason rating keeps the durable part of team strength and sheds the noise /
roster-churn part.

REGRESSION FACTOR — empirically tuned (default 0.90, LIGHT), not the textbook
1/3. `--validate` walk-forward on three held-out seasons (2023/2024/2025) shows
the FBS-AGGREGATE Brier improves monotonically as regression LIGHTENS: f=0.5
over-regresses and loses to straight carry-through in every holdout, and the
early-season (weeks 0-3) curve on 2025 is 0.1811 (f=0.5) → 0.1689 (0.75) →
0.1626 (0.90) → 0.1596 (1.0). The reason is that roster churn is HETEROGENEOUS
— it guts a few programs but barely touches durable blue-bloods — so a UNIFORM
heavy pull punishes the many stable teams (whose prior is predictive) to protect
the few churned ones, and the net Brier is worse. A uniform regression is the
wrong tool for churn; the team-specific returning-production/recruiting prior is
(a deferred model-side gap). f=0.90 sits within Brier-noise of the holdout
optimum (f=1.0) while keeping a token 10% pull as insurance against these three
holdouts all being "prior-persisted" years, and it complements the efficiency
model, which already carries a 0.5-regressed roster-blind prior at weight 0.55.

WHAT THIS SCRIPT DOES
---------------------
Walk-forward feeds every completed prior season (default 2021-2025) through the
PRODUCTION-FAITHFUL bare-K Elo formula (K=40 / home_adv=60, the exact RawElo the
MODEL-2 sweep in scripts/backtest_ncaaf_elo.py validated), regressing each
team's rating toward 1500 at every season boundary INCLUDING the final one into
the upcoming season. The result is written to elo_state.json["ncaaf"] as a fresh
preseason prior, keyed by NameNormalizer("ncaaf") so warm-seed keys match both
what predict_pair looks up and what the resolve-time model-update hook writes
live (they converge on the same keys instead of forking duplicates).

Bare-K (no MOV/SOS/recency multipliers) is deliberate: those constants were
never calibrated for ncaaf, recency-K decay would collapse a batch historical
feed toward flat 1500, and the K=40/home_adv=60 holdout number was measured on
exactly this bare formula. The live agent then continues FORWARD from this seed
with its full multiplier stack; the seed only needs to be a well-scaled,
correctly-keyed warm starting point.

State written (mirrors EloModelAgent._sector_state shape):
  ratings       : {normalized_team: regressed_elo}
  game_counts   : {normalized_team: lifetime_games}   # feeds the confidence gate
  season_games  : {}                                   # fresh upcoming season → early-K boost fires live
  h2h           : {}                                   # built live
  last_updated  : seed run date (ISO)                  # fresh → staleness guard passes

USAGE
-----
    python scripts/seed_ncaaf_elo.py                      # seed 2021-2025 → elo_state.json
    python scripts/seed_ncaaf_elo.py --dry-run            # preview, write nothing
    python scripts/seed_ncaaf_elo.py --seasons 2021,2022,2023,2024,2025
    python scripts/seed_ncaaf_elo.py --regress 0.5
    python scripts/seed_ncaaf_elo.py --validate --holdout 2025   # warm-vs-cold walk-forward

Re-run weekly in season (like scripts/seed_ncaaf_efficiency.py) so last_updated
stays fresh; the live resolve hook also advances it as games resolve.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from evmax.agents.models.elo_agent import (  # noqa: E402
    DEFAULT_ELO,
    HOME_ADVANTAGE_ELO,
    K_FACTORS,
    LOW_DATA_THRESHOLD,
)
from evmax.clients import cfb_espn as C  # noqa: E402
from evmax.matching.normalizer import NameNormalizer  # noqa: E402
from scripts.backtest_ncaaf_elo import RawElo, brier, fetch_season  # noqa: E402

SECTOR = "ncaaf"
STATE_PATH = _REPO / "data" / "models" / "elo_state.json"
PROD_K = K_FACTORS["ncaaf"]              # 40.0  (MODEL-2, 2026-08-07)
PROD_HOME = HOME_ADVANTAGE_ELO["ncaaf"]  # 60.0  (MODEL-2)
MEAN = DEFAULT_ELO                        # 1500.0

# Week buckets mirror scripts/backtest_ncaaf_efficiency.py — the whole point of
# warm-seeding is what happens in the EARLY bucket, so validation reports per
# bucket, not just pooled.
BUCKETS = [("0-3", 1, 3), ("4-8", 4, 8), ("9+", 9, 99)]


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
def _normalize_games(games: list[dict], nz: NameNormalizer) -> list[dict]:
    """Rekey each game's home/away to the canonical NameNormalizer slug.

    Falls back to the raw lowercased location when the normalizer yields nothing
    (unmapped team) so the game still contributes rather than being dropped.
    """
    out: list[dict] = []
    for g in games:
        h = nz.normalize(g["home"]) or g["home"]
        a = nz.normalize(g["away"]) or g["away"]
        out.append({**g, "home": h, "away": a})
    return out


def regress_toward_mean(ratings: dict[str, float], factor: float, mean: float = MEAN) -> None:
    """In-place FiveThirtyEight-style regression: r <- mean + (r - mean) * factor.

    factor=1.0 is a no-op (pure carry-through); factor=0.0 wipes to the mean.
    """
    for k in list(ratings):
        ratings[k] = mean + (ratings[k] - mean) * factor


def build_seed(
    seasons: list[int],
    regress: float,
    between_season_regress: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, float], dict[str, int]]:
    """Walk prior seasons chronologically → (ratings, lifetime game_counts).

    Regresses toward the mean at every season boundary (when between_season_regress)
    plus once more after the final season — the preseason prior for the upcoming
    year. With between_season_regress=False only the final regression is applied
    (straight carry-through during warmup), which the validation harness uses as
    the "warm, no between-season regression" arm.
    """
    nz = NameNormalizer(SECTOR)
    elo = RawElo(k=PROD_K, home_adv=PROD_HOME)
    counts: dict[str, int] = defaultdict(int)
    for i, season in enumerate(seasons):
        if i > 0 and between_season_regress:
            regress_toward_mean(elo.ratings, regress)
        games = _normalize_games(fetch_season(season), nz)
        for g in games:
            elo.update(g["home"], g["away"], g["home_won"], g["neutral"])
            counts[g["home"]] += 1
            counts[g["away"]] += 1
        if verbose:
            print(
                f"  fed {season}: {len(games):>4} games · {len(elo.ratings):>3} teams",
                file=sys.stderr,
            )
    # Final preseason regression into the upcoming season.
    regress_toward_mean(elo.ratings, regress)
    return dict(elo.ratings), dict(counts)


# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------
def write_state(
    ratings: dict[str, float],
    counts: dict[str, int],
    stamp: str,
    dry_run: bool = False,
) -> None:
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    state[SECTOR] = {
        "ratings": {k: round(v, 2) for k, v in sorted(ratings.items())},
        "game_counts": {k: int(counts.get(k, 0)) for k in sorted(ratings)},
        "season_games": {},   # fresh upcoming season → live early-K boost
        "h2h": {},            # built live by the resolve hook
        "last_updated": stamp,
    }
    if dry_run:
        print("[dry-run] would write elo_state.json['ncaaf'] with "
              f"{len(ratings)} teams, last_updated={stamp}")
        return
    stamp_c = stamp.replace("-", "")
    backup = STATE_PATH.with_suffix(f".backup.ncaaf_elo_{stamp_c}.json")
    if STATE_PATH.exists():
        shutil.copy2(STATE_PATH, backup)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"wrote elo_state.json['ncaaf'] · {len(ratings)} teams · last_updated={stamp}")
    if STATE_PATH.exists() and backup.exists():
        print(f"backup saved at {backup.name}")


# ----------------------------------------------------------------------------
# Validation: warm vs cold on a held-out season
# ----------------------------------------------------------------------------
def _week_index(game_date: str, season_start: dt.date) -> int:
    try:
        d = dt.date.fromisoformat(game_date[:10])
    except Exception:  # noqa: BLE001
        return 1
    return max(1, (d - season_start).days // 7 + 1)


def _bucket(week: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= week <= hi:
            return name
    return "9+"


def _elo_confidence(min_count: int) -> float:
    """Replicate EloModelAgent.predict_pair confidence tiers exactly."""
    if min_count == 0:
        return 0.30
    if min_count < LOW_DATA_THRESHOLD:   # <5
        return 0.45
    return 0.60 if min_count < 15 else 0.80


def _score_holdout(
    seed_ratings: dict[str, float],
    seed_counts: dict[str, int],
    holdout_games: list[dict],
    season_start: dt.date,
) -> dict:
    """Walk-forward score one config over the holdout season.

    Each game: predict on PRE-game rating, record Brier + whether elo would clear
    the 0.45 gate (confidence > 0.45 ⇔ min lifetime count >= 5), then update.
    Returns per-bucket + pooled aggregates over the games where elo FIRES (the
    production-relevant sample — a gated elo contributes nothing to the blend).
    """
    ratings = dict(seed_ratings)
    counts: dict[str, int] = defaultdict(int, seed_counts)
    elo = RawElo(k=PROD_K, home_adv=PROD_HOME)
    elo.ratings = ratings

    # agg[bucket] = [sum_brier_fired, n_fired, n_total, correct_fired]
    agg: dict[str, list[float]] = {b: [0.0, 0, 0, 0] for b, _, _ in BUCKETS}
    for g in holdout_games:
        wk = _week_index(g["date"], season_start)
        b = _bucket(wk)
        min_count = min(counts[g["home"]], counts[g["away"]])
        fires = _elo_confidence(min_count) > 0.45
        p_home = elo.expected_home(g["home"], g["away"], g["neutral"])
        agg[b][2] += 1
        if fires:
            agg[b][0] += brier(p_home, g["home_won"])
            agg[b][1] += 1
            if (p_home > 0.5) == bool(g["home_won"]):
                agg[b][3] += 1
        elo.update(g["home"], g["away"], g["home_won"], g["neutral"])
        counts[g["home"]] += 1
        counts[g["away"]] += 1

    def _summ(rows: list[list[float]]) -> dict:
        sb = sum(r[0] for r in rows)
        nf = sum(r[1] for r in rows)
        nt = sum(r[2] for r in rows)
        cf = sum(r[3] for r in rows)
        return {
            "brier": (sb / nf) if nf else float("nan"),
            "acc": (cf / nf) if nf else float("nan"),
            "n_fired": nf,
            "n_total": nt,
            "fire_rate": (nf / nt) if nt else float("nan"),
        }

    out = {b: _summ([agg[b]]) for b, _, _ in BUCKETS}
    out["overall"] = _summ(list(agg.values()))
    return out


def run_validation(train_seasons: list[int], holdout: int, regress: float) -> None:
    print(f"\n=== NCAAF elo warm-seed validation · train {train_seasons} · holdout {holdout} ===\n")
    season_start = dt.date.fromisoformat(C.SEASON_WINDOWS[holdout][0])
    nz = NameNormalizer(SECTOR)
    holdout_games = _normalize_games(fetch_season(holdout), nz)
    if not holdout_games:
        print("ABORT: no holdout games fetched.", file=sys.stderr)
        return

    # Three configs, all scored on the SAME holdout with identical gate logic.
    cold_r, cold_c = {}, {}
    warm_nore_r, warm_nore_c = build_seed(train_seasons, regress=1.0,
                                          between_season_regress=False, verbose=False)
    warm_reg_r, warm_reg_c = build_seed(train_seasons, regress=regress,
                                        between_season_regress=True, verbose=False)

    configs = [
        ("COLD (empty 1500 — production today)", cold_r, cold_c),
        ("WARM · straight carry-through", warm_nore_r, warm_nore_c),
        (f"WARM · regressed carry-forward (f={regress})", warm_reg_r, warm_reg_c),
    ]
    results = [(name, _score_holdout(r, c, holdout_games, season_start)) for name, r, c in configs]

    hdr = f"{'config':<44}{'bucket':<8}{'Brier':>9}{'Acc':>8}{'fire%':>8}{'n_fire':>8}{'n_tot':>7}"
    for name, res in results:
        print(hdr if name == results[0][0] else "")
        for b in ["0-3", "4-8", "9+", "overall"]:
            s = res[b]
            fr = "" if s["fire_rate"] != s["fire_rate"] else f"{100*s['fire_rate']:.0f}%"
            br = "" if s["brier"] != s["brier"] else f"{s['brier']:.4f}"
            ac = "" if s["acc"] != s["acc"] else f"{100*s['acc']:.1f}%"
            tag = name if b == "0-3" else ""
            print(f"{tag:<44}{b:<8}{br:>9}{ac:>8}{fr:>8}{s['n_fired']:>8}{s['n_total']:>7}")
        print()

    early = {name: res["0-3"] for name, res in results}
    print("EARLY-SEASON (weeks 0-3) takeaway:")
    for name, s in early.items():
        fr = 0.0 if s["fire_rate"] != s["fire_rate"] else s["fire_rate"]
        br = s["brier"]
        brs = "n/a (all gated)" if br != br else f"Brier {br:.4f}"
        print(f"  {name:<44} elo fires on {100*fr:>3.0f}% of early games · {brs}")


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025",
                    help="prior seasons to walk (chronological)")
    ap.add_argument("--regress", type=float, default=0.90,
                    help="carry-forward regression factor toward 1500 (0=full wipe, 1=none); "
                         "default 0.90 — light, empirically tuned (see module docstring)")
    ap.add_argument("--no-between-season-regress", action="store_true",
                    help="regress only the final preseason boundary, not between warmup seasons")
    ap.add_argument("--dry-run", action="store_true", help="preview, write nothing")
    ap.add_argument("--validate", action="store_true",
                    help="warm-vs-cold walk-forward on a held-out season; writes nothing")
    ap.add_argument("--holdout", type=int, default=2025, help="validation holdout season")
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]

    if args.validate:
        train = [s for s in seasons if s < args.holdout] or seasons
        run_validation(train, args.holdout, args.regress)
        return 0

    print(f"Seeding ncaaf Elo · seasons {seasons} · K={PROD_K} home_adv={PROD_HOME} "
          f"· regress={args.regress}", file=sys.stderr)
    ratings, counts = build_seed(
        seasons, regress=args.regress,
        between_season_regress=not args.no_between_season_regress,
    )
    if not ratings:
        print("ABORT: no ratings built (empty fetch) — refusing to overwrite state.",
              file=sys.stderr)
        return 1

    lo, hi = min(ratings.values()), max(ratings.values())
    print(f"built {len(ratings)} teams · rating spread [{lo:.0f}, {hi:.0f}] "
          f"· mean {sum(ratings.values())/len(ratings):.0f}", file=sys.stderr)

    stamp = dt.date.today().isoformat()
    write_state(ratings, counts, stamp, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
