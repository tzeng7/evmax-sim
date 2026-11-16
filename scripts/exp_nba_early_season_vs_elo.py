"""NBA early season: recency-weighted ridge rating vs the production Elo (standalone).

The shipped NBA stat stack abstains until each team has 20 games, so in the
opening weeks the blend is Elo + sharp. scripts/exp_basketball_recency.py shows the
ridge rating prices those games far better than a constant home rate; this lens
asks the live question: does it beat (or usefully add to) the Elo that is
actually there?

Elo side = the PRODUCTION `EloModelAgent` (K=20, home +100, MOV/SOS multipliers,
rest layer, H2H off like live) replayed over the same game table with the NBA
offseason regression (keep=0.50) applied at each season boundary. This is a
BEST-CASE Elo: the live NBA state is thinner and roster-blind, so a ridge win
here understates the live gap.

Early = the game's less-informed team has played < OPEN_GAMES games that season.
All variants are scored on identical games. Ridge goes through the usual logistic
map fitted on earlier seasons; Elo is shown raw (as live) and recalibrated.

Usage:
    python scripts/exp_nba_early_season_vs_elo.py [--holdout] [--hl 60] [--lam 300]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent))
sys.path.insert(0, str(_SCRIPTS))

import backtest_elo_h2h_rest as h2h  # noqa: E402
from evmax.agents.models.elo_agent import DEFAULT_ELO  # noqa: E402
import exp_basketball_recency as rec  # noqa: E402

OPEN_GAMES = 20
KEEP = 0.50
SECTOR = "nba"


def replay_elo(G):
    """Per-game production-Elo P(home) (pre-game) and the less-informed team's season game count."""
    teams = {}
    inv = {}
    # team index -> abbreviation is not stored; rebuild a stable name from the index
    names = [f"t{i}" for i in range(G["n_teams"])]
    tracker = h2h.ScheduleTracker()
    agent = h2h.make_replay_agent(tracker)
    agent.use_h2h = False
    agent.use_rest = True
    p = np.full(G["n"], np.nan)
    min_gp = np.zeros(G["n"])
    gp = defaultdict(int)
    cur = None
    for i in range(G["n"]):
        s = int(G["season"][i])
        if s != cur:
            if cur is not None:
                st = agent._sector_state(SECTOR)
                for t, r in list(st["ratings"].items()):
                    st["ratings"][t] = round(DEFAULT_ELO + KEEP * (r - DEFAULT_ELO), 2)
                    st["season_games"][t] = 0
            gp.clear()
            cur = s
        d = date.fromordinal(int(G["ord"][i]))
        h, a = names[G["h"][i]], names[G["a"][i]]
        h2h._ReplayClock._now = d
        p[i], _, _ = agent._win_probs(SECTOR, h, a, d)
        min_gp[i] = min(gp[h], gp[a])
        agent.update(h, a, G["hp"][i], G["ap"][i], SECTOR, event_date=d.isoformat())
        tracker.record(h, d)
        tracker.record(a, d)
        gp[h] += 1
        gp[a] += 1
    return p, min_gp


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--hl", type=float, default=60.0)
    ap.add_argument("--lam", type=float, default=300.0)
    args = ap.parse_args()

    cfg = rec.SPORTS["nba"]
    G = rec.load_nba()
    seasons = cfg["rank"] + cfg["confirm"] + (cfg["holdout"] if args.holdout else ())
    score_seasons = tuple(range(cfg["fit_start"], max(seasons) + 1))
    print(f"replaying production Elo over {G['n']} games ...", flush=True)
    p_elo, min_gp = replay_elo(G)
    dl_ridge = rec.deltas(G, "ridge", dict(lam=args.lam, hl=args.hl), cfg, score_seasons)
    dl_ew = rec.deltas(G, "ew", dict(hl=90.0), cfg, score_seasons)

    y = G["home_win"]
    s = G["season"]
    ok = np.isin(s, list(score_seasons)) & ~np.isnan(dl_ridge) & ~np.isnan(dl_ew) & ~np.isnan(p_elo) & (min_gp >= 4)
    early_all = ok & (min_gp < OPEN_GAMES)
    late_all = ok & (min_gp >= OPEN_GAMES)

    def brier_by_season(mask_fn):
        out = defaultdict(dict)
        for ev in seasons:
            tr = np.where(ok & (s < ev))[0]
            te = np.where(mask_fn & (s == ev))[0]
            if te.size == 0 or tr.size < 500:
                continue
            a_r, b_r = rec.fit_logistic(dl_ridge[tr], y[tr])
            a_w, b_w = rec.fit_logistic(dl_ew[tr], y[tr])
            a_e, b_e = rec.fit_logistic(logit(p_elo[tr]), y[tr])
            hr = y[tr].mean()
            P = {
                "const": np.full(te.size, hr),
                "elo_raw": p_elo[te],
                "elo_recal": 1 / (1 + np.exp(-np.clip(a_e * logit(p_elo[te]) + b_e, -30, 30))),
                "ew90": 1 / (1 + np.exp(-np.clip(a_w * dl_ew[te] + b_w, -30, 30))),
                "ridge": 1 / (1 + np.exp(-np.clip(a_r * dl_ridge[te] + b_r, -30, 30))),
            }
            P["blend_ridge_elo"] = 0.5 * P["ridge"] + 0.5 * P["elo_raw"]
            for k, v in P.items():
                out[k][ev] = (v - y[te]) ** 2
        return out

    def report(title, mask, groups):
        res = brier_by_season(mask)
        print(f"\n{title}")
        names = ["const", "elo_raw", "elo_recal", "ew90", "ridge", "blend_ridge_elo"]
        for gname, gs in groups.items():
            cols = {k: np.concatenate([res[k][e] for e in gs if e in res[k]]) for k in names if any(e in res[k] for e in gs)}
            if not cols:
                continue
            n = len(next(iter(cols.values())))
            line = f"  {gname:8s} n={n:5d} " + " ".join(f"{k}={v.mean():.4f}" for k, v in cols.items())
            for ref in ("elo_raw",):
                for k in ("ridge", "blend_ridge_elo"):
                    d = cols[k] - cols[ref]
                    line += f" | {k}-{ref}={1000 * d.mean():+.2f}/1000 z={d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}"
            print(line)

    groups = {"rank": cfg["rank"], "confirm": cfg["confirm"]}
    if args.holdout:
        groups["holdout"] = cfg["holdout"]
    report(f"EARLY season (less-informed team < {OPEN_GAMES} games)", early_all, groups)
    report(f"LATER season (>= {OPEN_GAMES} games) - shipped stack is live here", late_all, groups)


if __name__ == "__main__":
    main()
