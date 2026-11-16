"""Walk-forward RECENCY lens for the NBA and WNBA efficiency stacks (standalone).

Question: does weighting games by recency (and carrying last season across the
offseason) rate teams better than the shipped season-to-date average?

Shipped rating (the thing being tested against):
  NBA   evmax/agents/models/efficiency_agent.py : season-to-date ORTG/DRTG from
        nba_api LeagueDashTeamStats, abstains until each team has 20 games.
  WNBA  evmax/agents/models/wnba_efficiency_agent.py : season-to-date Dean Oliver
        ORTG/DRTG, empirical-Bayes shrunk to the league mean (k=8), min 4 games.
  Both: plain season-to-date ratio-of-sums, no recency, no opponent adjustment,
  no prior-season carry.

Protocol (same as scripts/exp_nfl_eff_ridge.py):
  * Ratings for a game use only games with date < that game's date.
  * Every variant goes through the SAME 2-parameter logistic map
    P(home) = sigmoid(a*delta + b) fitted ONLY on earlier seasons (expanding),
    so the metric compares rating QUALITY, not a tuned constant.
  * Choose hyperparameters on RANK seasons, confirm on CONFIRM, score HOLDOUT once
    (--holdout).
  * Primary table = games where EVERY variant produced a rating (so the shipped
    rating's 20-game NBA gate is respected). Secondary table = the early-season
    games the shipped rating cannot price, scored against a constant home-rate.

Usage:
    python scripts/exp_basketball_recency.py --sport nba
    python scripts/exp_basketball_recency.py --sport wnba --holdout
    python scripts/exp_basketball_recency.py --sport nba --variants "base,ridge:lam=300/hl=120"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

CACHE = _REPO_ROOT / "data" / "cache" / "basketball_recency"

SPORTS = {
    # seasons are start-year labels (2024 == 2024-25 for the NBA, calendar year for the WNBA)
    "nba": dict(load=range(2013, 2026), fit_start=2016, rank=(2019, 2020, 2021), confirm=(2022, 2023),
                holdout=(2024, 2025), min_gp=20, shrink_k=0.0),
    "wnba": dict(load=range(2018, 2027), fit_start=2020, rank=(2022, 2023), confirm=(2024,),
                 holdout=(2025, 2026), min_gp=4, shrink_k=8.0),
}


# ---------------------------------------------------------------- data

# ESPN abbreviations of real franchises (the scoreboard also carries All-Star / exhibition sides).
WNBA_FRANCHISES = {"ATL", "CHI", "CON", "DAL", "IND", "LA", "LV", "MIN", "NY", "PHX", "SEA", "WSH", "GS", "TOR", "POR"}

def _poss(fga, fta, oreb, tov):
    return fga - oreb + tov + 0.44 * fta


def load_nba():
    from nba_api.stats.endpoints import leaguegamelog

    CACHE.mkdir(parents=True, exist_ok=True)
    games = []
    for yr in SPORTS["nba"]["load"]:
        f = CACHE / f"nba_{yr}.json"
        if f.exists():
            rows = json.loads(f.read_text())
        else:
            lab = f"{yr}-{str(yr + 1)[2:]}"
            df = leaguegamelog.LeagueGameLog(season=lab, season_type_all_star="Regular Season",
                                             player_or_team_abbreviation="T", timeout=60).get_data_frames()[0]
            rows = df[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS", "FGA", "FTA", "OREB", "TOV"]].to_dict("records")
            f.write_text(json.dumps(rows))
            time.sleep(0.6)
        by = {}
        for r in rows:
            by.setdefault(r["GAME_ID"], []).append(r)
        for gid, two in by.items():
            if len(two) != 2:
                continue
            h = next((r for r in two if " vs. " in r["MATCHUP"]), None)
            a = next((r for r in two if " @ " in r["MATCHUP"]), None)
            if h is None or a is None:
                continue
            games.append(_game(yr, h["GAME_DATE"], h["TEAM_ABBREVIATION"], a["TEAM_ABBREVIATION"], h, a))
    return _finish(games)


def load_wnba():
    import httpx

    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.seed_wnba_efficiency import ESPN_BASE, _extract_team_stats

    CACHE.mkdir(parents=True, exist_ok=True)
    games = []
    with httpx.Client() as client:
        for yr in SPORTS["wnba"]["load"]:
            f = CACHE / f"wnba_{yr}.json"
            if f.exists():
                rows = json.loads(f.read_text())
            else:
                rows = []
                for m in range(5, 11):
                    try:
                        r = client.get(f"{ESPN_BASE}/scoreboard", params={"dates": f"{yr}{m:02d}", "limit": 200}, timeout=30)
                        r.raise_for_status()
                        events = r.json().get("events", [])
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {yr}-{m:02d} scoreboard failed: {exc}", file=sys.stderr)
                        continue
                    for e in events:
                        comp = e["competitions"][0]
                        if comp["status"]["type"]["name"] != "STATUS_FINAL" or (e.get("season") or {}).get("type") != 2:
                            continue
                        try:
                            s = client.get(f"{ESPN_BASE}/summary", params={"event": e["id"]}, timeout=30).json()
                        except Exception:  # noqa: BLE001
                            continue
                        bt = {t.get("homeAway"): t for t in s.get("boxscore", {}).get("teams", [])}
                        if "home" not in bt or "away" not in bt:
                            continue
                        hs, as_ = _extract_team_stats(bt["home"]), _extract_team_stats(bt["away"])
                        names = {c["homeAway"]: c["team"].get("abbreviation", "") for c in comp["competitors"]}
                        if not hs or not as_ or not names.get("home") or not names.get("away"):
                            continue
                        rows.append(dict(date=e["date"][:10], home=names["home"], away=names["away"], hs=hs, as_=as_))
                        time.sleep(0.05)
                f.write_text(json.dumps(rows))
                print(f"  wnba {yr}: {len(rows)} games cached")
            for r in rows:
                if r["home"] not in WNBA_FRANCHISES or r["away"] not in WNBA_FRANCHISES:
                    continue  # All-Star / exhibition sides (Team Wilson, USA, ...)
                h = dict(PTS=r["hs"]["pts"], FGA=r["hs"]["fga"], FTA=r["hs"]["fta"], OREB=r["hs"]["oreb"], TOV=r["hs"]["tov"])
                a = dict(PTS=r["as_"]["pts"], FGA=r["as_"]["fga"], FTA=r["as_"]["fta"], OREB=r["as_"]["oreb"], TOV=r["as_"]["tov"])
                games.append(_game(yr, r["date"], r["home"], r["away"], h, a))
    return _finish(games)


def _game(season, d, home, away, h, a):
    ph = _poss(h["FGA"], h["FTA"], h["OREB"], h["TOV"])
    pa = _poss(a["FGA"], a["FTA"], a["OREB"], a["TOV"])
    return dict(season=season, ord=date.fromisoformat(str(d)[:10]).toordinal(), home=home, away=away,
                hp=float(h["PTS"]), ap=float(a["PTS"]), poss=float((ph + pa) / 2.0))


def _finish(games):
    games = [g for g in games if g["poss"] > 40]
    games.sort(key=lambda g: (g["ord"], g["home"], g["away"]))
    teams = sorted({g["home"] for g in games} | {g["away"] for g in games})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(games)
    arr = dict(
        season=np.array([g["season"] for g in games]),
        ord=np.array([g["ord"] for g in games]),
        h=np.array([idx[g["home"]] for g in games]),
        a=np.array([idx[g["away"]] for g in games]),
        hp=np.array([g["hp"] for g in games]),
        ap=np.array([g["ap"] for g in games]),
        poss=np.array([g["poss"] for g in games]),
        home_win=np.array([1.0 if g["hp"] > g["ap"] else 0.0 for g in games]),
        n_teams=len(teams),
    )
    arr["n"] = n
    return arr


# ---------------------------------------------------------------- ratings
# Each rating fn takes (G, idx_before, cur_ord, cur_season, **kw) -> np.ndarray[n_teams] of
# net ratings (points per 100 possessions, higher = better) with NaN where unrated.

def rate_base(G, before, cur_ord, cur_season, min_gp=20, shrink_k=0.0, **_):
    """Shipped: season-to-date ratio-of-sums within the current season, min games, optional shrink."""
    s = before[G["season"][before] == cur_season]
    nt = G["n_teams"]
    pf = np.zeros(nt); pa = np.zeros(nt); po = np.zeros(nt); gp = np.zeros(nt)
    for side_pf, side_pa, tm in ((G["hp"], G["ap"], G["h"]), (G["ap"], G["hp"], G["a"])):
        np.add.at(pf, tm[s], side_pf[s]); np.add.at(pa, tm[s], side_pa[s])
        np.add.at(po, tm[s], G["poss"][s]); np.add.at(gp, tm[s], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        net = 100.0 * (pf - pa) / po
    if shrink_k:
        net = net * gp / (gp + shrink_k)
    net[gp < min_gp] = np.nan
    return net


def rate_ew(G, before, cur_ord, cur_season, hl=60.0, decay=1.0, **_):
    """Recency-weighted mean of per-game net rating (no opponent adjustment, no home term)."""
    age = cur_ord - G["ord"][before]
    w = 0.5 ** (age / hl) * decay ** (cur_season - G["season"][before]) * G["poss"][before]
    nt = G["n_teams"]
    num = np.zeros(nt); den = np.zeros(nt)
    m = 100.0 * (G["hp"][before] - G["ap"][before]) / G["poss"][before]
    np.add.at(num, G["h"][before], w * m); np.add.at(den, G["h"][before], w)
    np.add.at(num, G["a"][before], -w * m); np.add.at(den, G["a"][before], w)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[den < 1e-9] = np.nan
    return out


def rate_ridge(G, before, cur_ord, cur_season, lam=300.0, hl=120.0, decay=1.0, lookback_days=1100, **_):
    """Additive game-level ridge: rating_pts100 = mu + h*home + off[team] + dfa[opp], recency-weighted."""
    age = cur_ord - G["ord"][before]
    keep = age <= lookback_days
    b = before[keep]
    age = age[keep]
    if b.size < 50:
        return np.full(G["n_teams"], np.nan)
    w = 0.5 ** (age / hl) * decay ** (cur_season - G["season"][b]) * G["poss"][b]
    nt = G["n_teams"]
    p = 2 + 2 * nt
    g = b.size
    X = np.zeros((2 * g, p))
    rows = np.arange(g)
    X[:, 0] = 1.0
    X[rows, 1] = 1.0  # home offense rows
    X[rows, 2 + G["h"][b]] = 1.0
    X[rows, 2 + nt + G["a"][b]] = 1.0
    X[g + rows, 2 + G["a"][b]] = 1.0  # away offense rows
    X[g + rows, 2 + nt + G["h"][b]] = 1.0
    y = np.concatenate([100.0 * G["hp"][b] / G["poss"][b], 100.0 * G["ap"][b] / G["poss"][b]])
    ww = np.concatenate([w, w])
    pen = np.full(p, lam)
    pen[:2] = 1e-6
    Xw = X * ww[:, None]
    beta = np.linalg.solve(Xw.T @ X + np.diag(pen), Xw.T @ y)
    off, dfa = beta[2:2 + nt], beta[2 + nt:]
    seen = np.zeros(nt, dtype=bool)
    seen[G["h"][b]] = True; seen[G["a"][b]] = True
    net = off - dfa
    net[~seen] = np.nan
    return net


def rate_const(G, before, cur_ord, cur_season, **_):
    return np.zeros(G["n_teams"])


VARIANTS = {"base": rate_base, "ew": rate_ew, "ridge": rate_ridge, "const": rate_const}


def parse_variant(spec):
    name, _, arg = spec.partition(":")
    kw = {}
    if arg:
        for part in arg.split("/"):
            k, _, v = part.partition("=")
            kw[k] = float(v)
    return name, kw


# ---------------------------------------------------------------- scoring

def fit_logistic(d, y, iters=100):
    """Newton fit of P = sigmoid(a*d + b); d is standardised internally (points/100 poss are ~O(10))."""
    sd = float(np.std(d)) or 1.0
    a, b = _fit_std(d / sd, y, iters)
    return a / sd, b


def _fit_std(d, y, iters):
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a * d + b, -30, 30)))
        gvec = np.array([np.sum((p - y) * d), np.sum(p - y)])
        wgt = p * (1 - p) + 1e-9
        H = np.array([[np.sum(wgt * d * d), np.sum(wgt * d)], [np.sum(wgt * d), np.sum(wgt)]]) + np.eye(2) * 1e-6
        step = np.linalg.solve(H, gvec)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-8:
            break
    return a, b


def deltas(G, name, kw, sport_cfg, seasons):
    """Return delta[n] (NaN where a team is unrated) for games in `seasons`."""
    fn = VARIANTS[name]
    if name == "base":
        kw = {"min_gp": sport_cfg["min_gp"], "shrink_k": sport_cfg["shrink_k"], **kw}
    out = np.full(G["n"], np.nan)
    ords = G["ord"]
    # process one game-date at a time: ratings depend only on strictly earlier games
    for d in np.unique(ords[np.isin(G["season"], list(seasons))]):
        today = np.where(ords == d)[0]
        before = np.where(ords < d)[0]
        if before.size == 0:
            continue
        season = int(G["season"][today[0]])
        r = fn(G, before, int(d), season, **kw)
        dl = r[G["h"][today]] - r[G["a"][today]]
        out[today] = dl
    return out


def score_variant(G, dl, ids, eval_seasons):
    d, y, s = dl[ids], G["home_win"][ids], G["season"][ids]
    res, se = {}, {}
    for ev in eval_seasons:
        tr, te = s < ev, s == ev
        if tr.sum() < 200 or te.sum() == 0:
            continue
        a, b = fit_logistic(d[tr], y[tr])
        p = 1 / (1 + np.exp(-np.clip(a * d[te] + b, -30, 30)))
        res[ev] = (float(np.mean((p - y[te]) ** 2)), int(te.sum()))
        for gi, e in zip(ids[te], (p - y[te]) ** 2):
            se[int(gi)] = float(e)
    return res, se


def pooled(res, group):
    xs = [res[e] for e in group if e in res]
    n = sum(x[1] for x in xs)
    return (sum(x[0] * x[1] for x in xs) / n) if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=list(SPORTS), required=True)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--early", action="store_true", help="also score the early-season games the shipped rating cannot price")
    args = ap.parse_args()
    cfg = SPORTS[args.sport]
    default = ("base,ew:hl=60,ew:hl=120,ew:hl=240,ridge:lam=300/hl=60,ridge:lam=300/hl=120,ridge:lam=300/hl=240,ridge:lam=1000/hl=120"
               if args.sport == "nba" else
               "base,ew:hl=40,ew:hl=90,ew:hl=180,ridge:lam=100/hl=60,ridge:lam=100/hl=120,ridge:lam=100/hl=240,ridge:lam=300/hl=120")
    specs = [parse_variant(v) for v in (args.variants or default).split(",")]

    G = load_nba() if args.sport == "nba" else load_wnba()
    print(f"{args.sport}: {G['n']} games, {G['n_teams']} teams, seasons {sorted(set(G['season'].tolist()))}")
    groups = dict(rank=cfg["rank"], confirm=cfg["confirm"], holdout=cfg["holdout"] if args.holdout else ())
    eval_seasons = tuple(s for g in groups.values() for s in g)
    score_seasons = tuple(range(cfg["fit_start"], max(eval_seasons) + 1))

    dls = {}
    for name, kw in specs:
        t = time.time()
        dls[(name, tuple(sorted(kw.items())))] = deltas(G, name, kw, cfg, score_seasons)
        print(f"  computed {name}{kw} in {time.time() - t:.0f}s", flush=True)

    in_scope = np.isin(G["season"], list(score_seasons))
    rated = np.all([~np.isnan(dl) for dl in dls.values()], axis=0) & in_scope
    ids = np.where(rated)[0]
    print(f"common rated games: {ids.size}")
    base_se = None
    for key, dl in dls.items():
        res, se = score_variant(G, dl, ids, eval_seasons)
        line = f"{key[0]}{dict(key[1])}  rank={pooled(res, groups['rank']):.4f} confirm={pooled(res, groups['confirm']):.4f}"
        if args.holdout:
            line += f" holdout={pooled(res, groups['holdout']):.4f}"
        keys_rc = [gi for gi in se if G["season"][gi] in set(groups["rank"]) | set(groups["confirm"])]
        if base_se is None:
            base_se = se
        else:
            diff = np.array([se[gi] - base_se[gi] for gi in keys_rc])
            line += f"  d_vs_first={1000 * diff.mean():+.2f}/1000 z={diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))):+.2f}"
            if args.holdout:
                kh = [gi for gi in se if G["season"][gi] in set(groups["holdout"])]
                dh = np.array([se[gi] - base_se[gi] for gi in kh])
                line += f" | holdout d={1000 * dh.mean():+.2f}/1000 z={dh.mean() / (dh.std(ddof=1) / np.sqrt(len(dh))):+.2f}"
        print(line)

    if args.early:
        # games the shipped rating cannot price (early season): variants vs constant home rate
        print("\nEARLY-SEASON slice (shipped rating abstains): variant vs constant home-rate")
        const = deltas(G, "const", {}, cfg, score_seasons)
        base_nan = np.isnan(dls[next(iter(dls))]) & in_scope
        for key, dl in dls.items():
            if key[0] == "base":
                continue
            early = np.where(base_nan & ~np.isnan(dl))[0]
            if early.size == 0:
                continue
            both = np.where(in_scope & ~np.isnan(dl))[0]
            res, _ = score_variant(G, dl, both, eval_seasons)
            # restrict the reported Brier to the early games only
            d, y, s = dl[both], G["home_win"][both], G["season"][both]
            parts, parts_c, n_tot = 0.0, 0.0, 0
            for ev in eval_seasons:
                tr = s < ev
                if tr.sum() < 200:
                    continue
                a, b = fit_logistic(d[tr], y[tr])
                te = [gi for gi in early if G["season"][gi] == ev]
                if not te:
                    continue
                p = 1 / (1 + np.exp(-np.clip(a * dl[te] + b, -30, 30)))
                hr = y[tr].mean()
                parts += np.sum((p - G["home_win"][te]) ** 2)
                parts_c += np.sum((hr - G["home_win"][te]) ** 2)
                n_tot += len(te)
            if n_tot:
                print(f"{key[0]}{dict(key[1])}  early n={n_tot}  brier={parts / n_tot:.4f}  const_home_rate={parts_c / n_tot:.4f}")


if __name__ == "__main__":
    main()
