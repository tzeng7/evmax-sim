"""Fit + validate the dispersion of NFL prop pricing (yardage Gamma, receptions NegBin).

evmax/ev/prop_pricing.py prices every Kalshi ``X+`` threshold off ONE Pinnacle
anchor (line + devigged P(over)) by fixing a per-stat dispersion and solving
for the location. This script measures which dispersion is right, against
SETTLED outcomes, on the thresholds Kalshi actually lists.

Data (read-only):
  * Anchor — one per player-game: the last archived snapshot before kickoff
    (``archived_sharp_odds``; kickoff = the prop rows' ``event_date``). A real
    Pinnacle quote (``derived = 0``) or a derived rung's ``anchor_line`` /
    ``anchor_prob_over`` is used directly. Rows archived before 2026-10-09 only
    hold re-lined rungs; their anchor is recovered EXACTLY: the stored decimals
    devig (power) to the anchor prob, and inverting the pricing family in force
    then (:data:`LEGACY_SIGMA` / :data:`LEGACY_NEGBIN_K`) turns any rung back
    into the anchor line (it lands on a half point to ~1e-14).
  * Outcome — ``prop_observations.actual_value`` (nflverse-resolved).
  * Rungs — the Kalshi thresholds listed for that player-game
    (``prop_observations``), with the last pre-kickoff Kalshi bid/ask mid from
    ``archived_kalshi_markets`` as a calibration benchmark (rows with
    ``yes + no == 1`` exactly are synthesized quotes and skipped).

Scoring: rung Brier (and log loss) of P(Y >= K) vs 1{y >= K}. Families are fit
by minimizing train rung Brier. Validation:
  * walk-forward — fit on weeks < w, score week w, pooled;
  * frozen holdout — fit on weeks <= ``--split-week``, score the rest once.
Standard errors are clustered by player-game (all rungs of one player-game
share one outcome). NFL weeks are Tue-Mon blocks from the first game date.

Run:
    python scripts/fit_nfl_prop_dispersion.py \
        --archive-db ~/Projects/evmax/data/archive.db \
        --pred-db ~/Projects/evmax/data/predictions.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evmax.ev.devig import devig_two_way  # noqa: E402
from evmax.ev.prop_pricing import (  # noqa: E402
    _GAMMA_STAT_SCALE,
    _NEGBIN_STAT_K,
    price_kalshi_threshold,
)

# Pricing in force 2026-09-06 → 2026-10-09 (commit 8574b29 to this change):
# the family the legacy archived rungs were priced with.
LEGACY_SIGMA = {"receiving_yards": 24.0, "rushing_yards": 30.0, "passing_yards": 70.0}
LEGACY_NEGBIN_K = {"receptions": 5.0}
YARD_STATS = ("receiving_yards", "rushing_yards")
DEFAULT_STATS = ("receiving_yards", "rushing_yards", "receptions")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _legacy_anchor(stat: str, rungs: pd.DataFrame, p_anchor: float) -> float | None:
    """Anchor line from legacy re-lined rungs (inverting the legacy family)."""
    ok = rungs[(rungs.pk > 0.02) & (rungs.pk < 0.98)]
    if ok.empty:
        return None
    if stat in LEGACY_SIGMA:
        s = LEGACY_SIGMA[stat]
        mu = float(np.median(ok.K - 0.5 + s * stats.norm.ppf(ok.pk)))
        return float(np.round((mu - s * stats.norm.ppf(p_anchor)) * 2) / 2)
    k = LEGACY_NEGBIN_K.get(stat)
    if k is None:
        return None
    mus = []
    for K, pk in zip(ok.K, ok.pk):
        c = int(np.ceil(K - 1e-9))
        try:
            mus.append(optimize.brentq(
                lambda m: stats.nbinom.sf(c - 1, k, k / (k + m)) - pk, 1e-4, 200.0))
        except ValueError:
            continue
    if not mus:
        return None
    mu = float(np.median(mus))
    cands = np.arange(0.5, 40.5, 1.0)
    sf = stats.nbinom.sf(np.ceil(cands) - 1, k, k / (k + mu))
    return float(cands[np.argmin(np.abs(sf - p_anchor))])


def load_anchors(archive_db: Path, stats_: tuple[str, ...]) -> pd.DataFrame:
    con = _ro(archive_db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(archived_sharp_odds)")}
    extra = (", derived, anchor_line, anchor_prob_over" if "derived" in cols
             else ", NULL AS derived, NULL AS anchor_line, NULL AS anchor_prob_over")
    marks = ",".join("?" * len(stats_))
    df = pd.read_sql_query(
        f"""SELECT fetched_at, prop_player_name AS player, prop_stat_type AS stat,
                   substr(event_id, 6, 10) AS gd, total_line AS K, true_prob_over AS pk,
                   outcome_a_decimal AS oa, outcome_b_decimal AS ob, event_date AS kick{extra}
            FROM archived_sharp_odds
            WHERE sector = 'nfl' AND prop_player_name IS NOT NULL
              AND prop_stat_type IN ({marks})""",
        con, params=list(stats_),
    )
    con.close()
    df["fetched"] = pd.to_datetime(df.fetched_at, utc=True, format="ISO8601")
    df["kickoff"] = pd.to_datetime(df.kick, utc=True, format="ISO8601")
    df = df[df.fetched < df.kickoff]
    df = df[df.fetched == df.groupby(["player", "stat", "gd"]).fetched.transform("max")]

    out = []
    for (player, stat, gd), g in df.groupby(["player", "stat", "gd"]):
        quote = g[g.derived == 0]
        carried = g[g.anchor_line.notna()]
        if not quote.empty:
            line, p, src = float(quote.K.iloc[0]), float(quote.pk.iloc[0]), "quote"
        elif not carried.empty:
            line = float(carried.anchor_line.iloc[0])
            p = float(carried.anchor_prob_over.iloc[0])
            src = "carried"
        else:
            p = devig_two_way(float(g.oa.iloc[0]), float(g.ob.iloc[0]), method="power")[0]
            line = _legacy_anchor(stat, g, p)
            src = "legacy_inverted"
            if line is None:
                continue
        out.append(dict(player=player, stat=stat, gd=gd, line=line, p_anchor=p,
                        kickoff=g.kickoff.iloc[0], anchor_source=src))
    return pd.DataFrame(out)


def nfl_week(dates: pd.Series) -> pd.Series:
    """NFL week = Tue-Mon block counted from the Tuesday before the first game."""
    d = pd.to_datetime(dates)
    first = d.min()
    start = first - pd.Timedelta(days=(first.weekday() - 1) % 7)
    return ((d - start).dt.days // 7 + 1).astype(int)


def load_rungs(archive_db: Path, pred_db: Path, stats_: tuple[str, ...]) -> pd.DataFrame:
    anchors = load_anchors(archive_db, stats_)
    pcon = _ro(pred_db)
    marks = ",".join("?" * len(stats_))
    obs = pd.read_sql_query(
        f"""SELECT player_name AS player, stat_type AS stat, event_date AS gd,
                   MAX(actual_value) AS y
            FROM prop_observations
            WHERE sector = 'nfl' AND actual_value IS NOT NULL AND stat_type IN ({marks})
            GROUP BY player_name, stat_type, event_date""",
        pcon, params=list(stats_),
    )
    rungs = pd.read_sql_query(
        f"""SELECT DISTINCT player_name AS player, stat_type AS stat, event_date AS gd,
                   line AS K, substr(market_id, 8) AS ticker
            FROM prop_observations
            WHERE sector = 'nfl' AND venue = 'kalshi' AND market_id LIKE 'kalshi:%'
              AND stat_type IN ({marks})""",
        pcon, params=list(stats_),
    )
    pcon.close()
    pg = anchors.merge(obs, on=["player", "stat", "gd"])
    pg["week"] = nfl_week(pg.gd)
    rungs = rungs.merge(pg, on=["player", "stat", "gd"])

    acon = _ro(archive_db)
    km = pd.read_sql_query(
        "SELECT ticker, fetched_at, yes_price, no_price FROM archived_kalshi_markets "
        "WHERE sector = 'nfl' AND market_type = 'player_prop'", acon)
    acon.close()
    km = km[km.ticker.isin(set(rungs.ticker))]
    km = km[(km.yes_price + km.no_price - 1.0).abs() > 1e-9]
    km["fetched"] = pd.to_datetime(km.fetched_at, utc=True, format="ISO8601")
    km = km.merge(rungs[["ticker", "kickoff"]].drop_duplicates("ticker"), on="ticker")
    km = km[km.fetched < km.kickoff].sort_values("fetched").groupby("ticker").tail(1)
    km["mid"] = (km.yes_price + 1.0 - km.no_price) / 2.0
    rungs = rungs.merge(km[["ticker", "mid"]], on="ticker", how="left")
    rungs["hit"] = (rungs.y >= rungs.K).astype(float)
    rungs["pg"] = rungs.player + "|" + rungs.stat + "|" + rungs.gd
    return rungs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Families — vectorized P(Y >= K) from anchor arrays (L = line, p = P(Y > L))
# ---------------------------------------------------------------------------


def _bisect_mu(sf_of_mu, target, lo, hi, n=80):
    """Vectorized log-space bisection; sf_of_mu must increase with mu."""
    lo = np.full_like(target, np.log(lo), dtype=float)
    hi = np.full_like(target, np.log(hi), dtype=float)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        above = sf_of_mu(np.exp(mid)) > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.exp(0.5 * (lo + hi))


def normal_fixed(params, L, p, K):
    (s,) = params
    return stats.norm.sf(K - 0.5, L + s * stats.norm.ppf(p), s)


def gamma_power(params, L, p, K, power=None):
    """Gamma with Var = phi * mu**power (power=1: fixed scale, the shipped form)."""
    if power is None:
        phi, power = params
    else:
        (phi,) = params

    def sf(x, mu):
        shape, scale = mu ** (2 - power) / phi, phi * mu ** (power - 1)
        return special.gammaincc(shape, np.maximum(x, 0.0) / scale)

    mu = _bisect_mu(lambda m: sf(L, m), p, 1e-3, 5000.0)
    return np.where(K - 0.5 <= 0, 1.0, sf(K - 0.5, mu))


def zi_gamma(params, L, p, K):
    """Fixed-scale Gamma plus a 'dud' point mass pi at zero."""
    phi, pi = params[0], min(params[1], 0.4)
    target = np.minimum(p / (1 - pi), 0.999)
    mu = _bisect_mu(lambda m: special.gammaincc(m / phi, L / phi), target, 1e-3, 5000.0)
    return np.where(K >= 1, (1 - pi) * special.gammaincc(mu / phi, np.maximum(K - 0.5, 0) / phi), 1.0)


def lognormal(params, L, p, K):
    (s,) = params
    m = np.log(L) - s * stats.norm.ppf(1 - p)
    return stats.norm.sf((np.log(np.maximum(K - 0.5, 1e-9)) - m) / s)


def _nb_sf(c, mu, k):
    return np.where(c <= 0, 1.0, 1.0 - special.betainc(k, np.maximum(c, 1), k / (k + mu)))


def negbin_k(params, L, p, K):
    (k,) = params
    mu = _bisect_mu(lambda m: _nb_sf(np.ceil(L + 1e-9), m, k), p, 1e-3, 200.0)
    return _nb_sf(np.ceil(K - 1e-9), mu, k)


def negbin_nb1(params, L, p, K):
    (delta,) = params  # Var = mu * (1 + delta)
    mu = _bisect_mu(lambda m: _nb_sf(np.ceil(L + 1e-9), m, m / delta), p, 1e-3, 200.0)
    return _nb_sf(np.ceil(K - 1e-9), mu, mu / delta)


def yard_families(stat: str) -> dict:
    """name -> (fn, x0 or fixed params, fixed?)."""
    return {
        "legacy_normal": (normal_fixed, [LEGACY_SIGMA[stat]], True),
        "normal_refit": (normal_fixed, [30.0], False),
        "lognormal": (lognormal, [0.7], False),
        "gamma_cv": (lambda pr, L, p, K: gamma_power(pr, L, p, K, power=2.0), [0.5], False),
        "gamma_scale": (lambda pr, L, p, K: gamma_power(pr, L, p, K, power=1.0), [20.0], False),
        "gamma_power": (gamma_power, [20.0, 1.2], False),
        "zi_gamma": (zi_gamma, [20.0, 0.05], False),
    }


def reception_families() -> dict:
    return {
        "legacy_negbin": (negbin_k, [LEGACY_NEGBIN_K["receptions"]], True),
        "negbin_refit": (negbin_k, [10.0], False),
        "negbin_nb1": (negbin_nb1, [0.3], False),
    }


SHIPPED = {"receiving_yards": "gamma_scale", "rushing_yards": "gamma_scale",
           "receptions": "legacy_negbin"}
LEGACY = {"receiving_yards": "legacy_normal", "rushing_yards": "legacy_normal",
          "receptions": "legacy_negbin"}


def predict(fn, params, d):
    pr = fn(params, d.line.values, d.p_anchor.values, d.K.values.astype(float))
    return np.clip(pr, 1e-6, 1 - 1e-6)


def brier(pred, d):
    return float(np.mean((pred - d.hit.values) ** 2))


def logloss(pred, d):
    y = d.hit.values
    return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))


def fit(fn, x0, d):
    res = optimize.minimize(lambda lx: brier(predict(fn, np.exp(lx), d), d), np.log(x0),
                            method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
    return np.exp(res.x)


def clustered_delta(a, b, d):
    """Mean per-rung Brier difference a − b (/1000) with player-game clustered SE."""
    diff = (a - d.hit.values) ** 2 - (b - d.hit.values) ** 2
    m = diff.mean()
    s = pd.Series(diff - m).groupby(d.pg.values).sum()
    g = len(s)
    return m * 1000, np.sqrt(g / (g - 1) * (s ** 2).sum()) / len(diff) * 1000


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def check_module_parity(rungs: pd.DataFrame) -> None:
    """The vectorized shipped families must equal evmax.ev.prop_pricing."""
    for stat, fn, params in [
        ("receiving_yards", yard_families("receiving_yards")["gamma_scale"][0],
         [_GAMMA_STAT_SCALE["receiving_yards"]]),
        ("rushing_yards", yard_families("rushing_yards")["gamma_scale"][0],
         [_GAMMA_STAT_SCALE["rushing_yards"]]),
        ("receptions", negbin_k, [_NEGBIN_STAT_K["receptions"]]),
    ]:
        d = rungs[rungs.stat == stat].head(300)
        if d.empty:
            continue
        vec = fn(params, d.line.values, d.p_anchor.values, d.K.values.astype(float))
        mod = np.array([price_kalshi_threshold(stat, L, p, K)
                        for L, p, K in zip(d.line, d.p_anchor, d.K)])
        gap = float(np.max(np.abs(vec - mod)))
        if gap > 1e-6:
            raise SystemExit(f"{stat}: script family drifted from prop_pricing (max |Δ| {gap:.2e})")


def report_stat(rungs: pd.DataFrame, stat: str, split_week: int) -> None:
    d = rungs[rungs.stat == stat]
    if d.empty:
        print(f"\n## {stat}: no data")
        return
    fams = reception_families() if stat == "receptions" else yard_families(stat)
    train, hold = d[d.week <= split_week], d[d.week > split_week]
    print(f"\n## {stat}: {d.pg.nunique()} player-games / {len(d)} rungs "
          f"(holdout weeks > {split_week}: {hold.pg.nunique()} pg)")

    preds_hold: dict[str, np.ndarray] = {}
    rows = []
    for name, (fn, x0, fixed) in fams.items():
        wf = pd.Series(np.nan, index=d.index)
        for w in sorted(d.week.unique()):
            tr, te = d[d.week < w], d[d.week == w]
            if tr.empty:
                continue
            wf.loc[te.index] = predict(fn, x0 if fixed else fit(fn, x0, tr), te)
        pr_frozen = x0 if fixed else fit(fn, x0, train)
        pr_all = x0 if fixed else fit(fn, x0, d)
        preds_hold[name] = predict(fn, pr_frozen, hold)
        scored = d[wf.notna()]
        rows.append({
            "family": name,
            "wf_brier": brier(wf.dropna().values, scored) * 1000,
            "wf_logloss": logloss(wf.dropna().values, scored),
            "hold_brier": brier(preds_hold[name], hold) * 1000,
            "hold_logloss": logloss(preds_hold[name], hold),
            "params_train": np.round(pr_frozen, 3).tolist(),
            "params_all": np.round(pr_all, 3).tolist(),
        })
    print(pd.DataFrame(rows).set_index("family").round(4).to_string())

    shipped, legacy = SHIPPED[stat], LEGACY[stat]
    if shipped != legacy:
        m, se = clustered_delta(preds_hold[shipped], preds_hold[legacy], hold)
        print(f"holdout ΔBrier {shipped} − {legacy}: {m:+.2f} ± {se:.2f} /1000")
    hk = hold.assign(_s=preds_hold[shipped])[hold.mid.notna().values]
    if not hk.empty:
        m, se = clustered_delta(hk._s.values, hk.mid.values, hk)
        print(f"holdout ΔBrier {shipped} − kalshi_mid ({hk.pg.nunique()} pg with a mid): "
              f"{m:+.2f} ± {se:.2f} /1000")

    edges = ([-99, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 99] if stat == "receptions"
             else [-999, -30, -10, 10, 30, 50, 70, 999])
    h = hold.assign(legacy=preds_hold[legacy], shipped=preds_hold[shipped],
                    bucket=pd.cut(hold.K - hold.line, edges))
    cal = h.groupby("bucket", observed=True).apply(lambda x: pd.Series({
        "rungs": len(x), "pg": x.pg.nunique(), "realized": x.hit.mean(),
        "se": x.groupby("pg").hit.mean().std() / np.sqrt(max(x.pg.nunique(), 1)),
        "legacy": x.legacy.mean(), "shipped": x.shipped.mean(), "kalshi_mid": x.mid.mean(),
    }))
    print("holdout calibration by threshold distance K − line:")
    print(cal.round(3).to_string())

    # Grid: P(Y >= line + 0.5 + off) per player-game (shipped constants, all weeks).
    pgs = d.drop_duplicates("pg")
    offs = [-2, -1, 1, 2, 3, 4] if stat == "receptions" else [-40, -20, 20, 40, 60, 80]
    fn_l, x_l, _ = fams[legacy]
    fn_s, _x, _f = fams[shipped]
    x_s = ([_NEGBIN_STAT_K["receptions"]] if stat == "receptions"
           else [_GAMMA_STAT_SCALE[stat]])
    grid = []
    for off in offs:
        K = (pgs.line + 0.5 + off).values
        real = (pgs.y.values >= K).mean()
        grid.append({
            "K − line": off + 0.5,
            "legacy": fn_l(x_l, pgs.line.values, pgs.p_anchor.values, K).mean(),
            "shipped": fn_s(x_s, pgs.line.values, pgs.p_anchor.values, K).mean(),
            "realized": real, "se": np.sqrt(real * (1 - real) / len(pgs)),
        })
    print(f"all weeks, one row per player-game (n={len(pgs)}), shipped constants:")
    print(pd.DataFrame(grid).set_index("K − line").round(3).to_string())

    if stat in YARD_STATS:
        fn = fams["gamma_scale"][0]
        prof = {t: round(brier(predict(fn, [t], hold), hold) * 1000, 2)
                for t in (10, 15, 20, 25, 30, 35)}
        print(f"holdout Brier by θ (gamma_scale): {prof}   shipped θ = {_GAMMA_STAT_SCALE[stat]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive-db", type=Path, default=REPO_ROOT / "data" / "archive.db")
    ap.add_argument("--pred-db", type=Path, default=REPO_ROOT / "data" / "predictions.db")
    ap.add_argument("--split-week", type=int, default=2,
                    help="frozen split: train weeks <= N, holdout the rest")
    ap.add_argument("--stats", default=",".join(DEFAULT_STATS))
    args = ap.parse_args(argv)
    for path in (args.archive_db, args.pred_db):
        if not path.exists() or path.stat().st_size == 0:
            print(f"missing or empty database: {path} (pass --archive-db / --pred-db)")
            return 1

    stats_ = tuple(s.strip() for s in args.stats.split(",") if s.strip())
    rungs = load_rungs(args.archive_db, args.pred_db, stats_)
    if rungs.empty:
        print("no resolved NFL prop rungs found")
        return 1
    check_module_parity(rungs)
    pgs = rungs.drop_duplicates("pg")
    print("# NFL prop dispersion fit")
    print(f"weeks {pgs.week.min()}-{pgs.week.max()}, games {pgs.gd.min()} → {pgs.gd.max()}")
    print(pgs.groupby(["stat", "week"]).size().unstack(fill_value=0).to_string())
    print("anchor source:", pgs.anchor_source.value_counts().to_dict(),
          "| rungs with a Kalshi mid:", f"{rungs.mid.notna().mean():.1%}")
    for stat in stats_:
        report_stat(rungs, stat, args.split_week)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
