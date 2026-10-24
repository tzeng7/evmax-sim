"""Canonical, deterministic performance evaluation (`evmax cleanup eval`).

ONE definition of "how are we doing" — ROI, CLV, Brier — so a number quoted
from any lens always names the same rows. Before this module, ``shadow
metrics`` (scan-date window, YES-only ROI at ``captured_yes_price``, gross of
fees), ``shadow clv`` (CLV-non-null rows, optional staleness) and the
promotion board (event-date window, live-eligible gate) each scored a
DIFFERENT population, so an answer depended on which lens was run and nothing
printed how many rows each filter dropped.

Design rules (every one is enforced by tests/test_evaluation.py):

* **Funnel, not filters.** Rows pass through an ordered list of named stages;
  the result carries the row count after every stage. A claim like "only 10
  rows pass the gate" is only valid next to the funnel that shows the stage
  that cut 135 games to 10.
* **Reuse, never reimplement.** Contamination, live-eligibility, the CLV gate
  constants, ``game_key``, entry-price anchoring, fees and the Brier helpers
  are imported from the modules the board and gates already use.
* **Net of fees AND gross, side by side.** ROI and CLV are both reported both
  ways; the verdict uses net.
* **Game-clustered uncertainty.** An alt-spread ladder logs many correlated
  rungs per game, so the bootstrap resamples whole GAMES with a fixed seed.
* **Reproducible.** The result carries the spec, a sha256 of the exact
  market_ids scored and a DB fingerprint. Same DB + same spec => byte-identical
  JSON (no wall-clock inside the result; ``--since`` is explicit).
* **Own-side orientation.** ``ev_outcomes.outcome`` is 1 when the ROW's own
  side won (verified: for ``:no`` rows mean(outcome) tracks mean(blended)), and
  the entry price is the row's own ask, so one formula covers YES and NO rows.

Scope: game markets in ``ev_predictions``. Player props live in
``prop_observations`` with a different resolution path and are excluded
(stage ``scope`` drops ``::prop::`` ids and says so).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Optional

POPULATIONS = ("gate", "clean", "raw")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261018
GROUP_FIELDS = ("sector", "market_type", "venue")


@dataclass(frozen=True)
class EvalSpec:
    """Everything that selects rows. Serialized verbatim into the result."""

    sector: Optional[str] = None
    market_type: Optional[str] = None
    venue: Optional[str] = None
    mode: Optional[str] = None          # live | shadow | None = both
    league: Optional[str] = None
    side: Optional[str] = None          # lay | take (line sign)
    sources_token: Optional[str] = None
    since: Optional[str] = None         # event_date >= (YYYY-MM-DD), inclusive
    until: Optional[str] = None         # event_date <= (YYYY-MM-DD), inclusive
    # gate  = clean AND live-eligible (what the board / promotion gate scores)
    # clean = not contaminated (superseded-code rows dropped)
    # raw   = everything resolved and non-void (no code-state filtering)
    population: str = "gate"
    # Kalshi close-snapshot staleness cap for the CLV population (the board's
    # default is 3h). None disables the archive read.
    max_staleness_h: Optional[float] = 3.0
    group_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.population not in POPULATIONS:
            raise ValueError(f"population must be one of {POPULATIONS}")
        if self.side not in (None, "lay", "take"):
            raise ValueError("side must be 'lay' or 'take'")
        if self.mode not in (None, "live", "shadow"):
            raise ValueError("mode must be 'live' or 'shadow'")
        if self.venue not in (None, "kalshi", "polymarket_us"):
            raise ValueError("venue must be 'kalshi' or 'polymarket_us'")
        bad = [g for g in self.group_by if g not in GROUP_FIELDS]
        if bad:
            raise ValueError(f"group_by fields must be in {GROUP_FIELDS}, got {bad}")
        if self.max_staleness_h is not None and self.max_staleness_h < 0:
            raise ValueError("max_staleness_h must be >= 0")


# ---------------------------------------------------------------------------
# Row selection (single SQL, then ordered Python stages with counts)
# ---------------------------------------------------------------------------

_SELECT = """
    SELECT p.market_id, p.event_id, p.sector, p.market_type, p.venue, p.mode,
           p.league, p.line, p.model_sources, p.event_date, p.voided,
           p.void_reason, p.minutes_to_tipoff, p.placed, p.placed_at,
           p.placed_price, p.logged_at, p.kalshi_yes_price,
           p.blended_true_prob, p.sharp_true_prob, p.kalshi_clv_pct,
           p.placed_stake, p.maker_fill, o.outcome
    FROM ev_predictions p
    LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
    {where}
    ORDER BY p.market_id
"""


def _scope_where(spec: EvalSpec) -> tuple[str, list]:
    where = ["p.event_id NOT LIKE '%::prop::%'"]
    params: list = []
    for col, val in (
        ("p.sector", spec.sector and spec.sector.lower()),
        ("p.market_type", spec.market_type),
        ("p.venue", spec.venue),
        ("p.mode", spec.mode),
        ("p.league", spec.league and spec.league.lower()),
    ):
        if val is not None:
            where.append(f"{col} = ?")
            params.append(val)
    if spec.since:
        where.append("p.event_date >= ?")
        params.append(spec.since)
    if spec.until:
        where.append("p.event_date <= ?")
        params.append(spec.until)
    if spec.side:
        where.append("p.line < 0" if spec.side == "lay" else "p.line > 0")
    if spec.sources_token:
        where.append("p.model_sources LIKE ?")
        params.append(f"%{spec.sources_token}%")
    return "WHERE " + " AND ".join(where), params


def fetch_scope_rows(spec: EvalSpec, conn=None) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    where, params = _scope_where(spec)
    sql = _SELECT.format(where=where)
    if conn is not None:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    with get_connection() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def entry_price(r: dict) -> Optional[float]:
    """The CLV anchor: placed fill, else scan ask (resolver.clv_entry_price)."""
    from evmax.agents.cleanup.resolver import clv_entry_price

    p = clv_entry_price(r["placed"], r["placed_price"], r["kalshi_yes_price"])
    return p if p is not None and 0.0 < p < 1.0 else None


def _roi_stages(rows: list[dict], spec: EvalSpec) -> tuple[list[dict], list[tuple[str, int]]]:
    """Ordered stages up to the ROI/Brier population. Returns (rows, funnel)."""
    from evmax.agents.cleanup.contamination import is_contaminated
    from evmax.agents.odds.ev_gap_agent import has_full_blend

    funnel: list[tuple[str, int]] = [("scope", len(rows))]

    rows = [r for r in rows if r["outcome"] is not None]
    funnel.append(("resolved", len(rows)))

    # Cancelled/manual voids are not bets. stale_reverted voids STAY: they were
    # real scan decisions; dropping them biases results upward (same rule as
    # shadow._fetch_clv_rows).
    rows = [r for r in rows if not r["voided"] or r["void_reason"] == "stale_reverted"]
    funnel.append(("not_voided", len(rows)))

    # At/after-tip entries are in-play prices, not pre-game bets.
    rows = [r for r in rows if r["minutes_to_tipoff"] is None or r["minutes_to_tipoff"] > 0]
    funnel.append(("pre_game_entry", len(rows)))

    rows = [r for r in rows if entry_price(r) is not None]
    funnel.append(("valid_entry_price", len(rows)))

    if spec.population in ("clean", "gate"):
        rows = [
            r for r in rows
            if not is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"])
        ]
        funnel.append(("current_code", len(rows)))
    if spec.population == "gate":
        rows = [r for r in rows if has_full_blend(r["sector"], r["model_sources"], r["market_type"])]
        funnel.append(("live_eligible", len(rows)))
    return rows, funnel


def _fresh_close(rows: list[dict], max_staleness_h: float) -> tuple[list[dict], int]:
    """Close-snapshot staleness filter, applied to EVERY venue the same way.

    The close lives in archived_kalshi_markets for both venues (PolyUS under its
    venue-prefixed id, `resolver.close_lookup_ticker`); `shadow._fetch_clv_rows`
    exempts PolyUS rows from the cap, the eval deliberately does not, so the
    rule is identical for every sector AND venue.
    """
    from evmax.agents.cleanup.resolver import (
        close_lookup_ticker,
        clv_near_tip_ok,
        clv_not_before,
    )
    from evmax.archiver import DataArchiver

    archiver = DataArchiver()
    fresh, dropped = [], 0
    for r in rows:
        ticker, is_no = close_lookup_ticker(r["market_id"])
        if not ticker:
            dropped += 1
            continue
        not_before = clv_not_before(r["placed"], r["placed_at"], r["logged_at"])
        stale = archiver.get_kalshi_close_staleness_h(
            ticker, r["event_id"], not_before=not_before,
            side="no" if is_no else "yes",
            near_tip_ok=clv_near_tip_ok(r["placed"], ticker),
        )
        if stale is None or stale > max_staleness_h:
            dropped += 1
            continue
        fresh.append(r)
    return fresh, dropped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _fee_prob(venue: str, price: float) -> float:
    from evmax.fees import venue_fee_prob

    try:
        return venue_fee_prob(venue or "kalshi", price)
    except ValueError:
        return 0.0


def _pnl(r: dict, net: bool) -> float:
    """Flat $1-notional P&L on the row's own side."""
    from evmax.fees import bet_pnl

    price = entry_price(r)
    won = r["outcome"] == 1
    return bet_pnl(1.0, price, won, venue=(r["venue"] or "kalshi") if net else None)


def _game(r: dict) -> str:
    from evmax.cli.commands.shadow import game_key

    return game_key(r["event_id"]) if r["event_id"] else f"row:{r['market_id']}"


def _bootstrap_ci(per_game: list[tuple[float, float]]) -> Optional[tuple[float, float]]:
    """95% CI of sum(num)/sum(den), resampling GAMES. Fixed seed => deterministic."""
    if len(per_game) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(per_game)
    stats = []
    for _ in range(BOOTSTRAP_REPS):
        num = den = 0.0
        for _i in range(n):
            a, b = per_game[rng.randrange(n)]
            num += a
            den += b
        if den:
            stats.append(num / den)
    if not stats:
        return None
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[min(len(stats) - 1, int(0.975 * len(stats)))]


# ---------------------------------------------------------------------------
# Fixed output schema. EVERY group (any sector, empty or not) emits exactly
# these keys; undefined values are None. `conform` raises on an unknown key so
# a new field must be added here (and therefore to every sector) deliberately.
# ---------------------------------------------------------------------------
_CI = None
_STATS = {"mean_pp": None, "ci95_pp": None, "frac_positive": None}
ROI_TEMPLATE = {
    "row_digest": None, "n_rows": 0, "n_games": 0, "win_rate": None,
    "mean_entry_price": None, "mean_model_prob": None,
    "roi_gross_pct": None, "roi_gross_ci95_pct": None,
    "roi_net_pct": None, "roi_net_ci95_pct": None,
    "brier_blend": None, "brier_sharp": None,
    "brier_delta_per_1000": None, "brier_delta_z": None,
}
CLV_BASE = {
    "row_digest": None, "n_rows": 0, "n_games": 0,
    "gross": _STATS, "net_of_fees": _STATS,
    "clears_gross": False, "clears_net": False,
    "gate": {"min_games": None, "min_mean_pp": None, "min_frac_positive": None},
}
CLV_TEMPLATE = {**CLV_BASE, "uncapped": None, "empty_reason": None}
REALIZED_TEMPLATE = {
    "funnel": [], "n_bets": 0, "n_games": 0, "staked_usd": None,
    "pnl_est_usd": None, "roi_est_on_stake_pct": None, "roi_est_ci95_pct": None,
    "pnl_raw_usd": None, "pnl_fee_inclusive_usd": None,
    "price_kinds": {"round_cent_raw": 0, "decimal_odds_fee_inclusive": 0,
                    "unknown_treated_raw": 0},
    "maker_fills": None, "mean_fill_minus_scan_ask_pp": None,
    "row_digest": None, "empty_reason": None,
}


def conform(block: dict, template: dict, path: str = "") -> dict:
    """Return `block` on the template's exact key set (missing -> template default)."""
    extra = set(block) - set(template)
    if extra:
        raise KeyError(f"unknown eval field(s) {sorted(extra)} at '{path or 'root'}'")
    out = {}
    for k, default in template.items():
        if k not in block:
            out[k] = copy.deepcopy(default)
        elif isinstance(default, dict) and default and isinstance(block[k], dict):
            out[k] = conform(block[k], default, f"{path}.{k}")
        else:
            out[k] = block[k]
    return out


def sector_rules(sector: str) -> dict:
    """The sector-specific policy that shaped a group's populations.

    The PIPELINE is identical for every sector; only these data-driven rule
    tables differ, so they are reported (same keys everywhere) instead of
    being silent branches inside the stages.
    """
    from evmax.agents.cleanup.contamination import CONTAMINATION_RULES
    from evmax.agents.odds.ev_gap_agent import MIN_NONSHARP_MODELS, REQUIRED_BLEND_MODELS

    sec = (sector or "").lower()
    return {
        "contamination_rules": len(CONTAMINATION_RULES.get(sec, [])),
        "required_blend_models": sorted(REQUIRED_BLEND_MODELS.get(sec, [])),
        "min_nonsharp_models": (
            {"min_count": m["min_count"], "market_types": sorted(m["market_types"])}
            if (m := MIN_NONSHARP_MODELS.get(sec)) else None
        ),
    }


def _round(x: Optional[float], d: int = 4) -> Optional[float]:
    return None if x is None else round(x, d)


def _roi_block(rows: list[dict]) -> dict:
    if not rows:
        return {"n_rows": 0, "n_games": 0}
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(_game(r), []).append(r)
    out = {
        "n_rows": len(rows),
        "n_games": len(by_game),
        "win_rate": _round(sum(1 for r in rows if r["outcome"] == 1) / len(rows)),
        "mean_entry_price": _round(sum(entry_price(r) for r in rows) / len(rows)),
        "mean_model_prob": _round(sum(r["blended_true_prob"] for r in rows) / len(rows)),
    }
    for label, net in (("gross", False), ("net", True)):
        total = sum(_pnl(r, net) for r in rows)
        pg = [(sum(_pnl(r, net) for r in g), float(len(g))) for g in by_game.values()]
        ci = _bootstrap_ci(pg)
        out[f"roi_{label}_pct"] = _round(100 * total / len(rows), 2)
        out[f"roi_{label}_ci95_pct"] = (
            [round(100 * ci[0], 2), round(100 * ci[1], 2)] if ci else None
        )
    return out


def _brier_block(rows: list[dict]) -> dict:
    from evmax.agents.cleanup.value_audit import _brier, _paired_diff_stats

    if not rows:
        return {}
    paired = _paired_diff_stats(rows, "sharp_true_prob", "blended_true_prob")
    return {
        "brier_blend": _round(_brier(rows, "blended_true_prob")),
        "brier_sharp": _round(_brier(rows, "sharp_true_prob")),
        # positive => blend beats sharp
        "brier_delta_per_1000": _round(paired["mean"] * 1000, 2) if paired else None,
        "brier_delta_z": _round(paired["z"], 2) if paired else None,
    }


def _clv_block(rows: list[dict]) -> dict:
    from evmax.cli.commands.shadow import (
        CLV_MIN_FRAC_POSITIVE,
        CLV_MIN_MEAN_PP,
        MIN_CLV_RESOLVED,
        clv_clears,
    )

    if not rows:
        return {"n_rows": 0, "n_games": 0, "clears_gross": False, "clears_net": False}
    gross = [r["kalshi_clv_pct"] for r in rows]
    net = [
        c - 100.0 * _fee_prob(r["venue"], entry_price(r))
        for c, r in zip(gross, rows)
    ]
    by_game: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_game.setdefault(_game(r), []).append(i)
    games = len(by_game)

    def agg(vals: list[float]) -> dict:
        ci = _bootstrap_ci([(sum(vals[i] for i in idx), float(len(idx))) for idx in by_game.values()])
        return {
            "mean_pp": _round(sum(vals) / len(vals), 3),
            "ci95_pp": [round(ci[0], 3), round(ci[1], 3)] if ci else None,
            "frac_positive": _round(sum(1 for v in vals if v > 0) / len(vals), 3),
        }

    g, n = agg(gross), agg(net)
    return {
        "n_rows": len(rows),
        "n_games": games,
        "gross": g,
        "net_of_fees": n,
        # Gate = the shared promotion gate (n counted in GAMES).
        "clears_gross": clv_clears(games, g["mean_pp"], g["frac_positive"]),
        "clears_net": clv_clears(games, n["mean_pp"], n["frac_positive"]),
        "gate": {
            "min_games": MIN_CLV_RESOLVED,
            "min_mean_pp": CLV_MIN_MEAN_PP,
            "min_frac_positive": CLV_MIN_FRAC_POSITIVE,
        },
    }


_EMPTY_REASON = {
    "scope": "no rows match the spec",
    "resolved": "rows exist but none resolved yet (future / pending games)",
    "not_voided": "every resolved row is voided (cancelled / legacy market)",
    "pre_game_entry": "every row was logged at/after tip-off (in-play)",
    "valid_entry_price": "no usable entry price",
    "current_code": "every row is from a superseded code state",
    "live_eligible": "no row could log live (partial blend / sharp-only)",
    "clv_present": "no close captured yet (kalshi_clv_pct is NULL)",
}


def _empty_reason(funnel: list[tuple[str, int]], upto: str) -> Optional[str]:
    """Why a population is empty: the first funnel stage that hit zero."""
    for name, n in funnel:
        if n == 0:
            return _EMPTY_REASON.get(name, f"stage {name} left 0 rows (stale or missing close snapshot)")
        if name == upto:
            break
    return None


def _price_kind(q: float) -> str:
    """How a stored placed_price was entered (the DB does not record it).

    CLI fills are typed in whole cents (raw ask, fee NOT included). Dashboard
    fills are 1/decimal-odds at 2dp from fee-inclusive displayed odds. A price
    that is both (0.5, 0.25...) is classed round_cent (raw), the conservative
    reading.
    """
    if abs(q * 100 - round(q * 100)) < 1e-6:
        return "round_cent_raw"
    if abs(1.0 / q - round(1.0 / q, 2)) < 1e-6:
        return "decimal_odds_fee_inclusive"
    return "unknown_treated_raw"


def realized_block(scope_rows: list[dict]) -> dict:
    """Realized dollars on bets actually placed (placed=1), at the real stake.

    Population is independent of the live-eligible / current-code gates: money
    was bet whatever the code state. Cancelled voids are refunded and dropped.
    Because placed_price may or may not include the fee, three P&Ls are given:
    ``pnl_raw`` (fee added on top of the price), ``pnl_fee_inclusive`` (price
    already carries the fee) and ``pnl_est`` (each row by `_price_kind`).
    """
    from evmax.fees import bet_pnl

    placed = [r for r in scope_rows if r["placed"]]
    funnel = [("placed", len(placed))]
    placed = [r for r in placed if r["outcome"] is not None]
    funnel.append(("resolved", len(placed)))
    placed = [r for r in placed if not r["voided"] or r["void_reason"] == "stale_reverted"]
    funnel.append(("not_voided", len(placed)))
    placed = [
        r for r in placed
        if r["placed_stake"] and r["placed_stake"] > 0
        and r["placed_price"] is not None and 0 < r["placed_price"] < 1
    ]
    funnel.append(("valid_stake_and_price", len(placed)))
    out: dict = {"funnel": [{"stage": n, "n_rows": c} for n, c in funnel]}
    if not placed:
        first_zero = next(n for n, c in funnel if c == 0)
        out.update(n_bets=0, empty_reason=f"no placed bets survive stage '{first_zero}'")
        return out

    def pnls(r: dict) -> tuple[float, float, float]:
        S, q, won = r["placed_stake"], r["placed_price"], r["outcome"] == 1
        v = r["venue"] or "kalshi"
        raw = bet_pnl(S, q, won, venue=v, maker=bool(r["maker_fill"]))
        incl = bet_pnl(S, q, won, venue=None)
        est = incl if _price_kind(q) == "decimal_odds_fee_inclusive" else raw
        return raw, incl, est

    vals = [pnls(r) for r in placed]
    staked = sum(r["placed_stake"] for r in placed)
    by_game: dict[str, list[int]] = {}
    for i, r in enumerate(placed):
        by_game.setdefault(_game(r), []).append(i)
    ci = _bootstrap_ci([
        (sum(vals[i][2] for i in idx), sum(placed[i]["placed_stake"] for i in idx))
        for idx in by_game.values()
    ])
    kinds: dict[str, int] = {}
    for r in placed:
        k = _price_kind(r["placed_price"])
        kinds[k] = kinds.get(k, 0) + 1
    slip = [
        100 * (r["placed_price"] - r["kalshi_yes_price"])
        for r in placed if r["kalshi_yes_price"] is not None
    ]
    out.update(
        n_bets=len(placed),
        n_games=len(by_game),
        staked_usd=round(staked, 2),
        pnl_est_usd=round(sum(v[2] for v in vals), 2),
        roi_est_on_stake_pct=_round(100 * sum(v[2] for v in vals) / staked, 2),
        roi_est_ci95_pct=[round(100 * ci[0], 2), round(100 * ci[1], 2)] if ci else None,
        pnl_raw_usd=round(sum(v[0] for v in vals), 2),
        pnl_fee_inclusive_usd=round(sum(v[1] for v in vals), 2),
        price_kinds=kinds,
        maker_fills=sum(1 for r in placed if r["maker_fill"]),
        mean_fill_minus_scan_ask_pp=_round(sum(slip) / len(slip), 2) if slip else None,
        row_digest=_digest(placed),
    )
    return out


def _digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(r["market_id"] for r in rows)).encode()).hexdigest()[:16]


def evaluate_rows(scope_rows: list[dict], spec: EvalSpec) -> dict:
    """Evaluate ONE group of scope rows (already SQL-filtered by `spec`)."""
    roi_rows, funnel = _roi_stages(scope_rows, spec)

    clv_rows = [r for r in roi_rows if r["kalshi_clv_pct"] is not None]
    funnel.append(("clv_present", len(clv_rows)))
    clv_uncapped = clv_rows
    if spec.max_staleness_h is not None:
        clv_rows, dropped = _fresh_close(clv_rows, spec.max_staleness_h)
        funnel.append((f"close_fresh_{spec.max_staleness_h:g}h", len(clv_rows)))

    prev = None
    funnel_out = []
    for name, n in funnel:
        funnel_out.append({"stage": name, "n_rows": n, "dropped": None if prev is None else prev - n})
        # ROI and CLV populations are two branches: `clv_present` is a subset
        # of the last ROI stage, so its drop is measured against it.
        prev = n
    return {
        "funnel": funnel_out,
        "roi_population": conform(
            {"row_digest": _digest(roi_rows), **_roi_block(roi_rows), **_brier_block(roi_rows)},
            ROI_TEMPLATE, "roi_population",
        ),
        "clv_population": conform(
            {
                "row_digest": _digest(clv_rows),
                **_clv_block(clv_rows),
                # Sensitivity to the staleness cap: the same CLV with NO cap, so a
                # halved sample can be judged for bias instead of guessed at.
                "uncapped": (
                    conform({"row_digest": _digest(clv_uncapped), **_clv_block(clv_uncapped)},
                            CLV_BASE, "clv_population.uncapped")
                    if spec.max_staleness_h is not None else None
                ),
                "empty_reason": _empty_reason(funnel, funnel[-1][0]) if not clv_rows else None,
            },
            CLV_TEMPLATE, "clv_population",
        ),
        "sector_rules": [
            {"sector": sec, **sector_rules(sec)}
            for sec in sorted({r["sector"] or "?" for r in scope_rows})
        ],
        "roi_empty_reason": (
            _empty_reason(funnel, "live_eligible" if spec.population == "gate" else funnel[-1][0])
            if not roi_rows else None
        ),
        "realized": conform(realized_block(scope_rows), REALIZED_TEMPLATE, "realized"),
        "_roi_rows": roi_rows,
        "_clv_rows": clv_rows,
    }


def _db_fingerprint(conn=None) -> dict:
    from evmax.agents.cleanup.db import get_connection

    sql = """SELECT (SELECT COUNT(*) FROM ev_predictions),
                    (SELECT COUNT(*) FROM ev_outcomes WHERE outcome IS NOT NULL),
                    (SELECT MAX(resolved_at) FROM ev_outcomes)"""
    if conn is not None:
        a, b, c = conn.execute(sql).fetchone()
    else:
        with get_connection() as cx:
            a, b, c = cx.execute(sql).fetchone()
    return {"predictions": a, "resolved": b, "last_resolved_at": c}


def run_eval(spec: EvalSpec, conn=None) -> dict:
    """Public entrypoint: the one evaluation every report must come from."""
    scope = fetch_scope_rows(spec, conn)
    groups: dict[tuple, list[dict]] = {}
    for r in scope:
        key = tuple((r[g] or ("kalshi" if g == "venue" else "?")) for g in spec.group_by)
        groups.setdefault(key, []).append(r)
    if not spec.group_by:
        groups = {(): scope}

    results = []
    for key in sorted(groups):
        res = evaluate_rows(groups[key], spec)
        res.pop("_roi_rows")
        res.pop("_clv_rows")
        results.append({"group": dict(zip(spec.group_by, key)), **res})
    return {
        "spec": asdict(spec),
        "db": _db_fingerprint(conn),
        "definitions": {
            "roi": "flat $1 notional per row on the row's OWN side at entry price "
                   "(placed fill else scan ask); win = ev_outcomes.outcome==1; "
                   "net = evmax.fees.bet_pnl with the row's venue (taker)",
            "clv": "ev_predictions.kalshi_clv_pct (pp, entry->close); net subtracts "
                   "100*venue_fee_prob(entry); gate counts GAMES",
            "n_games": "distinct shadow.game_key(event_id)",
            "window": "event_date inclusive",
        },
        "groups": results,
    }


def to_json(result: dict) -> str:
    return json.dumps(result, indent=2, sort_keys=True)


def export_rows(spec: EvalSpec, which: str = "roi", conn=None) -> list[dict]:
    """The exact scored rows (single group = whole spec), for audit/CSV."""
    res = evaluate_rows(fetch_scope_rows(spec, conn), spec)
    return res["_roi_rows" if which == "roi" else "_clv_rows"]
