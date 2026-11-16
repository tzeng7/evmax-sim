"""Live anchored-entry trigger for laddered (spread/total) markets.

Turns one watch-listings sweep's in-memory data — Kalshi markets, order-book
depth, fresh devigged Pinnacle odds — into shadow-loggable EVGaps priced at
the CROSSABLE order-book price, the entry point the Track A candlestick
backfill validated (WNBA spread lay: +1.32pp mean Kalshi CLV over 116
declustered games, p<0.001; see scripts/eval_wnba_anchored_backfill.py).

Why this exists: the regular scanner logs `captured_yes_price` at scan time,
a median ~23h after listing — after Kalshi has already converged toward the
sharp line — so the live shadow stream could never reproduce the anchored
backtest. This module logs the entry the moment it first becomes measurable:
the first hourly sweep where a concurrent Pinnacle anchor can price the
ticker's line AND the edge/depth gates clear. `ev_predictions`'s
UNIQUE(market_id) freeze-on-first-insert then IS the first-anchored-sweep
rule — later sweeps are no-ops.

Row separation: every gap's model_sources carries an `anchored_entry` token
so `cleanup shadow clv --sources-token anchored_entry` scores this stream in
isolation from historical scan-time rows.

Pure function over in-memory inputs — no fetches, no DB reads. Persistence
happens in the caller (cleanup.py watch-listings --log-entries) via
log_gaps(mode_resolver=lambda cat: "shadow").
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from evmax.agents.cleanup.listings_eval import (
    _fair_yes_total,
    _sharp_game_key,
    _spread_pred_yes,
)
from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.ev.calculator import calculate_ev, effective_price
from evmax.models.market import MarketType, PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.sectors.registry import get_handler
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

# Matches the Track A evaluation gates (eval_wnba_anchored_backfill.py).
DEFAULT_EV_MIN_PP = 2.0
DEFAULT_DEPTH_MIN_USD = 50.0
# Skip imminent/in-play markets: near tip the "anchored entry" story no longer
# applies (that region is the scan-time stream Track A superseded).
DEFAULT_MIN_LEAD_MINUTES = 20.0

SOURCES_TOKEN = "anchored_entry"

# ---------------------------------------------------------------------------
# Early-totals stream (2026-11-16, shadow only)
#
# A total rung Pinnacle does NOT post at exactly that line has no anchor price
# (the stream above borrows a neighbour within half a point, or skips it). For
# the sectors below such a rung is priced by the fitted LinePMF
# (models_ml/line_pmf.py, token `total_pmf`) off Pinnacle's most balanced rung,
# and logged when the net-of-fee edge is >= EARLY_TOTAL_EDGE_MIN_PP and kickoff
# is >= EARLY_TOTAL_MIN_LEAD_H away.
#
# Evidence (market-tape replay 2026-11-16, Kalshi NFL totals, CLV to the venue
# close net of the taker fee, game-clustered 95% CI): edge >= 1pp and lead
# >= 24h -> fit +2.80pp [+1.51, +4.12] (105 bets / 38 games), holdout from
# 2026-10-20 +1.11pp [+0.06, +2.16] (71 / 33). Inside 24h the same rule is flat
# (-0.60 / +0.49). On rungs Pinnacle posts, the holdout is +0.30 [-2.56, +1.98].
# One season, CLV not realized profit, and the replay had no depth gate.
#
# WARNING (same day): re-scored on the archived order book, the replay does NOT
# hold. The book's crossable price was above the archived market-list price in
# 58% of the triggered entries (mean +2.6pp); entries that keep a 1pp edge at
# the book price have book-to-book CLV of +0.56 [-0.74, +1.72] (fit) and
# -0.18 [-0.80, +0.45] (holdout). This stream prices at the book, so it is the
# honest forward check; expect few rows and a result near zero.
#
# Rows carry `early_total` INSTEAD of `anchored_entry`, so the 1pp stream never
# pools into the 2pp anchored-entry gate: read it with
# `cleanup shadow clv nfl -m total --venue kalshi --sources-token early_total`.
# ---------------------------------------------------------------------------
EARLY_TOTAL_SECTORS: frozenset[str] = frozenset({"nfl"})
EARLY_TOTAL_TOKEN = "early_total"
EARLY_TOTAL_EDGE_MIN_PP = 1.0
EARLY_TOTAL_MIN_LEAD_H = 24.0


def _model_total_fair(ladder: dict, yes_side: str, line: float, sector: str) -> Optional[float]:
    """P(YES) for a total rung Pinnacle does not post, from the fitted LinePMF.

    Anchored on the most balanced rung of Pinnacle's ladder (its main total).
    Returns None unless the fitted model priced it: the normal-curve fallback
    is not the pricing the replay validated.
    """
    if not ladder:
        return None
    from evmax.models.odds import SharpBook
    from evmax.models_ml.total_distribution import TOTAL_PMF_TOKEN, TotalDistributionModel

    main_line, (p_over, p_under) = min(ladder.items(), key=lambda kv: abs(kv[1][0] - 0.5))
    if p_over is None or p_under is None:
        return None
    so = SharpOdds(
        event_id="early_total::total", book=SharpBook.pinnacle, sector=sector,
        outcome_a_label="over", outcome_b_label="under",
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_over=p_over, true_prob_under=p_under, total_line=main_line,
    )
    pred = TotalDistributionModel().predict(
        so, target_line=line, sector=sector, yes_is_under=(yes_side == "under"))
    if pred is None or pred.method != TOTAL_PMF_TOKEN:
        return None
    return pred.true_prob


def _group_sharp(sharp_odds: list[SharpOdds]) -> dict:
    """{game_key: {"tip": dt, "spread": SharpOdds|None, "totals": {line: (p_o, p_u)}}}.

    Unlike the offline evaluator there is no time series to walk — everything
    in one sweep is concurrent by construction, so each game carries at most
    one spread anchor and one totals ladder.
    """
    games: dict = {}
    for so in sharp_odds:
        key = _sharp_game_key(so.event_id)
        if key is None or so.event_date is None:
            continue
        g = games.setdefault(key, {"tip": None, "spread": None, "totals": {}, "event_id": None})
        tip = so.event_date if so.event_date.tzinfo else so.event_date.replace(tzinfo=timezone.utc)
        if g["tip"] is None or tip < g["tip"]:
            g["tip"] = tip
        if "::total::" in so.event_id and so.total_line is not None:
            if so.true_prob_over is not None:
                g["totals"][so.total_line] = (so.true_prob_over, so.true_prob_under)
        elif so.event_id.endswith("::spread") and so.spread_line is not None:
            g["spread"] = so
            g["event_id"] = so.event_id
    return games


def _minutes_to_tip(tip: datetime, now: datetime) -> int:
    return int((tip - now).total_seconds() / 60)


def build_anchored_entries(
    markets: list[PredictionMarket],
    books: dict[str, Optional[dict]],
    sharp_odds: list[SharpOdds],
    sector: str,
    now: Optional[datetime] = None,
    ev_min_pp: float = DEFAULT_EV_MIN_PP,
    depth_min_usd: float = DEFAULT_DEPTH_MIN_USD,
    min_lead_minutes: float = DEFAULT_MIN_LEAD_MINUTES,
) -> list[EVGap]:
    """Anchored-entry EVGaps for one sweep. See module docstring.

    Both sides of every ticker are evaluated independently, mirroring the
    scanner's conventions exactly so resolution/CLV plumbing needs no changes:

      YES side: crossable price = order-book yes_ask; EVGap.kalshi_yes_price
                is set to that ask (log_gaps copies it into
                captured_yes_price, which backfill_clv anchors CLV to).
      NO side:  crossable price = 1 - order-book yes_bid; emitted with the
                scanner's NO-side shape — market_id "{id}:no", spread line
                negated with yes_team = opponent, totals yes_team flipped
                with the line unflipped.

    Entries without an order-book row fall back to the market snapshot's
    yes_price for the YES ask and skip the NO side (no bid visible).
    """
    now = now or datetime.now(timezone.utc)
    handler = get_handler(sector)
    normalize = lambda s: handler.normalize_team(s or "")  # noqa: E731
    anchors = _group_sharp(sharp_odds)

    # This stream is Kalshi laddered spread/total (WNBA + NFL per --entry-sectors;
    # sector-generic here). Price the crossable ask net of the Kalshi trading fee
    # (same rule as the scanner) unless fees are toggled off, so the anchored-entry
    # EV gate matches the live pricing path.
    fee_venue = "kalshi" if get_settings().fees_in_pricing else None

    gaps: list[EVGap] = []
    for m in markets:
        mt = m.market_type.value if hasattr(m.market_type, "value") else str(m.market_type)
        if mt not in ("spread", "total") or m.line is None or m.event_date is None:
            continue
        home = normalize(m.team_home)
        away = normalize(m.team_away)
        if not home or not away:
            continue
        ed = m.event_date if m.event_date.tzinfo else m.event_date.replace(tzinfo=timezone.utc)
        game = (ed.date().isoformat(), frozenset((home, away)))
        g = anchors.get(game)
        if g is None or g["tip"] is None:
            continue
        tip = g["tip"]
        if _minutes_to_tip(tip, now) < min_lead_minutes:
            continue

        yes_team = (m.yes_team or "").strip().lower()
        stream_token, edge_min = SOURCES_TOKEN, ev_min_pp
        if mt == "spread":
            if g["spread"] is None:
                continue
            pred = _spread_pred_yes(g["spread"], sector, yes_team, m.line, normalize)
            fair = pred.true_prob if pred else None
            # spread_dist (normal CDF) or spread_pmf (NFL key-number PMF): the
            # token tells the CLV gate which pricing selected the entry.
            sources = f"sharp+{pred.method}" if pred else "sharp+spread_dist"
        else:
            # Early-totals stream: a rung Pinnacle does not post at EXACTLY this
            # line, far enough from kickoff, is priced by the fitted model. It
            # takes precedence over the half-point-neighbour price below (a
            # neighbour is a different bet across an NFL key total).
            posted = any(abs(k - m.line) < 0.01 for k in g["totals"])
            early = (
                sector.lower() in EARLY_TOTAL_SECTORS and not posted
                and _minutes_to_tip(tip, now) >= EARLY_TOTAL_MIN_LEAD_H * 60
            )
            model_fair = (
                _model_total_fair(g["totals"], yes_team, m.line, sector) if early else None
            )
            if model_fair is not None:
                fair = model_fair
                sources = "sharp+total_pmf"
                stream_token = EARLY_TOTAL_TOKEN
                edge_min = EARLY_TOTAL_EDGE_MIN_PP
            else:
                fair = _fair_yes_total(g["totals"], yes_team, m.line)
                sources = "sharp"
        if fair is None or not (0 < fair < 1):
            continue

        d = books.get(m.ticker)
        ask = d["yes_ask"] if d and d.get("yes_ask") is not None else m.yes_price
        bid = d["yes_bid"] if d else None
        so = g["spread"]
        title = (
            f"{so.outcome_a_label} vs {so.outcome_b_label}"
            if so is not None and so.outcome_a_label
            else f"{(m.team_away or '?')} vs {(m.team_home or '?')}"
        )
        common = dict(
            event_id=g["event_id"] or f"{sector}::{game[0]}::{'_vs_'.join(sorted(game[1]))}::{mt}",
            sector=sector,
            market_type=mt,
            kelly_full=0.0,
            kelly_fraction=0.0,  # shadow-only stream, never bankroll-sized
            match_confidence=1.0,
            volume_usd=m.volume_usd,
            spread_pct=m.spread_pct,
            event_date=tip,
            minutes_to_tipoff=_minutes_to_tip(tip, now),
            event_title=title,
            venue="kalshi",
        )

        # YES side (spread lay / total over-as-listed)
        if ask is not None and 0 < ask < 1:
            # Gate the anchor edge and stored EV on the fee-inclusive ask, but
            # keep kalshi_yes_price = raw crossable ask (CLV anchors to it).
            eff_ask = effective_price(ask, fee_venue)
            edge = (fair - eff_ask) * 100
            depth_usd = (d.get("yes_ask_depth_usd") if d else None) or 0.0
            if edge >= edge_min and depth_usd >= depth_min_usd:
                ev, _ = calculate_ev(eff_ask, fair)
                gaps.append(EVGap(
                    market_id=m.id,
                    yes_team=yes_team,
                    kalshi_yes_price=ask,
                    sharp_true_prob=fair,
                    blended_true_prob=fair,  # pure anchor — no sim, by design
                    ev_pct=ev,
                    line=m.line,
                    model_sources=f"{sources}+{stream_token}",
                    **common,
                ))

        # NO side (spread take / total under)
        if bid is not None and 0 < bid < 1:
            no_price = 1.0 - bid
            eff_no_price = effective_price(no_price, fee_venue)
            edge = ((1.0 - fair) - eff_no_price) * 100  # fair_no - net no cost
            depth_usd = (d.get("yes_bid_depth_usd") if d else None) or 0.0
            if edge >= edge_min and depth_usd >= depth_min_usd and 0 < no_price < 1:
                ev, _ = calculate_ev(eff_no_price, 1.0 - fair)
                if mt == "spread":
                    no_team = next(iter(frozenset((home, away)) - {yes_team}), "?")
                    no_line = -m.line
                else:
                    no_team = "under" if yes_team == "over" else "over"
                    no_line = m.line
                gaps.append(EVGap(
                    market_id=f"{m.id}:no",
                    yes_team=no_team,
                    kalshi_yes_price=no_price,
                    sharp_true_prob=1.0 - fair,
                    blended_true_prob=1.0 - fair,
                    ev_pct=ev,
                    line=no_line,
                    model_sources=f"{sources}+no_side+{stream_token}",
                    **common,
                ))

    if gaps:
        logger.info("anchored_entries_built", sector=sector, count=len(gaps))
    return gaps
