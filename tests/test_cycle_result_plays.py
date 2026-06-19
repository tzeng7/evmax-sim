"""CycleResult.plays() / loggable_gaps() — the single canonical definition of
an "actionable play" that the CLI scan, dashboard, and portfolio scans all
consume so the full_blend/prop filters can't drift between surfaces."""

from __future__ import annotations

from datetime import date, datetime

from evmax.agents.coordinator import CycleResult
from evmax.agents.odds.ev_gap_agent import EVGap


def _gap(market_id, *, full_blend=True, market_type="moneyline", ev=0.05,
         event_id=None, kelly=0.02):
    return EVGap(
        market_id=market_id,
        event_id=event_id or f"evt:{market_id}",
        sector="nba",
        yes_team="LAL",
        market_type=market_type,
        kalshi_yes_price=0.45,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=ev,
        kelly_full=0.04,
        kelly_fraction=kelly,
        match_confidence=0.95,
        volume_usd=5000.0,
        spread_pct=0.02,
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources="elo+sharp",
        line=None,
        event_title="Lakers vs Warriors",
        full_blend=full_blend,
    )


def _cycle(gaps):
    return CycleResult(ev_gaps=gaps, bankroll=500.0, kelly_fraction=0.5)


def test_plays_drops_partial_blend_by_default():
    full = _gap("full", full_blend=True)
    partial = _gap("partial", full_blend=False)
    cycle = _cycle([full, partial])
    ids = {g.market_id for g in cycle.plays()}
    assert ids == {"full"}


def test_plays_can_keep_partial_blend():
    cycle = _cycle([_gap("full"), _gap("partial", full_blend=False)])
    ids = {g.market_id for g in cycle.plays(require_full_blend=False)}
    assert ids == {"full", "partial"}


def test_plays_sorted_by_ev_desc():
    cycle = _cycle([_gap("low", ev=0.02), _gap("high", ev=0.20), _gap("mid", ev=0.08)])
    assert [g.market_id for g in cycle.plays()] == ["high", "mid", "low"]


def test_plays_drop_props_and_map_handicap():
    game = _gap("game", market_type="moneyline")
    prop = _gap("prop", market_type="player_prop",
                event_id="nba::2026-04-15::prop::LeBron::points::24.5")
    mh = _gap("mh", market_type="map_handicap")
    cycle = _cycle([game, prop, mh])
    ids = {g.market_id for g in cycle.plays(drop_props=True, drop_map_handicap=True)}
    assert ids == {"game"}


def test_loggable_gaps_keeps_partial_blend_but_drops_props():
    full = _gap("full", full_blend=True)
    partial = _gap("partial", full_blend=False)
    prop = _gap("prop", market_type="player_prop",
                event_id="nba::2026-04-15::prop::LeBron::points::24.5")
    cycle = _cycle([full, partial, prop])
    ids = {g.market_id for g in cycle.loggable_gaps()}
    # Partial-blend stays (logged as shadow); props are routed elsewhere.
    assert ids == {"full", "partial"}
