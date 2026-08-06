"""select_display_gaps() — the scan table's play-selection cap.

Regression guard: every live game-market gap the scanner persists to
ev_predictions MUST be shown in the scan table. `--top` may cap prop rows, but
truncating a game play would surface it only in the dashboard's Open Positions
(which reads live ev_predictions rows directly), never in the scan — the bug
this test locks down.
"""

from __future__ import annotations

from datetime import date, datetime

from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.cli.commands.agents import select_display_gaps


def _gap(market_id, *, market_type="moneyline", ev=0.05):
    return EVGap(
        market_id=market_id,
        event_id=f"evt:{market_id}",
        sector="nba",
        yes_team="LAL",
        market_type=market_type,
        kalshi_yes_price=0.45,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=ev,
        kelly_full=0.04,
        kelly_fraction=0.02,
        match_confidence=0.95,
        volume_usd=5000.0,
        spread_pct=0.02,
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources="elo+sharp",
        line=None,
        event_title="Lakers vs Warriors",
        full_blend=True,
    )


def _prop_gap(market_id, *, ev=0.05):
    g = _gap(market_id, market_type="player_prop", ev=ev)
    g.event_id = f"evt::prop::{market_id}"
    return g


def test_game_plays_never_truncated_by_top():
    # 30 game plays, top=25 — every game play must survive (they're all logged live).
    games = [_gap(f"g{i}", ev=0.30 - i * 0.001) for i in range(30)]
    out = select_display_gaps(games, top=25, max_props=10)
    assert out == games  # order preserved, none dropped


def test_top_caps_only_prop_rows():
    games = [_gap(f"g{i}") for i in range(20)]
    props = [_prop_gap(f"p{i}") for i in range(10)]
    out = select_display_gaps(games + props, top=25, max_props=10)
    # 20 game plays shown in full; top leaves room for 5 props.
    assert out[:20] == games
    assert out[20:] == props[:5]


def test_max_props_still_bounds_props():
    games = [_gap("g0")]
    props = [_prop_gap(f"p{i}") for i in range(20)]
    out = select_display_gaps(games + props, top=100, max_props=8)
    assert len(out) == 1 + 8  # game + max_props cap, not top-limited


def test_no_props_shows_all_games():
    games = [_gap(f"g{i}") for i in range(5)]
    out = select_display_gaps(games, top=25, max_props=10)
    assert out == games


def test_game_count_over_top_leaves_no_prop_room():
    games = [_gap(f"g{i}") for i in range(30)]
    props = [_prop_gap(f"p{i}") for i in range(5)]
    out = select_display_gaps(games + props, top=25, max_props=10)
    # All game plays shown; no room for props once games exceed top.
    assert out == games
