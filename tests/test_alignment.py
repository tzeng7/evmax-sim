"""YES-side alignment (evmax/matching/alignment.py).

Regression suite for the 2026-09-05 NCAAF incident: the EV agent's fuzzy
YES-label matcher scored "western michigan" vs "michigan" at 100 and, testing
outcome A first, priced the Western Michigan YES contract at Michigan's 97%
(+2145% EV). Alignment is now canonical-equality / unique-token-subset /
closed-world venue-code resolution, and unresolvable ⇒ not priced.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.agents.odds.ev_gap_agent import EVGapAgent
from evmax.matching.alignment import (
    Alignment,
    YesOutcome,
    align_yes_side,
    alignment_looks_suspect,
    resolve_side,
)
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def _market(sector, home, away, yes, *, ask=0.10, mt=MarketType.moneyline):
    no_price = round(min(0.99, max(0.01, (1.0 - ask) + 0.02)), 4)
    return PredictionMarket(
        id=f"kalshi:{home}_vs_{away}:{yes}", source=MarketSource.kalshi, sector=sector,
        market_type=mt, title=f"{home} vs {away}", ticker="T", yes_price=ask, no_price=no_price,
        volume_usd=10_000.0, open_interest_usd=5_000.0, team_home=home, team_away=away,
        yes_team=yes, event_date=datetime(2026, 9, 5, 20, 0, tzinfo=timezone.utc),
    )


def _sharp(sector, a, b, pa=0.95, pb=0.05, draw=None, over=None, under=None, total_line=None):
    return SharpOdds(
        event_id=f"{sector}::2026-09-05::{a}_vs_{b}", book=SharpBook.pinnacle, sector=sector,
        outcome_a_label=a, outcome_b_label=b, outcome_a_decimal=1 / pa, outcome_b_decimal=1 / pb,
        true_prob_a=pa, true_prob_b=pb, true_prob_draw=draw, true_prob_over=over,
        true_prob_under=under, total_line=total_line, margin=0.04,
        event_date=datetime(2026, 9, 5, 20, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Golden containment table — one team's tokens are a subset of the other's.
# Both YES contracts of every pair must align to their OWN side.
# ---------------------------------------------------------------------------

CONTAINMENT_PAIRS = [
    ("ncaaf", "Michigan", "Western Michigan"),
    ("ncaaf", "Florida", "Florida Atlantic"),
    ("ncaaf", "Washington", "Washington State"),
    ("ncaaf", "Arizona", "Northern Arizona"),
    ("ncaab", "Michigan", "Michigan State"),
    ("ncaab", "Ohio State", "Ohio"),
    ("ncaab", "Kentucky", "Western Kentucky"),
    ("ncaab", "Texas", "Texas State"),
    ("ncaaw", "Miami", "Miami (OH)"),
    ("soccer", "Manchester City", "Manchester United"),
    ("nfl", "Los Angeles Rams", "Los Angeles Chargers"),
    ("nfl", "New York Giants", "New York Jets"),
]


@pytest.mark.parametrize("sector,home,away", CONTAINMENT_PAIRS)
def test_containment_pairs_align_to_own_side(sector, home, away):
    norm = NameNormalizer(sector)
    sharp = _sharp(sector, home, away)
    for yes, want in ((home, YesOutcome.A), (away, YesOutcome.B)):
        al = align_yes_side(_market(sector, home, away, yes), sharp, norm)
        assert al is not None, f"{sector}: {yes!r} unresolved"
        assert al.outcome is want, f"{sector}: {yes!r} aligned to {al.outcome} via {al.method}"


def test_western_michigan_regression_gets_underdog_prob():
    """The exact incident row: WMU YES at 4¢, Michigan 97.2% / WMU 2.8%."""
    norm = NameNormalizer("ncaaf")
    sharp = _sharp("ncaaf", "Michigan", "Western Michigan", pa=0.972, pb=0.028)
    al = align_yes_side(_market("ncaaf", "Michigan", "Western Michigan", "western michigan", ask=0.04), sharp, norm)
    assert al.outcome is YesOutcome.B
    assert al.sharp_prob(sharp) == pytest.approx(0.028)


def test_yes_label_shared_by_both_sides_is_ambiguous():
    """'florida' vs {South Florida, Florida International}: two subset hits ⇒ None."""
    norm = NameNormalizer("ncaaf")
    sharp = _sharp("ncaaf", "South Florida", "Florida International")
    m = _market("ncaaf", "South Florida", "Florida International", "florida")
    # The venue-team step can't disambiguate either (same two names) and ncaaf
    # has no price fallback.
    assert align_yes_side(m, sharp, norm) is None


# ---------------------------------------------------------------------------
# resolve_side rules
# ---------------------------------------------------------------------------

def test_resolve_side_exact_beats_subset():
    assert resolve_side("michigan", "michigan", "western michigan") == (YesOutcome.A, "canonical")
    assert resolve_side("western michigan", "michigan", "western michigan") == (YesOutcome.B, "canonical")


def test_resolve_side_token_subset_unique():
    assert resolve_side("sinner", "jannik sinner", "carlos alcaraz") == (YesOutcome.A, "tokens")


def test_resolve_side_token_subset_ambiguous_returns_none():
    # Same-surname draw: never guess (tennis rule).
    assert resolve_side("zverev", "alexander zverev", "mischa zverev") is None


def test_resolve_side_short_label_never_subset_matches():
    assert resolve_side("m", "mainz 05", "hoffenheim") is None


# ---------------------------------------------------------------------------
# Player sectors: surname YES labels vs full-name sharp labels
# ---------------------------------------------------------------------------

def test_tennis_surname_aligns():
    norm = NameNormalizer("tennis")
    sharp = _sharp("tennis", "Jonas Forejtek", "Kyrian Jacquet", pa=0.6, pb=0.4)
    al = align_yes_side(_market("tennis", "Jonas Forejtek", "Kyrian Jacquet", "jacquet", ask=0.4), sharp, norm)
    assert al is not None and al.outcome is YesOutcome.B


def test_ufc_surname_aligns():
    norm = NameNormalizer("ufc")
    sharp = _sharp("ufc", "Kamaru Usman", "Dricus Du Plessis", pa=0.45, pb=0.55)
    al = align_yes_side(_market("ufc", "Kamaru Usman", "Dricus Du Plessis", "usman", ask=0.45), sharp, norm)
    assert al is not None and al.outcome is YesOutcome.A


# ---------------------------------------------------------------------------
# Venue-team step: ticker codes resolved against the market's OWN teams
# ---------------------------------------------------------------------------

def test_esports_code_prefix_resolves_via_venue_teams():
    norm = NameNormalizer("lol")
    sharp = _sharp("lol", "FlyQuest", "LYON", pa=0.7, pb=0.3)
    al = align_yes_side(_market("lol", "FlyQuest", "LYON", "ly", ask=0.3), sharp, norm)
    assert al is not None and al.outcome is YesOutcome.B and al.method == "venue_team"


def test_esports_unmatchable_code_uses_price_fallback_only_when_decisive():
    norm = NameNormalizer("lol")
    sharp = _sharp("lol", "Disguised", "FlyQuest", pa=0.7, pb=0.3)
    # decisive: ask 0.31 sits on b
    al = align_yes_side(_market("lol", "Disguised", "FlyQuest", "dsg", ask=0.31), sharp, norm)
    assert al is not None and al.outcome is YesOutcome.B and al.method == "price"
    # near coin flip: refused
    sharp2 = _sharp("lol", "Disguised", "FlyQuest", pa=0.52, pb=0.48)
    assert align_yes_side(_market("lol", "Disguised", "FlyQuest", "dsg", ask=0.50), sharp2, norm) is None


def test_price_fallback_disabled_outside_esports():
    norm = NameNormalizer("nba")
    sharp = _sharp("nba", "Lakers", "Celtics", pa=0.58, pb=0.42)
    assert align_yes_side(_market("nba", "Lakers", "Celtics", "xxx", ask=0.60), sharp, norm) is None


# ---------------------------------------------------------------------------
# Totals / draws
# ---------------------------------------------------------------------------

def test_total_alignment():
    norm = NameNormalizer("nba")
    sharp = _sharp("nba", "over", "under", over=0.5, under=0.5, total_line=220.5)
    al = align_yes_side(_market("nba", "Lakers", "Celtics", "under", mt=MarketType.total), sharp, norm)
    assert al.outcome is YesOutcome.UNDER and al.is_under
    al = align_yes_side(_market("nba", "Lakers", "Celtics", "over", mt=MarketType.total), sharp, norm)
    assert al.outcome is YesOutcome.OVER


def test_total_without_sharp_probs_is_unaligned():
    norm = NameNormalizer("nba")
    sharp = _sharp("nba", "over", "under", total_line=220.5)
    assert align_yes_side(_market("nba", "Lakers", "Celtics", "over", mt=MarketType.total), sharp, norm) is None


def test_draw_alignment():
    norm = NameNormalizer("soccer")
    sharp = _sharp("soccer", "Arsenal", "Chelsea", pa=0.5, pb=0.25, draw=0.25)
    al = align_yes_side(_market("soccer", "Arsenal", "Chelsea", "tie"), sharp, norm)
    assert al.outcome is YesOutcome.DRAW and al.sharp_prob(sharp) == 0.25
    assert align_yes_side(_market("soccer", "Arsenal", "Chelsea", "tie"), _sharp("soccer", "Arsenal", "Chelsea"), norm) is None


# ---------------------------------------------------------------------------
# Sanity flag
# ---------------------------------------------------------------------------

def test_suspect_flag_fires_when_other_side_sits_on_ask():
    sharp = _sharp("ncaaf", "Michigan", "Western Michigan", pa=0.972, pb=0.028)
    wrong = Alignment(YesOutcome.A, "canonical", "michigan")
    assert alignment_looks_suspect(wrong, 0.04, sharp)
    right = Alignment(YesOutcome.B, "canonical", "western michigan")
    assert not alignment_looks_suspect(right, 0.04, sharp)


# ---------------------------------------------------------------------------
# End-to-end through the EV agent
# ---------------------------------------------------------------------------

def test_ev_agent_prices_underdog_yes_at_underdog_prob():
    agent = EVGapAgent()
    sharp = _sharp("ncaaf", "Michigan", "Western Michigan", pa=0.972, pb=0.028)
    m = _market("ncaaf", "Michigan", "Western Michigan", "western michigan", ask=0.04)
    gap, payload = agent._evaluate_pair(
        m, sharp, 100.0, "ncaaf", blended_preds={}, injuries={}, model_sources={}, return_blend=True,
    )
    assert payload is not None
    assert payload["yes_is_outcome_b"] is True
    # sharp prob for the YES side is the underdog's, so no four-digit EV
    assert gap is None or gap.ev_pct < 1.0


def test_ev_agent_drops_unalignable_market_and_counts_it():
    agent = EVGapAgent()
    sharp = _sharp("nba", "Lakers", "Celtics", pa=0.58, pb=0.42)
    m = _market("nba", "Lakers", "Celtics", "xxx", ask=0.60)
    gap, payload = agent._evaluate_pair(
        m, sharp, 100.0, "nba", blended_preds={}, injuries={}, model_sources={}, return_blend=True,
    )
    assert gap is None and payload is None
    assert agent._alignment_failures["nba"] == 1
