"""Tests for the World Cup (national-team) sector.

Covers the four pieces that make WC a distinct sector from club soccer:
  - the alias map collapses FIFA codes + Pinnacle/ESPN name variants to one
    canonical per nation (incl. the USA noise-word edge case),
  - Kalshi KXWCGAME parsing yields teams, yes_team (incl. TIE), moneyline,
  - the Elo agent allocates draw mass and uses a neutral (0) home edge,
  - the registry + Pinnacle/resolver wiring is consistent.
"""

from __future__ import annotations

import pytest

from evmax.agents.models.elo_agent import HOME_ADVANTAGE_ELO, K_FACTORS, EloModelAgent
from evmax.clients.kalshi import SECTOR_SERIES_MAP, KalshiClient
from evmax.clients.esports_pinnacle import SECTOR_SPORT_LEAGUES
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType
from evmax.sectors.registry import get_handler


# ---------------------------------------------------------------------------
# Alias / normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        # FIFA 3-letter codes (from Kalshi tickers)
        ("cod", "dr congo"),
        ("iri", "iran"),
        ("kor", "south korea"),
        ("usa", "united states"),
        ("uzb", "uzbekistan"),
        ("tur", "turkiye"),
        ("arg", "argentina"),
        # Kalshi title full names that diverge from canonical
        ("Congo DR", "dr congo"),
        ("IR Iran", "iran"),
        ("Korea Republic", "south korea"),
        # Pinnacle full names
        ("DR Congo", "dr congo"),
        ("Iran", "iran"),
        ("South Korea", "south korea"),
        ("USA", "united states"),
        # ESPN full names (resolver path) — "United States" must NOT noise-strip
        # "united" down to "states"
        ("United States", "united states"),
        ("Bosnia-Herzegovina", "bosnia and herzegovina"),
        # draw outcome
        ("tie", "tie"),
    ],
)
def test_worldcup_normalization(raw, expected):
    nn = NameNormalizer("worldcup")
    assert nn.normalize(raw) == expected


def test_worldcup_handler_enrich_normalizes_codes():
    """enrich_market should canonicalize team_home/team_away from FIFA codes."""
    from evmax.models.market import MarketSource, PredictionMarket

    h = get_handler("worldcup")
    m = PredictionMarket(
        id="kalshi:test",
        source=MarketSource.kalshi,
        sector="worldcup",
        market_type=MarketType.moneyline,
        title="Congo DR vs Uzbekistan Winner?",
        ticker="KXWCGAME-26JUN27CODUZB-COD",
        yes_price=0.5,
        no_price=0.5,
        team_home="uzb",
        team_away="cod",
        yes_team="cod",
    )
    out = h.enrich_market(m)
    assert out.team_home == "uzbekistan"
    assert out.team_away == "dr congo"


def test_worldcup_kalshi_and_pinnacle_keys_agree():
    """A Kalshi WC market and the Pinnacle full-name spelling must produce the
    same canonical event key (the matching crux for the 3 divergent nations)."""
    h = get_handler("worldcup")
    # Kalshi parses codes; Pinnacle ships full names — both canonicalize equal.
    k = h.make_event_key("usa", "tur", "2026-06-25")          # codes
    p = h.make_event_key("USA", "Turkiye", "2026-06-25")      # Pinnacle names
    assert k == p
    assert "united_states" in k and "turkiye" in k


# ---------------------------------------------------------------------------
# Kalshi parsing
# ---------------------------------------------------------------------------

def _raw_wc_market(ticker: str, title: str, yes_price: float = 0.45) -> dict:
    return {
        "ticker": ticker,
        "title": title,
        "yes_bid_dollars": yes_price - 0.02,
        "yes_ask_dollars": yes_price,
        "no_bid_dollars": 1 - yes_price - 0.02,
        "no_ask_dollars": 1 - yes_price,
        "volume_fp": 1000,
        "open_interest_fp": 500,
    }


def test_parse_wc_moneyline_team_market():
    kc = KalshiClient.__new__(KalshiClient)
    raw = _raw_wc_market("KXWCGAME-26JUN27CODUZB-COD", "Congo DR vs Uzbekistan Winner?")
    m = kc._parse_market(raw, "worldcup")
    assert m is not None
    assert m.market_type == MarketType.moneyline
    # ticker {AWAY}{HOME} = COD UZB; outcome COD => yes = away team
    assert {m.team_home, m.team_away} == {"uzb", "cod"}
    assert m.yes_team == "dr congo"   # outcome code normalized via worldcup alias


def test_parse_wc_tie_market():
    kc = KalshiClient.__new__(KalshiClient)
    raw = _raw_wc_market("KXWCGAME-26JUN27CODUZB-TIE", "Congo DR vs Uzbekistan Winner?")
    m = kc._parse_market(raw, "worldcup")
    assert m is not None
    assert m.market_type == MarketType.moneyline
    assert m.yes_team == "tie"        # draw market — drives true_prob_draw downstream


# ---------------------------------------------------------------------------
# Elo agent
# ---------------------------------------------------------------------------

def test_worldcup_elo_constants():
    # National teams play sparsely → larger K; neutral venues → no home edge.
    assert K_FACTORS["worldcup"] == 40.0
    assert HOME_ADVANTAGE_ELO["worldcup"] == 0.0


def test_worldcup_elo_allocates_draw_and_is_neutral():
    agent = EloModelAgent()
    agent.seed_ratings("worldcup", {"spain": 1700.0, "panama": 1450.0})
    pa, pb, draw = agent._win_probs("worldcup", "spain", "panama")
    # 3-way: draw mass allocated, probabilities sum to 1
    assert draw is not None and draw > 0.0
    assert pa == pytest.approx(1.0 - pb - draw, abs=1e-6)
    # Neutral venue: swapping home/away (no home bonus) is symmetric.
    pa2, pb2, draw2 = agent._win_probs("worldcup", "panama", "spain")
    assert pa == pytest.approx(pb2, abs=1e-6)
    assert draw == pytest.approx(draw2, abs=1e-6)


# ---------------------------------------------------------------------------
# Registry / wiring consistency
# ---------------------------------------------------------------------------

def test_worldcup_registry_consistent():
    import evmax.categories as cats

    cats.validate_registry()  # raises on drift
    spec = next(c for c in cats.all_categories() if c.key == "worldcup")
    assert spec.mode == "shadow"
    assert set(spec.models) == {"elo", "sharp"}
    assert spec.market_types == (MarketType.moneyline,)


def test_worldcup_series_and_pinnacle_wired():
    assert SECTOR_SERIES_MAP["worldcup"] == ["KXWCGAME"]
    sport_id, league_ids = SECTOR_SPORT_LEAGUES["worldcup"]
    assert sport_id == 29 and league_ids == [2686]
