"""The per-league dimension inside the `soccer` sector.

Covers evmax/sectors/soccer_leagues.py (league identity), the tier config's
league-keyed lookups + per-tier disagreement ramp (soccer_tiers.py), the
venue parsers stamping PredictionMarket.league, the EVGap → ev_predictions
`league` column (logger + backfill), the ensemble's per-event ramp override,
and the shadow CLI / promotion board `league` filters.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.sectors import soccer_leagues as SL
from evmax.sectors import soccer_tiers


# ---------------------------------------------------------------------------
# League identity
# ---------------------------------------------------------------------------


def test_kalshi_series_map_matches_sector_series_map():
    """Every soccer series the scanner polls must resolve to a league, and no
    league mapping may reference a series the scanner doesn't poll."""
    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    assert set(SL.KALSHI_SERIES_LEAGUE) == set(SECTOR_SERIES_MAP["soccer"])
    assert set(SL.KALSHI_SERIES_LEAGUE.values()) == set(SL.SOCCER_LEAGUES)


def test_polymarket_betting_slugs_all_map():
    from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP

    for slug in POLYMARKET_US_LEAGUE_MAP["soccer"]:
        assert SL.league_for_polymarket_slug(slug) in SL.SOCCER_LEAGUES, slug
    assert SL.league_for_polymarket_slug("wnba") is None
    assert SL.league_for_polymarket_slug(None) is None


@pytest.mark.parametrize(
    "ticker, league",
    [
        ("KXEPLGAME-26APR24MANCHE-CHE", "epl"),
        ("kalshi:KXUCLGAME-26SEP16RMBAR-RMA", "ucl"),
        ("kalshi:KXMLSGAME-26JUL16MTLTOR-TOR:no", "mls"),
        ("KXNBAGAME-26APR24LALGSW-LAL", None),
        ("", None),
        (None, None),
    ],
)
def test_league_for_ticker(ticker, league):
    assert SL.league_for_ticker(ticker) == league


def test_league_for_market_id_polymarket_reads_slug_league_token():
    """PolyUS slugs embed the league as the SECOND dash token. Only that
    position is trusted: `sea` is Seattle in `...-aus-sea-...` but Serie A as
    a league slug."""
    assert SL.league_for_market_id("polymarket_us:atc-mls-nas-atl-2026-07-17-nas") == "mls"
    assert SL.league_for_market_id("polymarket_us:tsc-epl-bre-sun-2026-09-05-2pt5:no") == "epl"
    assert SL.league_for_market_id("polymarket_us:tsc-mls-aus-sea-2026-09-05-3pt5") == "mls"
    assert SL.league_for_market_id("polymarket_us:aec-wnba-gsv-tor-2026-07-08") is None
    assert SL.league_for_market_id("polymarket_us:junk") is None
    assert SL.league_for_market_id("kalshi:KXLALIGAGAME-26SEP05RMABAR-RMA") == "laliga"


# ---------------------------------------------------------------------------
# Tier config keyed by league
# ---------------------------------------------------------------------------


class _Mkt:
    def __init__(self, ticker="", league=None):
        self.ticker = ticker
        self.league = league


def test_sharp_weight_for_league():
    assert soccer_tiers.sharp_weight_for_league("epl") == 0.85
    assert soccer_tiers.sharp_weight_for_league("UCL") == 0.85
    assert soccer_tiers.sharp_weight_for_league("mls") == 0.40
    assert soccer_tiers.sharp_weight_for_league(None) == 0.40
    assert soccer_tiers.sharp_weight_for_league("nowhere") == 0.40


def test_polymarket_market_gets_tier_weight_by_league():
    """Regression: a PolyUS market has NO Kalshi ticker, so the ticker-only
    lookup used to hand every PolyUS EPL/UCL game the 0.40 MLS default."""
    assert soccer_tiers.sharp_weight_for_market(_Mkt(ticker="", league="epl")) == 0.85
    assert soccer_tiers.sharp_weight_for_market(_Mkt(ticker="", league="mls")) == 0.40
    # ticker fallback for markets that predate the league field
    assert soccer_tiers.sharp_weight_for_market(
        _Mkt(ticker="KXSERIEAGAME-26SEP05JUVINT-JUV")
    ) == 0.85
    assert soccer_tiers.sharp_weight_for_market(_Mkt()) == 0.40


def test_shipped_ramps_reproduce_sector_default():
    """Ship-state: top tier carries the exact sector-level soccer ramp, MLS
    carries none (inherits the sector ramp) — so today's blend is unchanged."""
    from evmax.agents.models.ensemble_agent import EnsembleModelAgent

    sector_ramp = EnsembleModelAgent.DISAGREEMENT_OVERRIDES["soccer"]
    for lg in ("epl", "laliga", "bundesliga", "seriea", "ligue1", "ucl", "uel"):
        assert soccer_tiers.disagreement_ramp_for_league(lg) == sector_ramp, lg
    assert soccer_tiers.disagreement_ramp_for_league("mls") is None
    assert soccer_tiers.disagreement_ramp_for_market(_Mkt(league="mls")) is None
    assert soccer_tiers.disagreement_ramp_for_market(
        _Mkt(ticker="KXEPLGAME-26APR24MANCHE-CHE")
    ) == sector_ramp


def test_tier_yaml_ramp_validation(tmp_path, monkeypatch):
    bad = tmp_path / "tiers.yaml"
    bad.write_text(
        "default_sharp_weight: 0.40\n"
        "tiers:\n  t:\n    sharp_weight: 0.5\n    leagues: [mls]\n"
        "    disagreement_ramp: {threshold: 0.2, saturate_at: 0.1, cap: 1.0}\n"
    )
    monkeypatch.setattr(soccer_tiers, "_CONFIG_PATH", bad)
    soccer_tiers.reset_cache()
    try:
        with pytest.raises(ValueError, match="disagreement_ramp"):
            soccer_tiers.disagreement_ramp_for_league("mls")
        # sharp weight lookup is independent of the ramp validation
        assert soccer_tiers.sharp_weight_for_league("mls") == 0.5
    finally:
        soccer_tiers.reset_cache()


def test_legacy_kalshi_series_tier_shape_still_works(tmp_path, monkeypatch):
    legacy = tmp_path / "tiers.yaml"
    legacy.write_text(
        "default_sharp_weight: 0.40\n"
        "tiers:\n  top:\n    sharp_weight: 0.9\n    kalshi_series: [KXEPLGAME]\n"
    )
    monkeypatch.setattr(soccer_tiers, "_CONFIG_PATH", legacy)
    soccer_tiers.reset_cache()
    try:
        assert soccer_tiers.sharp_weight_for_league("epl") == 0.9
        assert soccer_tiers.sharp_weight_for_ticker("KXEPLGAME-26APR24MANCHE-CHE") == 0.9
        assert soccer_tiers.sharp_weight_for_league("mls") == 0.40
    finally:
        soccer_tiers.reset_cache()


# ---------------------------------------------------------------------------
# Venue parsers stamp PredictionMarket.league
# ---------------------------------------------------------------------------


def _kalshi_raw(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "title": "x wins",
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.45",
        "no_bid_dollars": "0.50",
        "no_ask_dollars": "0.60",
        "volume_fp": 10,
        "open_interest_fp": 5,
    }


def test_kalshi_parser_stamps_league():
    from evmax.clients.kalshi import KalshiClient

    c = KalshiClient()
    m = c._parse_market(_kalshi_raw("KXMLSGAME-26JUL16MTLTOR-TOR"), "soccer")
    assert m is not None and m.league == "mls"
    m = c._parse_market(_kalshi_raw("KXUCLGAME-26SEP10BMUBOG-BOG"), "soccer")
    assert m is not None and m.league == "ucl"


def test_polymarket_parser_stamps_league():
    from tests.test_polymarket_us_client import _client, _drawable_markets, _event, _moneyline_market

    soccer = _client()._parse_event(_event(_drawable_markets()), "soccer", "ucl")
    assert soccer and all(m.league == "ucl" for m in soccer)
    wnba = _client()._parse_event(_event([_moneyline_market()]), "wnba", "wnba")
    assert wnba and all(m.league is None for m in wnba)


# ---------------------------------------------------------------------------
# EVGap → ev_predictions.league (logger + backfill)
# ---------------------------------------------------------------------------


def _gap(market_id: str, league=None, sector="soccer") -> EVGap:
    return EVGap(
        market_id=market_id,
        event_id=f"evt:{market_id}",
        sector=sector,
        yes_team="chelsea",
        market_type="moneyline",
        kalshi_yes_price=0.45,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=0.07,
        kelly_full=0.04,
        kelly_fraction=0.02,
        match_confidence=0.95,
        volume_usd=5_000.0,
        spread_pct=0.02,
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources="elo+form+poisson+xg+sharp",
        event_title="Chelsea vs Arsenal",
        league=league,
    )


@pytest.fixture
def real_db(tmp_path, monkeypatch):
    """A real predictions.db in tmp — exercises get_connection()'s migration
    (the `league` ALTER + backfill) rather than a hand-written schema."""
    from evmax.agents.cleanup import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "predictions.db")
    return db_module


def test_logger_persists_gap_league_and_derives_from_kalshi_id(real_db):
    from evmax.agents.cleanup.logger import log_gaps

    gaps = [
        _gap("kalshi:KXEPLGAME-26SEP05CHEARS-CHE", league="epl"),
        _gap("kalshi:KXMLSGAME-26SEP05LAFCSEA-SEA"),               # no league on gap → derive
        _gap("polymarket_us:atc-epl-che-ars-2026-09-05-che"),     # no league on gap → slug token
        _gap("polymarket_us:nonsense"),                            # underivable → NULL
        _gap("kalshi:KXNBAGAME-26SEP05LALGSW-LAL", sector="nba"),
    ]
    assert log_gaps(gaps, sharp_weight_used=0.85, bankroll_used=500.0) == 5
    with real_db.get_connection() as conn:
        got = dict(conn.execute("SELECT market_id, league FROM ev_predictions").fetchall())
    assert got == {
        "kalshi:KXEPLGAME-26SEP05CHEARS-CHE": "epl",
        "kalshi:KXMLSGAME-26SEP05LAFCSEA-SEA": "mls",
        "polymarket_us:atc-epl-che-ars-2026-09-05-che": "epl",
        "polymarket_us:nonsense": None,
        "kalshi:KXNBAGAME-26SEP05LALGSW-LAL": None,
    }


def test_backfill_fills_only_kalshi_soccer_rows_and_is_idempotent(real_db):
    from evmax.agents.cleanup.db import backfill_league_column

    with real_db.get_connection() as conn:
        for mid, sector in [
            ("kalshi:KXLALIGAGAME-26SEP05RMABAR-RMA", "soccer"),
            ("kalshi:KXUELGAME-26SEP05ROMLYO-ROM:no", "soccer"),
            ("polymarket_us:tsc-mls-aus-sea-2026-09-05-3pt5", "soccer"),
            ("polymarket_us:nonsense", "soccer"),
            ("kalshi:KXNFLGAME-26SEP05KCBUF-KC", "nfl"),
        ]:
            conn.execute(
                """INSERT INTO ev_predictions
                   (scan_date, market_id, event_id, sector, yes_team, market_type,
                    kalshi_yes_price, sharp_true_prob, blended_true_prob, ev_pct, kelly_fraction)
                   VALUES ('2026-09-05', ?, 'e', ?, 'x', 'moneyline', 0.5, 0.5, 0.5, 0.02, 0.01)""",
                (mid, sector),
            )
        conn.commit()
        assert backfill_league_column(conn) == 3
        # second pass: the unresolvable PolyUS row is still NULL, so the
        # pending probe fires, but nothing is updated
        assert backfill_league_column(conn) == 0
        got = dict(conn.execute("SELECT market_id, league FROM ev_predictions").fetchall())
    assert got["kalshi:KXLALIGAGAME-26SEP05RMABAR-RMA"] == "laliga"
    assert got["kalshi:KXUELGAME-26SEP05ROMLYO-ROM:no"] == "uel"
    assert got["polymarket_us:tsc-mls-aus-sea-2026-09-05-3pt5"] == "mls"   # not 'seriea'
    assert got["polymarket_us:nonsense"] is None
    assert got["kalshi:KXNFLGAME-26SEP05KCBUF-KC"] is None


def test_get_connection_backfills_pre_column_database(tmp_path, monkeypatch):
    """A DB created before the column existed gets the ALTER + backfill on open."""
    from evmax.agents.cleanup import db as db_module

    path = tmp_path / "predictions.db"
    raw = sqlite3.connect(str(path))
    raw.executescript(db_module.SCHEMA.replace(
        "    league              TEXT,                       -- league inside a multi-league sector (soccer: epl/ucl/mls/...); NULL elsewhere\n",
        "",
    ))
    assert "league" not in [r[1] for r in raw.execute("PRAGMA table_info(ev_predictions)")]
    raw.execute(
        """INSERT INTO ev_predictions
           (scan_date, market_id, event_id, sector, yes_team, market_type,
            kalshi_yes_price, sharp_true_prob, blended_true_prob, ev_pct, kelly_fraction)
           VALUES ('2026-09-05', 'kalshi:KXBUNDESLIGAGAME-26SEP05BAYDOR-BAY', 'e', 'soccer',
                   'x', 'moneyline', 0.5, 0.5, 0.5, 0.02, 0.01)"""
    )
    raw.commit(); raw.close()
    monkeypatch.setattr(db_module, "DB_PATH", path)
    with db_module.get_connection() as conn:
        assert conn.execute("SELECT league FROM ev_predictions").fetchone()[0] == "bundesliga"


# ---------------------------------------------------------------------------
# Ensemble: per-event disagreement ramp override
# ---------------------------------------------------------------------------


class TestEnsemblePerEventRamp:
    def test_params_win_over_sector_override(self):
        from evmax.agents.models.ensemble_agent import EnsembleModelAgent

        kw = dict(model_a=0.50, model_b=0.50, model_draw=None,
                  sharp_a=0.56, sharp_b=0.44, sharp_draw=None,
                  base_sharp_weight=0.40, sector="soccer")
        # 6pt gap: the soccer sector ramp (0.04/0.10/1.00) is already ramping.
        sector_w = EnsembleModelAgent._disagreement_sharp_weight(**kw)
        assert sector_w > 0.40
        # Explicit global-default triple: below its 10pt threshold → untouched.
        # (The old kwarg-default heuristic would have mistaken 0.10/0.30/0.95
        # for "unset" and applied the sector ramp instead.)
        assert EnsembleModelAgent._disagreement_sharp_weight(
            **kw, params=(0.10, 0.30, 0.95)
        ) == 0.40
        # And a harsher explicit ramp saturates fully.
        assert EnsembleModelAgent._disagreement_sharp_weight(
            **kw, params=(0.01, 0.05, 1.00)
        ) == 1.0

    @pytest.mark.asyncio
    async def test_run_threads_disagreement_by_event(self):
        from evmax.agents.base import AgentRequest
        from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
        from evmax.agents.models.ensemble_agent import EnsembleModelAgent
        from evmax.models.market import MarketSource, MarketType, PredictionMarket
        from evmax.models.odds import SharpBook, SharpOdds

        class _Stub(ModelAgent):
            name = "elo"
            weight = 1.0

            def load_state(self) -> None:
                self._state = {}

            def save_state(self) -> None:
                pass

            async def update(self, *args, **kwargs):
                return None

            async def predict_pair(self, market, sharp_odds):
                return ModelAgentPrediction(
                    event_id=sharp_odds.event_id, model_name=self.name,
                    true_prob_a=0.50, true_prob_b=0.50, true_prob_draw=None,
                    confidence=0.9, weight=self.weight, sample_size=17,
                )

        def _pair(ev):
            sharp = SharpOdds(
                event_id=ev, book=SharpBook.pinnacle, sector="soccer",
                market_type=MarketType.moneyline,
                outcome_a_label="team a", outcome_b_label="team b",
                outcome_a_decimal=1 / 0.56, outcome_b_decimal=1 / 0.44,
                true_prob_a=0.56, true_prob_b=0.44, margin=0.03,
            )
            market = PredictionMarket(
                id=f"kalshi:X-{ev}", source=MarketSource.kalshi, sector="soccer",
                market_type=MarketType.moneyline, title="A vs B",
                yes_price=0.5, no_price=0.5, team_home="team a", team_away="team b",
                event_id=ev,
            )
            return {"market": market, "sharp": sharp}

        ens = EnsembleModelAgent(models=[_Stub()], sharp_weight=0.40)
        resp = await ens(AgentRequest(
            sector="soccer",
            params={
                "pairs": [_pair("e1"), _pair("e2")],
                "sharp_weight": 0.40,
                "disagreement_by_event": {"e2": (0.10, 0.30, 0.95)},
            },
        ))
        b1, b2 = resp.data["e1"], resp.data["e2"]
        # e1: sector soccer ramp on a 6pt gap → pulled toward sharp.
        assert b1.effective_sharp_weight > 0.40
        # e2: explicit global-default ramp → 6pt is below threshold → base weight.
        assert b2.effective_sharp_weight == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Shadow CLI + promotion board league filters
# ---------------------------------------------------------------------------


def _seed_clv_db(db_module, rows):
    """rows: (market_id, league, clv, outcome, venue)."""
    with db_module.get_connection() as conn:
        for mid, league, clv, outcome, venue in rows:
            conn.execute(
                """INSERT INTO ev_predictions
                   (scan_date, market_id, event_id, event_title, sector, yes_team, market_type,
                    kalshi_yes_price, sharp_true_prob, blended_true_prob, ev_pct, kelly_fraction,
                    model_sources, mode, kalshi_clv_pct, venue, league, event_date)
                   VALUES ('2026-09-01', ?, ?, 'A vs B', 'soccer', 'a', 'moneyline',
                           0.5, 0.55, 0.56, 0.03, 0.01, 'elo+form+poisson+xg+sharp', 'live',
                           ?, ?, ?, '2026-09-02')""",
                (mid, f"evt:{mid}", clv, venue, league),
            )
            conn.execute(
                "INSERT INTO ev_outcomes (market_id, event_id, event_date, sector, yes_team, "
                "outcome) VALUES (?, ?, '2026-09-02', 'soccer', 'a', ?)",
                (mid, f"evt:{mid}", outcome),
            )
        conn.commit()


def test_clv_stats_league_filter(real_db):
    from evmax.cli.commands.shadow import clv_stats

    rows = [(f"kalshi:KXEPLGAME-{i}", "epl", 2.0, 1, "kalshi") for i in range(5)]
    rows += [(f"kalshi:KXMLSGAME-{i}", "mls", -3.0, 0, "kalshi") for i in range(4)]
    rows += [("polymarket_us:nonsense", None, 9.0, 1, "polymarket_us")]
    _seed_clv_db(real_db, rows)
    assert clv_stats("soccer")["n"] == 10
    epl = clv_stats("soccer", league="EPL")
    assert epl["n"] == 5 and epl["mean_clv_pp"] == pytest.approx(2.0)
    mls = clv_stats("soccer", league="mls")
    assert mls["n"] == 4 and mls["mean_clv_pp"] == pytest.approx(-3.0)


def test_clv_leagues_command_segments_and_buckets_unknown(real_db):
    from evmax.cli.commands.shadow import app

    rows = [(f"kalshi:KXEPLGAME-{i}", "epl", 2.0, 1, "kalshi") for i in range(3)]
    rows += [(f"kalshi:KXMLSGAME-{i}", "mls", -3.0, 0, "kalshi") for i in range(2)]
    rows += [("polymarket_us:nonsense", None, 9.0, 1, "polymarket_us")]
    _seed_clv_db(real_db, rows)
    res = CliRunner().invoke(app, ["clv-leagues", "soccer"])
    assert res.exit_code == 0, res.stdout
    out = res.stdout
    assert "Premier League" in out and "+2.00pp" in out
    assert "MLS" in out and "-3.00pp" in out
    assert "unknown" in out and "+9.00pp" in out
    # non-soccer refused
    assert CliRunner().invoke(app, ["clv-leagues", "nba"]).exit_code == 1


def test_promotion_board_league_filter(real_db):
    from evmax.agents.cleanup.promotion_board import compute_promotion_board

    rows = [(f"kalshi:KXEPLGAME-{i}", "epl", 2.0, 1, "kalshi") for i in range(3)]
    rows += [(f"kalshi:KXMLSGAME-{i}", "mls", -3.0, 0, "kalshi") for i in range(2)]
    _seed_clv_db(real_db, rows)
    pooled = compute_promotion_board(days=30, staleness_h=None, sector="soccer")
    mls = compute_promotion_board(days=30, staleness_h=None, sector="soccer", league="mls")
    assert len(pooled) == 1 and pooled[0]["n_logged"] == 5
    assert len(mls) == 1 and mls[0]["n_logged"] == 2
    # the CLV gate is computed on the league's rows only
    assert mls[0]["gates"]["clv_mean"]["value"] == pytest.approx(-3.0)
    assert pooled[0]["gates"]["clv_mean"]["value"] == pytest.approx(0.0)
