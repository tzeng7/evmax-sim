"""Five league-shadowed soccer leagues (Liga MX / J League / Eredivisie /
Brasileirão / Championship) + the league-level shadow mechanism.

Mode is per SECTOR, so a league wired into the live `soccer` sector would be
live on its first scan. `data/soccer_league_tiers.yaml` `shadow_leagues`
holds it back at the same three sites the Polymarket US venue firewall uses,
and `cleanup shadow promote-league` lifts it on per-league CLV.
"""
from __future__ import annotations

import dataclasses
from datetime import date, datetime

import pytest
from typer.testing import CliRunner

from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.sectors import soccer_leagues as SL
from evmax.sectors import soccer_tiers

NEW = ("ligamx", "jleague", "eredivisie", "brasileirao", "championship")


# ---------------------------------------------------------------------------
# Identifiers wired everywhere
# ---------------------------------------------------------------------------


def test_new_leagues_in_every_map():
    from evmax.clients.esports_pinnacle import SECTOR_SPORT_LEAGUES, SOCCER_DRAW_LEAGUES
    from evmax.clients.kalshi import SECTOR_SERIES_MAP
    from evmax.agents.cleanup.resolver import ESPN_SOCCER_LEAGUES

    for lg in NEW:
        assert lg in SL.SOCCER_LEAGUES and lg in SL.LEAGUE_DISPLAY
    series = {s for s, lg in SL.KALSHI_SERIES_LEAGUE.items() if lg in NEW}
    assert series == {
        "KXLIGAMXGAME", "KXJLEAGUEGAME", "KXEREDIVISIEGAME",
        "KXBRASILEIROGAME", "KXEFLCHAMPIONSHIPGAME",
    }
    assert series <= set(SECTOR_SERIES_MAP["soccer"])
    for pid in (2242, 2157, 1928, 1834, 1977):
        assert pid in SECTOR_SPORT_LEAGUES["soccer"][1]
        assert pid in SOCCER_DRAW_LEAGUES  # every soccer league prices the draw
    for slug in ("mex.1", "jpn.1", "ned.1", "bra.1", "eng.2"):
        assert slug in ESPN_SOCCER_LEAGUES
    for slug, lg in (("lmx", "ligamx"), ("bra", "brasileirao"), ("eflch", "championship")):
        assert SL.league_for_polymarket_slug(slug) == lg


def test_new_leagues_start_top_tier_and_shadowed():
    for lg in NEW:
        assert soccer_tiers.sharp_weight_for_league(lg) == 0.85, lg
        assert lg in soccer_tiers.shadow_leagues(), lg
        assert soccer_tiers.league_is_live(lg) is False
    for lg in ("epl", "mls", "ucl"):
        assert soccer_tiers.league_is_live(lg) is True
    assert soccer_tiers.league_is_live(None) is True  # never holds back league-less gaps


def test_seed_scripts_cover_new_leagues():
    import importlib.util, pathlib
    for path, needle in (
        ("scripts/seed_espn.py", '("soccer", "mex.1")'),
        ("scripts/seed_soccer_xg.py", '("soccer", "bra.1")'),
    ):
        assert needle in pathlib.Path(path).read_text(), path


# ---------------------------------------------------------------------------
# Spellings: Pinnacle / ESPN / Kalshi agree under the soccer normalizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pinnacle, espn, kalshi",
    [
        ("Club America", "América", "America"),
        ("Club Leon", "León", "Leon"),
        ("Deportivo Toluca", "Toluca", "Toluca"),
        ("Tokyo Verdy", "Tokyo Verdy 1969", "Tokyo V"),
        ("JEF United Chiba", "JEF United Ichihara-Chiba", "United Chiba"),
        ("Urawa Red Diamonds", "Urawa Red Diamonds", "Urawa"),
        ("Fagiano Okayama", "Fagiano Okayama", "Fagiano O"),
        ("PSV", "PSV Eindhoven", "Eindhoven"),
        ("Sparta Rotterdam", "Sparta Rotterdam", "Sparta"),
        ("PEC Zwolle", "PEC Zwolle", "Zwolle"),
        ("Athletico Paranaense", "Athletico-PR", "Paranaense"),
        ("Atletico Mineiro", "Atlético-MG", "Atletico Mineiro"),
        ("Red Bull Bragantino", "Red Bull Bragantino", "Bragantino"),
        ("Sao Paulo", "São Paulo", "Sao Paulo"),
        ("Wolverhampton", "Wolverhampton Wanderers", "Wolverhampton"),
        ("Stoke City", "Stoke City", "Stoke"),
        ("Sheffield United", "Sheffield United", "Sheffield United"),
    ],
)
def test_spellings_agree(pinnacle, espn, kalshi):
    from evmax.sectors.registry import get_handler
    n = get_handler("soccer").normalize_team
    assert n(pinnacle) == n(espn) == n(kalshi), (n(pinnacle), n(espn), n(kalshi))


def test_fc_tokyo_seed_and_predict_normalize_to_the_same_key():
    """NameNormalizer (seed + resolve) strips 'FC' from 'FC Tokyo' -> 'tokyo'
    while the alias-only handler leaves 'fc tokyo'; without the explicit alias
    the seed stores 'tokyo' and predict looks up 'fc tokyo' -> MISSING. The
    alias makes both agree, and must not collide with Tokyo Verdy."""
    from evmax.matching.normalizer import NameNormalizer
    from evmax.sectors.registry import get_handler
    nn = NameNormalizer("soccer").normalize
    h = get_handler("soccer").normalize_team
    assert nn("FC Tokyo") == h("FC Tokyo") == h("tokyo") == "tokyo"
    assert h("Tokyo Verdy") == "tokyo verdy" != "tokyo"


def test_liga_mx_santos_does_not_merge_with_brazil_santos():
    """ESPN calls Santos Laguna plain 'Santos' — the override renames it at the
    ESPN extraction sites so seeds/resolve never merge the two clubs."""
    from evmax.sectors.registry import get_handler
    n = get_handler("soccer").normalize_team
    assert SL.espn_display_name("mex.1", "Santos") == "Santos Laguna"
    assert SL.espn_display_name("bra.1", "Santos") == "Santos"
    assert SL.espn_display_name(None, "Santos") == "Santos"
    assert n(SL.espn_display_name("mex.1", "Santos")) != n(SL.espn_display_name("bra.1", "Santos"))
    assert n("Santos Laguna") == n(SL.espn_display_name("mex.1", "Santos"))


def test_kalshi_parser_stamps_new_leagues():
    from evmax.clients.kalshi import KalshiClient
    c = KalshiClient()
    for ticker, lg in (
        ("KXLIGAMXGAME-26SEP19AMECDG-AME", "ligamx"),
        ("KXJLEAGUEGAME-26SEP13URDFAG-URD", "jleague"),
        ("KXEREDIVISIEGAME-26SEP13PSVSPA-TIE", "eredivisie"),
        ("KXBRASILEIROGAME-26SEP20FLARBB-FLA", "brasileirao"),
        ("KXEFLCHAMPIONSHIPGAME-26SEP13SHUWOL-WOL", "championship"),
    ):
        raw = {"ticker": ticker, "title": "x wins", "yes_bid_dollars": "0.40",
               "yes_ask_dollars": "0.45", "no_bid_dollars": "0.50", "no_ask_dollars": "0.60",
               "volume_fp": 10, "open_interest_fp": 5}
        m = c._parse_market(raw, "soccer")
        assert m is not None and m.league == lg, ticker


# ---------------------------------------------------------------------------
# League shadow: three enforcement sites + promotion
# ---------------------------------------------------------------------------


def _gap(market_id: str, league=None, venue="kalshi") -> EVGap:
    return EVGap(
        market_id=market_id, event_id=f"evt:{market_id}", sector="soccer",
        yes_team="x", market_type="moneyline", kalshi_yes_price=0.45,
        sharp_true_prob=0.55, blended_true_prob=0.56, ev_pct=0.05, kelly_full=0.04,
        kelly_fraction=0.02, match_confidence=0.95, volume_usd=5_000.0, spread_pct=0.02,
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources="elo+form+poisson+xg+sharp", event_title="A vs B",
        league=league, venue=venue,
    )


@pytest.fixture
def tiers(tmp_path, monkeypatch):
    """Isolated tier YAML: epl live, ligamx shadowed."""
    path = tmp_path / "tiers.yaml"
    path.write_text(
        "default_sharp_weight: 0.40\n"
        "# header comment\n"
        "shadow_leagues:\n"
        "  - ligamx   # trailing comment\n"
        "  - jleague\n"
        "tiers:\n  top_tier:\n    sharp_weight: 0.85\n    leagues: [epl, ligamx, jleague]\n"
    )
    monkeypatch.setattr(soccer_tiers, "_CONFIG_PATH", path)
    soccer_tiers.reset_cache()
    yield path
    soccer_tiers.reset_cache()


@pytest.fixture
def real_db(tmp_path, monkeypatch):
    from evmax.agents.cleanup import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "predictions.db")
    return db_module


def test_logger_demotes_shadow_league_rows(tiers, real_db, monkeypatch):
    from evmax.agents.cleanup import logger as logger_module
    from evmax.agents.cleanup.logger import log_gaps
    # soccer's live/shadow base comes from the registry; pin it live here.
    monkeypatch.setattr(logger_module, "get_mode", lambda *a, **k: "live", raising=False)
    gaps = [
        _gap("kalshi:KXEPLGAME-1", league="epl"),
        _gap("kalshi:KXLIGAMXGAME-1", league="ligamx"),
        _gap("kalshi:KXJLEAGUEGAME-1"),                 # league derived from the id
    ]
    assert log_gaps(gaps, sharp_weight_used=0.85, bankroll_used=500.0) == 3
    with real_db.get_connection() as conn:
        got = dict(conn.execute("SELECT market_id, mode FROM ev_predictions").fetchall())
    assert got["kalshi:KXEPLGAME-1"] == "live"
    assert got["kalshi:KXLIGAMXGAME-1"] == "shadow"
    assert got["kalshi:KXJLEAGUEGAME-1"] == "shadow"


def test_playlist_badge_mirrors_league_shadow(tiers, monkeypatch):
    import evmax.modes
    from evmax.web import playlist
    monkeypatch.setattr(evmax.modes, "get_mode", lambda *a, **k: "live")
    live = playlist.gap_to_dict(_gap("kalshi:KXEPLGAME-1", league="epl"), 500.0)
    shadow = playlist.gap_to_dict(_gap("kalshi:KXLIGAMXGAME-1", league="ligamx"), 500.0)
    assert live["mode"] == "live"
    assert shadow["mode"] == "shadow"


def test_coordinator_zeroes_kelly_and_reattaches(tiers):
    """The coordinator's league split: shadow-league gaps leave the live list
    with kelly 0 and are re-attached after sizing. Exercised through the same
    list comprehension shape the run_cycle block uses."""
    gaps = [_gap("kalshi:KXEPLGAME-1", league="epl"), _gap("kalshi:KXLIGAMXGAME-1", league="ligamx")]
    shadow = [dataclasses.replace(g, kelly_fraction=0.0) for g in gaps
              if not soccer_tiers.league_is_live(g.league)]
    live = [g for g in gaps if soccer_tiers.league_is_live(g.league)]
    assert [g.league for g in live] == ["epl"]
    assert [(g.league, g.kelly_fraction) for g in shadow] == [("ligamx", 0.0)]


def test_remove_shadow_league_is_line_based_and_keeps_comments(tiers):
    text = tiers.read_text()
    new, removed = soccer_tiers.remove_shadow_league(text, "ligamx")
    assert removed
    assert "# header comment" in new and "  - jleague\n" in new and "ligamx   #" not in new
    assert "leagues: [epl, ligamx, jleague]" in new  # the tier membership line is untouched
    again, removed2 = soccer_tiers.remove_shadow_league(new, "ligamx")
    assert not removed2 and again == new


def test_promote_league_cli_gates_on_clv(tiers, monkeypatch):
    from evmax.cli.commands import shadow as shadow_cli
    runner = CliRunner()
    monkeypatch.setattr(
        shadow_cli, "clv_stats",
        lambda *a, **k: {"n": 5, "mean_clv_pp": 1.0, "frac_positive": 0.8, "clears": False},
    )
    res = runner.invoke(shadow_cli.app, ["promote-league", "ligamx", "--yes"])
    assert res.exit_code == 1 and "does NOT clear" in res.stdout
    assert not soccer_tiers.league_is_live("ligamx")

    monkeypatch.setattr(
        shadow_cli, "clv_stats",
        lambda *a, **k: {"n": 40, "mean_clv_pp": 1.0, "frac_positive": 0.6, "clears": True},
    )
    res = runner.invoke(shadow_cli.app, ["promote-league", "ligamx", "--yes"])
    assert res.exit_code == 0, res.stdout
    assert soccer_tiers.league_is_live("ligamx")
    assert not soccer_tiers.league_is_live("jleague")
    assert "- ligamx" not in tiers.read_text()

    res = runner.invoke(shadow_cli.app, ["promote-league", "nowhere", "--yes"])
    assert res.exit_code == 1
