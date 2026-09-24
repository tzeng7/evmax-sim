"""Every remaining model agent resolves a team label through the shared
unique-match rule (``evmax/agents/models/_team_lookup.resolve_team_key``).

Before this change each of these agents carried a private first-match fallback
(``startswith`` / ``endswith`` / substring with no word boundary) and returned
the FIRST store entry that matched. Every ``OLD:`` note below is what the
pre-unification code returned for that exact input — each test fails on it.
The ``still_resolves`` cases are labels the old code got right (archived
Pinnacle spellings among them) and must keep getting right.
"""

from __future__ import annotations

from datetime import date

import pytest

from evmax.agents.models import nfl_qb_elo_agent
from evmax.agents.models._college_efficiency import college_season_start_year, resolve_team
from evmax.agents.models.efficiency_agent import EfficiencyModelAgent
from evmax.agents.models.matchup_agent import MatchupAgent
from evmax.agents.models.nfl_efficiency_agent import NflEfficiencyModelAgent
from evmax.agents.models.nhl_xg_agent import NhlXgModelAgent
from evmax.agents.models.possession_sim_agent import PossessionSimAgent
from evmax.agents.models.shot_quality_agent import ShotQualityAgent
from evmax.agents.models.wnba_efficiency_agent import WNBAEfficiencyModelAgent
from evmax.agents.models.wnba_possession_sim_agent import WNBAPossessionSimAgent
from evmax.models_ml.point_projection import PointProjectionModel


def _key_of(store: dict, val):
    if val is None:
        return None
    return next(k for k, v in store.items() if v is val)


# ---------------------------------------------------------------------------
# NBA — efficiency, possession_sim, shot_quality, matchup (nickname-keyed
# stores, same shape as data/models/{efficiency,shot_quality,matchup}_state)
# ---------------------------------------------------------------------------

def _nba_teams() -> dict:
    # Dict order matters for the old first-match loops: "thunder" and
    # "warriors" come first, exactly as a substring scan would meet them.
    return {
        "thunder": {"full_name": "oklahoma city thunder", "gp": 82},
        "warriors": {"full_name": "golden state warriors", "gp": 82},
        "nuggets": {"full_name": "denver nuggets", "gp": 82},
        "pelicans": {"full_name": "new orleans pelicans", "gp": 82},
        "magic": {"full_name": "orlando magic", "gp": 82},
        "lakers": {"full_name": "los angeles lakers", "gp": 82},
        "la clippers": {"full_name": "la clippers", "gp": 82},
        "trail blazers": {"full_name": "portland trail blazers", "gp": 82},
    }


def _eff(teams, label):
    return _key_of(teams, EfficiencyModelAgent()._resolve_team(teams, label))


def _sim(teams, label):
    return _key_of(teams, PossessionSimAgent()._resolve_team(teams, label))


def _shot(teams, label):
    agent = ShotQualityAgent()
    agent._team_shooting = teams
    return _key_of(teams, agent._resolve_team(label))


def _matchup(teams, label):
    for t in teams.values():
        t.pop("full_name", None)  # matchup_state stores carry no full_name
    return _key_of(teams, MatchupAgent()._resolve(label, teams))


NBA_RESOLVERS = pytest.mark.parametrize(
    "resolve", [_eff, _sim, _shot, _matchup], ids=["efficiency", "possession_sim", "shot_quality", "matchup"],
)


class TestNbaAgents:
    @NBA_RESOLVERS
    def test_totals_side_label_is_not_a_team(self, resolve):
        # OLD: "oklahoma city thunder".endswith("under") → Thunder for efficiency/
        # possession_sim/shot_quality; matchup's "thunder".endswith("under") too.
        assert resolve(_nba_teams(), "under") is None

    @NBA_RESOLVERS
    def test_los_angeles_clippers_reaches_la_clippers_key(self, resolve):
        # Archived Pinnacle label. OLD: None in every NBA agent — the stored
        # key is "la clippers" and no fallback bridged "los angeles" ↔ "la".
        assert resolve(_nba_teams(), "los angeles clippers") == "la clippers"

    @NBA_RESOLVERS
    def test_ticker_codes_resolve_to_their_own_team(self, resolve):
        # OLD (full_name substring): "den" in "golden state warriors" → Warriors;
        # "orl" in "new orleans pelicans" → Pelicans. matchup had no full_name
        # scan and returned None for both.
        assert resolve(_nba_teams(), "den") == "nuggets"
        assert resolve(_nba_teams(), "orl") == "magic"

    @NBA_RESOLVERS
    @pytest.mark.parametrize("label,key", [
        ("los angeles lakers", "lakers"),
        ("lakers", "lakers"),
        ("portland trail blazers", "trail blazers"),
        ("oklahoma city thunder", "thunder"),
    ])
    def test_still_resolves(self, resolve, label, key):
        assert resolve(_nba_teams(), label) == key


# ---------------------------------------------------------------------------
# WNBA — efficiency + possession_sim (both staticmethod _resolve_team)
# ---------------------------------------------------------------------------

def _wnba_teams() -> dict:
    return {
        "dream": {"full_name": "atlanta dream"},
        "sparks": {"full_name": "los angeles sparks"},
        "wings": {"full_name": "dallas wings"},
        "valkyries": {"full_name": "golden state valkyries"},
        "aces": {"full_name": "las vegas aces"},
    }


WNBA_RESOLVERS = pytest.mark.parametrize(
    "agent_cls", [WNBAEfficiencyModelAgent, WNBAPossessionSimAgent],
    ids=["wnba_efficiency", "wnba_possession_sim"],
)


class TestWnbaAgents:
    @WNBA_RESOLVERS
    def test_codes_do_not_substring_match_another_team(self, agent_cls):
        teams = _wnba_teams()
        # OLD: "la" in "atlanta dream" → Dream; "gs" in "dallas wings" → Wings.
        assert _key_of(teams, agent_cls._resolve_team(teams, "la")) == "sparks"
        assert _key_of(teams, agent_cls._resolve_team(teams, "gs")) == "valkyries"

    @WNBA_RESOLVERS
    def test_still_resolves(self, agent_cls):
        teams = _wnba_teams()
        assert _key_of(teams, agent_cls._resolve_team(teams, "las vegas aces")) == "aces"
        assert _key_of(teams, agent_cls._resolve_team(teams, "aces")) == "aces"
        # All-Star exhibition sides stay unresolved.
        assert agent_cls._resolve_team(teams, "team cooper") is None


# ---------------------------------------------------------------------------
# NHL — nhl_xg (full-name keys, MoneyPuck abbreviations)
# ---------------------------------------------------------------------------

def _nhl_teams() -> dict:
    return {
        "new york islanders": {"abbrev": "NYI"},
        "new york rangers": {"abbrev": "NYR"},
        "los angeles kings": {"abbrev": "LAK"},
        "st louis blues": {"abbrev": "STL"},
    }


class TestNhlXg:
    def test_la_is_the_kings_not_a_substring_hit(self):
        teams = _nhl_teams()
        # OLD: substring fallback — "la" in "new york islanders" → Islanders.
        assert _key_of(teams, NhlXgModelAgent()._resolve_team(teams, "la")) == "los angeles kings"

    @pytest.mark.parametrize("label,key", [
        ("st. louis blues", "st louis blues"),   # archived Pinnacle spelling
        ("new york rangers", "new york rangers"),
        ("ny rangers", "new york rangers"),      # nickname dictionary
        ("l.a", "los angeles kings"),            # MoneyPuck dotted abbrev
        ("lak", "los angeles kings"),
    ])
    def test_still_resolves(self, label, key):
        teams = _nhl_teams()
        assert _key_of(teams, NhlXgModelAgent()._resolve_team(teams, label)) == key

    def test_legacy_dotted_key_still_found(self):
        teams = {"st. louis blues": {"abbrev": "STL"}}
        got = NhlXgModelAgent()._resolve_team(teams, "st louis blues")
        assert got is teams["st. louis blues"]

    def test_non_team_labels_unresolved(self):
        assert NhlXgModelAgent()._resolve_team(_nhl_teams(), "home goals (2 games)") is None


# ---------------------------------------------------------------------------
# NFL — nfl_efficiency + nfl_qb_elo (one shared resolve_nfl_team_key)
# ---------------------------------------------------------------------------

def _nfl_teams() -> dict:
    return {
        "new york jets": {"abbrev": "NYJ"},
        "new york giants": {"abbrev": "NYG"},
        "los angeles rams": {"abbrev": "LA"},
        "los angeles chargers": {"abbrev": "LAC"},
        "kansas city chiefs": {"abbrev": "KC"},
    }


def _nfl_eff(teams, label):
    return _key_of(teams, NflEfficiencyModelAgent()._resolve_team(teams, label))


def _nfl_qb(teams, label):
    return nfl_qb_elo_agent._resolve_team(teams, label)


NFL_RESOLVERS = pytest.mark.parametrize("resolve", [_nfl_eff, _nfl_qb], ids=["nfl_efficiency", "nfl_qb_elo"])


class TestNflAgents:
    @NFL_RESOLVERS
    def test_ambiguous_city_is_refused(self, resolve):
        # OLD: substring fallback returned the first key containing the label —
        # "new york" → Jets, "los angeles" → Rams.
        assert resolve(_nfl_teams(), "new york") is None
        assert resolve(_nfl_teams(), "los angeles") is None

    @NFL_RESOLVERS
    def test_alias_code_resolves(self, resolve):
        # OLD: None ("LAR" is not in NFL_ABBREV_TO_NAME, which uses "LA").
        assert resolve(_nfl_teams(), "lar") == "los angeles rams"

    @NFL_RESOLVERS
    @pytest.mark.parametrize("label,key", [
        ("kansas city chiefs", "kansas city chiefs"),  # archived Pinnacle spelling
        ("chiefs", "kansas city chiefs"),              # Kalshi yes_team spelling
        ("kc", "kansas city chiefs"),
        ("la", "los angeles rams"),                    # NFL_ABBREV_TO_NAME
        ("ny giants", "new york giants"),              # nickname dictionary
    ])
    def test_still_resolves(self, resolve, label, key):
        assert resolve(_nfl_teams(), label) == key


# ---------------------------------------------------------------------------
# NCAAB / NCAAW — _college_efficiency.resolve_team + all four college agents
# ---------------------------------------------------------------------------

def _college_teams() -> dict:
    return {
        "miami": {"ortg": 118.0, "drtg": 92.0, "pace": 68.0, "gp": 25, "tov_pct": 0.15,
                  "full_name": "miami hurricanes"},
        "miami oh": {"ortg": 95.0, "drtg": 110.0, "pace": 66.0, "gp": 25, "tov_pct": 0.20,
                     "full_name": "miami (oh) redhawks"},
        "mississippi valley state": {"ortg": 90.0, "drtg": 112.0, "pace": 67.0, "gp": 25, "tov_pct": 0.21,
                                     "full_name": "mississippi valley state delta devils"},
        "southern": {"ortg": 96.0, "drtg": 106.0, "pace": 67.0, "gp": 25, "tov_pct": 0.19,
                     "full_name": "southern jaguars"},
        "nicholls": {"ortg": 100.0, "drtg": 103.0, "pace": 67.0, "gp": 25, "tov_pct": 0.18,
                     "full_name": "nicholls colonels"},
    }


@pytest.mark.parametrize("sector", ["ncaab", "ncaaw"])
class TestCollegeResolveTeam:
    def test_miami_ohio_is_not_the_hurricanes(self, sector):
        teams = _college_teams()
        # OLD: "miami ohio".startswith("miami ") → the Miami (FL) Hurricanes.
        # Archived Pinnacle label (NCAAB tournament, March 2026).
        assert _key_of(teams, resolve_team(teams, "Miami Ohio", sector)) == "miami oh"
        assert _key_of(teams, resolve_team(teams, "Miami Florida", sector)) == "miami"

    def test_other_schools_do_not_borrow_a_prefix(self, sector):
        teams = _college_teams()
        # OLD: full_name "mississippi valley state delta devils".startswith("mississippi ")
        # → MVSU for Ole Miss; "southern california".startswith("southern ") → Southern U.
        assert resolve_team(teams, "Mississippi", sector) is None
        assert resolve_team(teams, "Southern California", sector) is None

    @pytest.mark.parametrize("label,key", [
        ("McNeese State", "mcneese"),
        ("Nicholls State", "nicholls"),
        ("Grambling State", "grambling"),
        ("Sam Houston State", "sam houston"),
        ("Middle Tennessee State", "middle tennessee"),
        ("Central Connecticut State", "central connecticut"),
    ])
    def test_state_spelling_of_an_espn_location_key(self, sector, label, key):
        # ESPN's location drops "State"; the ncaab/ncaaw.yaml alias maps it back
        # (the shared rule never drops "state" on its own — Alabama ≠ Alabama
        # State). Each one resolved before only through the unguarded prefix.
        teams = {key: {"full_name": key}, "tennessee": {"full_name": "tennessee volunteers"}}
        assert _key_of(teams, resolve_team(teams, label, sector)) == key


def _college_state(teams: dict) -> dict:
    return {
        "source_season": str(college_season_start_year(date.today())),
        "league_avg_ortg": 104.0, "league_avg_drtg": 104.0, "league_avg_pace": 68.0,
        "league_avg_tov_pct": 0.17, "hca_eff": 4.0, "fetched_at": date.today().isoformat(),
        "teams": teams,
    }


def _college_pair(sector, home, away):
    from evmax.models.market import MarketSource, MarketType, PredictionMarket
    from evmax.models.odds import SharpBook, SharpOdds

    event_id = f"{sector}::2026-01-15::{home}_vs_{away}"
    market = PredictionMarket(
        id="t1", event_id=event_id, ticker="TEST", title=f"{home} vs {away}",
        yes_price=0.5, no_price=0.5, sector=sector, source=MarketSource.kalshi,
        market_type=MarketType.moneyline, team_home=home, team_away=away, yes_team=home,
    )
    sharp = SharpOdds(
        event_id=event_id, book=SharpBook.pinnacle, sector=sector, team_a=home, team_b=away,
        true_prob_a=0.5, true_prob_b=0.5, outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        outcome_a_label=home, outcome_b_label=away,
    )
    return market, sharp


def _college_agent(kind: str):
    if kind == "ncaab_efficiency":
        from evmax.agents.models.ncaab_efficiency_agent import NcaabEfficiencyModelAgent as cls
    elif kind == "ncaaw_efficiency":
        from evmax.agents.models.ncaaw_efficiency_agent import NcaawEfficiencyModelAgent as cls
    elif kind == "ncaab_possession_sim":
        from evmax.agents.models.ncaab_possession_sim_agent import NcaabPossessionSimAgent as cls
    else:
        from evmax.agents.models.ncaaw_possession_sim_agent import NcaawPossessionSimAgent as cls
    agent = cls()
    state = _college_state(_college_teams())
    if kind.endswith("possession_sim"):
        agent._efficiency_data = state
    else:
        agent._state = state
    return agent


class TestCollegeAgents:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", [
        "ncaab_efficiency", "ncaaw_efficiency", "ncaab_possession_sim", "ncaaw_possession_sim",
    ])
    async def test_miami_ohio_priced_with_its_own_rating(self, kind):
        # Pinnacle labels as archived. OLD: both sides resolved to "miami", so the
        # game priced the Hurricanes against themselves (≈ home-edge coin flip).
        sector = kind.split("_", 1)[0]
        agent = _college_agent(kind)
        market, sharp = _college_pair(sector, "Miami Ohio", "Miami Florida")
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a < 0.30   # weak RedHawks vs strong Hurricanes


# ---------------------------------------------------------------------------
# point_projection — _resolve_key via the shared rule; CITY_TO_TEAM scoped
# ---------------------------------------------------------------------------

@pytest.fixture
def projection(monkeypatch):
    def _init(self):
        self._poisson_state = {}
        self._elo_state = {
            "ncaab": {"ratings": {"cal state fullerton titans": 1400.0, "tennessee": 1700.0}},
            "nhl": {"ratings": {"florida panthers": 1550.0, "carolina hurricanes": 1600.0}},
            "nba": {"ratings": {"lakers": 1600.0, "la clippers": 1550.0}},
            "nfl": {"ratings": {"titans": 1450.0}},
        }
        self._efficiency_state = {}
        self._possession_agent = None

    monkeypatch.setattr(PointProjectionModel, "_load_state", _init)
    return PointProjectionModel()


def _elo_for(model: PointProjectionModel, sector: str, label: str):
    # The exact chain project() runs: _normalize_team → _get_elo.
    return model._get_elo(sector, model._normalize_team(sector, label))


class TestPointProjection:
    def test_college_city_is_not_an_nfl_nickname(self, projection):
        # OLD: CITY_TO_TEAM "tennessee" → "titans", then "…titans".endswith → Cal
        # State Fullerton's 1400.
        assert _elo_for(projection, "ncaab", "Tennessee") == 1700.0

    def test_nhl_carolina_is_the_hurricanes(self, projection):
        # OLD: CITY_TO_TEAM "carolina" → "panthers" → Florida Panthers.
        assert _elo_for(projection, "nhl", "Carolina") == 1600.0

    def test_nba_and_nfl_city_map_still_applies(self, projection):
        assert _elo_for(projection, "nba", "Los Angeles Lakers") == 1600.0
        assert _elo_for(projection, "nba", "Los Angeles Clippers") == 1550.0
        assert _elo_for(projection, "nfl", "Tennessee") == 1450.0

    def test_detect_sector_uses_the_sector_rule(self, projection):
        assert projection.detect_sector("Carolina Hurricanes") == "nhl"
