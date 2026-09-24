"""Tests for PlayoffAgent — playoff series detection and probability adjustments."""

from __future__ import annotations

import pytest

from evmax.agents.intelligence.playoff_agent import (
    PlayoffAgent,
    PlayoffSeries,
    _infer_round_number,
    _parse_series_summary,
)


def _make_series(
    team_a: str = "boston celtics",
    team_b: str = "indiana pacers",
    wins_a: int = 2,
    wins_b: int = 1,
    round_num: int = 1,
    round_name: str = "First Round",
    sector: str = "nba",
) -> PlayoffSeries:
    return PlayoffSeries(
        team_a=team_a,
        team_b=team_b,
        team_a_abbrev="BOS",
        team_b_abbrev="IND",
        series_wins_a=wins_a,
        series_wins_b=wins_b,
        round_num=round_num,
        round_name=round_name,
        sector=sector,
    )


class TestPlayoffSeriesProperties:
    def test_game_number(self):
        s = _make_series(wins_a=2, wins_b=1)
        assert s.game_number == 4

    def test_is_elimination_for_b(self):
        s = _make_series(wins_a=3, wins_b=1)
        assert s.is_closeout_for_a is True
        # At 3-1, team B IS facing elimination (one more loss = done)
        assert s.is_elimination_for_b is True
        assert s.is_elimination_for_a is False

    def test_is_elimination_trailing_3_2(self):
        s = _make_series(wins_a=3, wins_b=2)
        assert s.is_elimination_for_b is True
        assert s.is_closeout_for_a is True

    def test_game_7(self):
        s = _make_series(wins_a=3, wins_b=3)
        assert s.is_game_7 is True
        assert s.is_elimination_for_a is False
        assert s.is_elimination_for_b is False

    def test_not_elimination_early_series(self):
        s = _make_series(wins_a=1, wins_b=0)
        assert s.is_elimination_for_a is False
        assert s.is_elimination_for_b is False
        assert s.is_game_7 is False

    def test_is_finals(self):
        s = _make_series(round_num=4, round_name="NBA Finals")
        assert s.is_finals is True

    def test_not_finals(self):
        s = _make_series(round_num=1, round_name="First Round")
        assert s.is_finals is False

    def test_wnba_semis_best_of_5(self):
        # Only the WNBA Semifinals are best-of-5 (First Round is best-of-3,
        # Finals best-of-7 — see TestWnbaSeriesLength).
        s = _make_series(
            wins_a=2, wins_b=2, round_num=2,
            round_name="WNBA Semifinals - Game 5", sector="wnba",
        )
        assert s.is_game_7 is True  # Game 5 decides a best-of-5
        assert s._clinch == 3


class TestApplyAdjustments:
    def test_no_playoff_data_passthrough(self):
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data={},
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="celtics",
            team_b="pacers",
        )
        assert a == 0.55
        assert b == 0.45
        assert notes == ""
        assert is_playoff is False

    def test_no_matching_series_passthrough(self):
        data = {"lakers_vs_nuggets": _make_series(
            team_a="los angeles lakers", team_b="denver nuggets",
        )}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="celtics",
            team_b="pacers",
        )
        assert a == 0.55
        assert is_playoff is False

    def test_playoff_hca_boost(self):
        """Home team in a normal playoff game gets +1.5% HCA bump."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=1, wins_b=1)}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert is_playoff is True
        # Home team (boston = team_a in series, matches team_a in eval) gets boost
        assert a > 0.55
        assert "playoff_hca" in notes

    def test_elimination_boost(self):
        """Team facing elimination gets +3% boost."""
        # Boston leads 3-2, Indiana facing elimination (is away team in series)
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=2)}
        # Indiana is team_b in the eval AND team_b in the series
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.60,
            true_prob_b=0.40,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert is_playoff is True
        # Indiana (facing elimination) should get a boost, Boston (closeout) gets discount
        # Net: b should be higher than 0.40
        assert b > 0.40
        assert "elim_boost" in notes or "closeout" in notes

    def test_game_7_home_boost(self):
        """Game 7 gives massive home court advantage."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=3)}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",  # home
            team_b="indiana pacers",
        )
        assert is_playoff is True
        assert a > 0.55  # Home team gets game 7 + playoff HCA boost
        assert "game7_home" in notes

    def test_finals_bonus(self):
        """Finals games get small additional intensity adjustment."""
        data = {"boston celtics_vs_indiana pacers": _make_series(
            wins_a=2, wins_b=1, round_num=4, round_name="NBA Finals",
        )}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert "finals" in notes

    def test_probs_renormalize_to_1(self):
        """After adjustments, probabilities should sum to ~1.0."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=3)}
        a, b, _, _ = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert abs(a + b - 1.0) < 0.001

    def test_probs_clamped(self):
        """Extreme probabilities should be clamped to [0.02, 0.98]."""
        data = {"boston celtics_vs_indiana pacers": _make_series(wins_a=3, wins_b=3)}
        a, b, _, _ = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.98,
            true_prob_b=0.02,
            team_a="boston celtics",
            team_b="indiana pacers",
        )
        assert a <= 0.98
        assert b >= 0.02


class TestPaceAdjustment:
    def test_nba_pace(self):
        assert PlayoffAgent.pace_adjustment("nba") == 0.95

    def test_nhl_pace(self):
        assert PlayoffAgent.pace_adjustment("nhl") == 0.97

    def test_non_playoff_sport(self):
        assert PlayoffAgent.pace_adjustment("soccer") == 1.0


class TestHelpers:
    def test_parse_series_tied(self):
        h, a = _parse_series_summary("Series tied 2-2", "BOS", "IND")
        assert h == 2
        assert a == 2

    def test_parse_series_leads(self):
        h, a = _parse_series_summary("BOS leads 3-1", "BOS", "IND")
        assert h == 3
        assert a == 1

    def test_parse_series_away_leads(self):
        h, a = _parse_series_summary("IND leads 2-1", "BOS", "IND")
        assert h == 1
        assert a == 2

    def test_parse_series_empty(self):
        h, a = _parse_series_summary("", "BOS", "IND")
        assert h == 0
        assert a == 0

    def test_infer_round_first(self):
        assert _infer_round_number("First Round - Game 3", "nba") == 1

    def test_infer_round_semis(self):
        assert _infer_round_number("Conference Semifinals - Game 5", "nba") == 2

    def test_infer_round_conf_finals(self):
        assert _infer_round_number("Conference Finals - Game 7", "nba") == 3

    def test_infer_round_finals(self):
        assert _infer_round_number("NBA Finals - Game 1", "nba") == 4

    def test_infer_round_wnba_finals(self):
        assert _infer_round_number("WNBA Finals - Game 3", "wnba") == 3


class TestPlayInTournament:
    """Tests for Play-In tournament (ESPN season type 5) handling."""

    def _make_playin(
        self,
        team_a: str = "phoenix suns",
        team_b: str = "golden state warriors",
        sector: str = "nba",
    ) -> PlayoffSeries:
        return PlayoffSeries(
            team_a=team_a,
            team_b=team_b,
            team_a_abbrev="PHX",
            team_b_abbrev="GSW",
            series_wins_a=0,
            series_wins_b=0,
            round_num=0,
            round_name="Play-In Tournament",
            sector=sector,
            is_play_in=True,
        )

    def test_playin_properties(self):
        s = self._make_playin()
        assert s.is_play_in is True
        assert s.is_playoff is True
        assert s.game_number == 1
        assert s.is_elimination_for_a is True
        assert s.is_elimination_for_b is True
        assert s.is_game_7 is False
        assert s.is_closeout_for_a is False
        assert s.is_closeout_for_b is False

    def test_playin_hca_boost_home(self):
        """Play-in home team gets +2.0% HCA boost (on top of +1.5% playoff HCA)."""
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="phoenix suns",
            team_b="golden state warriors",
        )
        assert is_playoff is True
        assert a > 0.55  # Home team gets both playoff HCA + play-in HCA
        assert "playin_hca" in notes
        assert "playoff_hca" in notes

    def test_playin_no_elimination_boost(self):
        """Play-in should NOT trigger one-sided elimination boost."""
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="phoenix suns",
            team_b="golden state warriors",
        )
        assert "elim_boost" not in notes

    def test_playin_symmetric_for_away(self):
        """When eval team_a is the away team in the series, away gets no HCA boost."""
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        # Swap: team_a in eval = warriors (away), team_b in eval = suns (home)
        a, b, notes, is_playoff = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.45,
            true_prob_b=0.55,
            team_a="golden state warriors",
            team_b="phoenix suns",
        )
        assert is_playoff is True
        # Suns (team_b in eval, home in series) should get the boosts
        assert b > 0.55

    def test_playin_probs_sum_to_1(self):
        data = {"phoenix suns_vs_golden state warriors": self._make_playin()}
        a, b, _, _ = PlayoffAgent.apply_adjustments(
            playoff_data=data,
            true_prob_a=0.55,
            true_prob_b=0.45,
            team_a="phoenix suns",
            team_b="golden state warriors",
        )
        assert abs(a + b - 1.0) < 0.001


# ---------------------------------------------------------------------------
# WNBA series length by round (2025+ format): First Round best-of-3,
# Semifinals best-of-5, Finals best-of-7. The agent used to treat every WNBA
# round as best-of-5, which missed first-round elimination/decider games and
# flagged false ones in the Finals.
# ---------------------------------------------------------------------------


def _wnba(wins_a: int, wins_b: int, round_num: int, round_name: str) -> PlayoffSeries:
    return PlayoffSeries(
        team_a="las vegas aces",
        team_b="phoenix mercury",
        team_a_abbrev="LV",
        team_b_abbrev="PHX",
        series_wins_a=wins_a,
        series_wins_b=wins_b,
        round_num=round_num,
        round_name=round_name,
        sector="wnba",
    )


def _adjust(series: PlayoffSeries) -> tuple[float, float, str, bool]:
    return PlayoffAgent.apply_adjustments(
        playoff_data={"las vegas aces_vs_phoenix mercury": series},
        true_prob_a=0.55,
        true_prob_b=0.45,
        team_a="las vegas aces",
        team_b="phoenix mercury",
    )


class TestWnbaSeriesLength:
    def test_first_round_is_best_of_3(self):
        assert _wnba(0, 0, 1, "First Round - Game 1")._clinch == 2

    def test_first_round_game_2_trailing_team_faces_elimination(self):
        s = _wnba(0, 1, 1, "First Round - Game 2")  # away leads 1-0
        assert s.is_elimination_for_a is True
        assert s.is_closeout_for_b is True
        assert s.is_game_7 is False

    def test_first_round_game_3_is_the_decider(self):
        s = _wnba(1, 1, 1, "First Round - Game 3")
        assert s.is_game_7 is True
        assert s.is_elimination_for_a is False
        assert s.is_elimination_for_b is False

    def test_first_round_game_1_has_no_series_state(self):
        s = _wnba(0, 0, 1, "First Round - Game 1")
        assert not (s.is_elimination_for_a or s.is_elimination_for_b)
        assert not (s.is_closeout_for_a or s.is_closeout_for_b)
        assert s.is_game_7 is False

    def test_semis_is_best_of_5(self):
        s = _wnba(2, 1, 2, "WNBA Semifinals - Game 4")
        assert s._clinch == 3
        assert s.is_closeout_for_a is True
        assert s.is_elimination_for_b is True

    def test_finals_is_best_of_7(self):
        assert _wnba(0, 0, 3, "WNBA Finals - Game 1")._clinch == 4

    def test_finals_2_1_is_not_an_elimination_game(self):
        # Regression: under best-of-5 this read as closeout/elimination.
        s = _wnba(2, 1, 3, "WNBA Finals - Game 4")
        assert not (s.is_elimination_for_a or s.is_elimination_for_b)
        assert not (s.is_closeout_for_a or s.is_closeout_for_b)

    def test_finals_2_2_is_not_the_decider(self):
        assert _wnba(2, 2, 3, "WNBA Finals - Game 5").is_game_7 is False

    def test_finals_3_2_closeout_and_3_3_decider(self):
        s = _wnba(3, 2, 3, "WNBA Finals - Game 6")
        assert s.is_closeout_for_a is True
        assert s.is_elimination_for_b is True
        assert _wnba(3, 3, 3, "WNBA Finals - Game 7").is_game_7 is True

    def test_unrecognized_round_disables_series_state(self):
        s = _wnba(1, 1, 0, "Playoff Series")
        assert s._clinch is None
        assert s.is_game_7 is False
        assert not (s.is_elimination_for_a or s.is_elimination_for_b)
        assert not (s.is_closeout_for_a or s.is_closeout_for_b)
        assert s.is_finals is False

    def test_unrecognized_round_still_gets_playoff_hca_only(self):
        a, _, notes, is_playoff = _adjust(_wnba(1, 1, 0, "Playoff Series"))
        assert is_playoff is True
        assert "playoff_hca" in notes
        assert "game7_home" not in notes
        assert "elim_boost" not in notes
        assert a > 0.55

    def test_first_round_elimination_boost_applied(self):
        # Home (Aces) down 0-1 in a best-of-3 → faces elimination.
        _, _, notes, _ = _adjust(_wnba(0, 1, 1, "First Round - Game 2"))
        assert "elim_boost" in notes

    def test_finals_2_1_gets_no_elimination_boost(self):
        _, _, notes, _ = _adjust(_wnba(2, 1, 3, "WNBA Finals - Game 4"))
        assert "elim_boost" not in notes
        assert "closeout" not in notes
        assert "finals:" in notes


class TestFinalsFlag:
    def test_wnba_finals_is_finals(self):
        assert _wnba(0, 0, 3, "WNBA Finals - Game 1").is_finals is True

    def test_wnba_semifinals_is_not_finals(self):
        # Regression: "Semifinals" contains "final".
        assert _wnba(0, 0, 2, "WNBA Semifinals - Game 1").is_finals is False

    def test_nba_semifinals_is_not_finals(self):
        s = _make_series(round_num=2, round_name="East Semifinals - Game 1")
        assert s.is_finals is False

    def test_wnba_semifinals_gets_no_finals_bump(self):
        _, _, notes, _ = _adjust(_wnba(1, 0, 2, "WNBA Semifinals - Game 2"))
        assert "finals:" not in notes


class TestInferRoundWnba:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("First Round - Game 1", 1),
            ("First Round - Game 3", 1),
            ("Semifinals - Game 1", 2),
            ("WNBA Semifinals - Game 2", 2),
            ("WNBA Finals - Game 4", 3),
            ("Playoff Series", 0),
            ("", 0),
        ],
    )
    def test_round_from_espn_headline(self, name: str, expected: int):
        assert _infer_round_number(name, "wnba") == expected


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def get(self, *args, **kwargs) -> _FakeResponse:
        return _FakeResponse(self._payload)


def _espn_event(headline: str, home_wins: int, away_wins: int) -> dict:
    """Minimal ESPN scoreboard event in the shape the 2025 WNBA playoffs used."""
    return {
        "season": {"type": 3},
        "competitions": [{
            "notes": [{"headline": headline}],
            "series": {
                "type": "playoff",
                "competitors": [
                    {"id": "17", "wins": home_wins},
                    {"id": "5", "wins": away_wins},
                ],
            },
            "competitors": [
                {"id": "17", "homeAway": "home",
                 "team": {"displayName": "Las Vegas Aces", "abbreviation": "LV"}},
                {"id": "5", "homeAway": "away",
                 "team": {"displayName": "Phoenix Mercury", "abbreviation": "PHX"}},
            ],
        }],
    }


class TestFetchWnbaRound:
    @pytest.mark.parametrize(
        ("headline", "round_num", "clinch"),
        [
            ("First Round - Game 2", 1, 2),
            ("WNBA Semifinals - Game 3", 2, 3),
            ("WNBA Finals - Game 4", 3, 4),
        ],
    )
    async def test_espn_headline_sets_series_length(
        self, monkeypatch, headline: str, round_num: int, clinch: int,
    ):
        import evmax.agents.intelligence.playoff_agent as mod

        payload = {"events": [_espn_event(headline, home_wins=1, away_wins=0)]}
        monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: _FakeClient(payload))

        series_map = await PlayoffAgent()._fetch_playoff_data("unused", "wnba")

        s = series_map["las vegas aces_vs_phoenix mercury"]
        assert (s.series_wins_a, s.series_wins_b) == (1, 0)
        assert s.round_num == round_num
        assert s._clinch == clinch
