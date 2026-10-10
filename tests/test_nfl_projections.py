"""evmax.nfl_projections — data layer, team-game table, ratings, QB layer, game model, live helpers."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evmax.nfl_projections import data, live, team_games
from evmax.nfl_projections.game_model import (
    MEDIAN_OUTDOOR_WIND_MPH, Combiner, GameModelConfig, context_features, fit_combiner,
    walk_forward,
)
from evmax.nfl_projections.ratings import (
    QBRatings, fit_qb_ratings, fit_rating, nfl_season_of, recency_weights,
)

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


# ── data layer ────────────────────────────────────────────────────────────────

def test_data_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("EVMAX_NFL_PROJ_DATA", str(tmp_path / "env"))
    assert data.data_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert data.data_dir() == tmp_path / "env"
    monkeypatch.delenv("EVMAX_NFL_PROJ_DATA")
    assert data.data_dir().parts[-3:] == ("data", "backtest", "nfl_projections")


def test_load_games_and_pbp_fold_relocated_franchises(tmp_path):
    g = pd.DataFrame({"game_id": ["2016_01_OAK_SD"], "gameday": ["2016-09-11"],
                      "home_team": ["SD"], "away_team": ["OAK"]})
    g.to_parquet(data.games_file(tmp_path))
    out = data.load_games(tmp_path)
    assert out.loc[0, "home_team"] == "LAC" and out.loc[0, "away_team"] == "LV"
    assert out["gameday"].dtype.kind == "M"

    p = pd.DataFrame({c: [None] for c in data.PBP_COLUMNS})
    p[["posteam", "defteam", "home_team", "away_team"]] = ["STL", "OAK", "STL", "OAK"]
    data.pbp_file(2015, tmp_path).parent.mkdir(parents=True)
    p.to_parquet(data.pbp_file(2015, tmp_path))
    out = data.load_pbp([2015, 2099], tmp_path)  # a missing season is skipped
    assert list(out.loc[0, ["posteam", "defteam"]]) == ["LA", "LV"]


# ── team-game table ───────────────────────────────────────────────────────────

def _play(play_id, posteam, defteam, *, pas=1, epa=0.1, success=1, wp=0.5, drive=1,
          result="Punt", inside20=0, passer="QB1", qb_epa=0.1, dropback=1):
    return {"game_id": "2024_01_BBB_AAA", "play_id": play_id, "season": 2024, "week": 1,
            "season_type": "REG", "posteam": posteam, "defteam": defteam, "home_team": "AAA",
            "away_team": "BBB", "play_type": "pass" if pas else "run", "pass": pas, "rush": 1 - pas,
            "qb_dropback": dropback, "epa": epa, "success": success, "wp": wp, "pass_oe": 5.0,
            "down": 1, "yards_gained": 5, "fixed_drive": drive, "fixed_drive_result": result,
            "drive_time_of_possession": "2:30", "drive_inside20": inside20, "interception": 0,
            "fumble_lost": 0, "passer_player_id": passer if pas else None,
            "passer_player_name": passer if pas else None, "qb_epa": qb_epa if pas else None}


def _schedule_row(game_id="2024_01_BBB_AAA", home="AAA", away="BBB", hs=24, as_=17, location="Home"):
    return {"game_id": game_id, "season": 2024, "week": 1, "game_type": "REG",
            "gameday": pd.Timestamp("2024-09-08"), "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_, "location": location}


def test_build_team_games_counts_filters_and_starters():
    pbp = pd.DataFrame([
        # AAA: QB1 starts (first dropback), QB2 takes over and throws more
        # drive 1 ends in a TD; inside20 is set on only one of its plays (seen in ~0.1%
        # of real drives) -> the drive still counts as a red-zone trip
        _play(1, "AAA", "BBB", passer="QB1", drive=1, result="Touchdown"),
        _play(2, "AAA", "BBB", passer="QB2", drive=1, result="Touchdown", inside20=1),
        _play(3, "AAA", "BBB", passer="QB2", drive=3, epa=5.0, wp=0.97),   # garbage time
        _play(4, "AAA", "BBB", pas=0, epa=-0.2, success=0, drive=3, dropback=0),
        # BBB
        _play(5, "BBB", "AAA", passer="QB9", drive=2, result="Field goal", inside20=1),
    ])
    pbp.loc[len(pbp)] = {**_play(6, "AAA", "BBB", drive=3), "play_type": "no_play"}  # nullified
    tg = team_games.build_team_games(pbp, pd.DataFrame([_schedule_row()]))
    a = tg.set_index("team").loc["AAA"]
    assert a["plays"] == 4 and a["comp_plays"] == 3          # no_play excluded; wp 0.97 not competitive
    assert a["epa_pp"] == pytest.approx((0.1 + 0.1 - 0.2) / 3)
    assert a["drives"] == 2 and a["rz_drives"] == 1 and a["rz_td"] == 1
    assert a["first_qb_id"] == "QB1" and a["starter_id"] == "QB2"
    assert a["home"] == 1 and a["opp"] == "BBB" and a["points_for"] == 24 and a["points_against"] == 17
    b = tg.set_index("team").loc["BBB"]
    assert b["home"] == 0 and b["points_for"] == 17


def test_build_team_games_neutral_site_has_no_home_team():
    pbp = pd.DataFrame([_play(1, "AAA", "BBB"), _play(2, "BBB", "AAA", passer="QB9", drive=2)])
    tg = team_games.build_team_games(pbp, pd.DataFrame([_schedule_row(location="Neutral")]))
    assert tg["home"].tolist() == [0, 0] and tg["neutral"].tolist() == [1, 1]


# ── ratings ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ts,season", [("2024-09-08", 2024), ("2025-01-12", 2024),
                                       ("2025-02-09", 2024), ("2025-03-01", 2025)])
def test_nfl_season_of(ts, season):
    assert nfl_season_of(pd.Timestamp(ts)) == season


def test_recency_weights_exclude_offseason_and_never_exceed_one():
    days = pd.Series(pd.to_datetime(["2024-12-29", "2025-02-09", "2025-09-07"]))
    cutoff = pd.Timestamp("2025-09-14")
    plain = recency_weights(days, cutoff, 70.0)
    comp = recency_weights(days, cutoff, 70.0, offseason_days=180.0)
    assert comp[0] > plain[0] and comp[1] > plain[1]              # last season counts as more recent
    assert comp[2] == pytest.approx(plain[2])                      # same-season rows unaffected
    assert np.all(comp <= 1.0) and comp[1] > comp[0]               # order within last season kept
    # age is clipped at 0, never negative
    assert recency_weights(days, cutoff, 70.0, offseason_days=10_000.0).max() == 1.0


def _league(seasons=(2022, 2023), off=None, deff=None, hfa=2.0, mu=22.0, seed=0, noise=0.0):
    """Synthetic round-robin league: every pair plays home and away each season."""
    rng = np.random.default_rng(seed)
    off = off or {"AAA": 4.0, "BBB": 1.0, "CCC": -1.0, "DDD": -4.0}
    deff = deff or {"AAA": -3.0, "BBB": 0.0, "CCC": 1.0, "DDD": 2.0}
    tg_rows, game_rows = [], []
    for season in seasons:
        day = pd.Timestamp(f"{season}-09-10")
        week = 1
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                gid = f"{season}_{week:02d}_{a}_{h}"
                hp = mu + hfa + off[h] + deff[a] + noise * rng.normal()
                ap = mu + off[a] + deff[h] + noise * rng.normal()
                for team, opp, home, pts, pa in ((h, a, 1, hp, ap), (a, h, 0, ap, hp)):
                    tg_rows.append({
                        "game_id": gid, "season": season, "week": week, "season_type": "REG",
                        "gameday": day, "team": team, "opp": opp, "home": home, "neutral": 0,
                        "points_for": pts, "points_against": pa, "plays": 60, "comp_plays": 50,
                        "epa_pp": pts / 100.0, "sr": 0.4 + pts / 1000.0, "starter_id": f"{team}-QB",
                        "starter_epa": 0.05, "starter_dropbacks": 35, "first_qb_id": f"{team}-QB"})
                game_rows.append({
                    "game_id": gid, "season": season, "week": week, "game_type": "REG",
                    "gameday": day, "home_team": h, "away_team": a, "home_score": hp,
                    "away_score": ap, "location": "Home", "roof": "outdoors", "wind": 5.0,
                    "temp": 60.0, "div_game": 0, "spread_line": hp - ap, "total_line": hp + ap})
                day += pd.Timedelta(days=7)
                week += 1
    return pd.DataFrame(tg_rows), pd.DataFrame(game_rows)


def test_fit_rating_recovers_additive_truth():
    tg, _ = _league()
    fit = fit_rating(tg, "points_for", pd.Timestamp("2030-01-01"), half_life_days=1e9, lam=1e-6)
    assert fit.hfa == pytest.approx(2.0, abs=1e-6)
    # off/def are identified up to a constant; compare expected points directly
    assert fit.expect("AAA", "DDD", 1) == pytest.approx(22 + 2 + 4 + 2, abs=1e-6)
    assert fit.expect("DDD", "AAA", 0) == pytest.approx(22 - 4 - 3, abs=1e-6)


def test_fit_rating_ignores_rows_on_or_after_cutoff():
    tg, _ = _league(seasons=(2022,))
    cutoff = tg["gameday"].sort_values().iloc[len(tg) // 2]
    before = fit_rating(tg, "points_for", cutoff)
    poisoned = tg.copy()
    poisoned.loc[poisoned["gameday"] >= cutoff, "points_for"] = 999.0
    after = fit_rating(poisoned, "points_for", cutoff)
    assert before.expect("AAA", "BBB", 1) == pytest.approx(after.expect("AAA", "BBB", 1))


# ── QB layer ──────────────────────────────────────────────────────────────────

def test_qb_delta_usual_starter_zero_backup_negative_missing_zero():
    tg, _ = _league(seasons=(2022,))
    tg.loc[tg["team"] == "AAA", "starter_epa"] = 0.20
    q = fit_qb_ratings(tg, pd.Timestamp("2023-01-01"))
    assert q.delta("AAA", "AAA-QB") == pytest.approx(0.0, abs=1e-12)
    # unknown starter -> shrunk to league mean - 0.05, below this team's QB
    assert q.prior == pytest.approx(0.0875 - 0.05)
    assert q.delta("AAA", "never-seen") == pytest.approx(q.prior - q.rating("AAA-QB"))
    assert q.delta("AAA", "never-seen") < 0
    assert q.delta("AAA", None) == 0.0 and q.delta("AAA", float("nan")) == 0.0
    assert q.delta("ZZZ", "AAA-QB") == 0.0                          # team with no history


def test_qb_ratings_empty_history():
    tg, _ = _league(seasons=(2022,))
    q = fit_qb_ratings(tg, pd.Timestamp("2000-01-01"))
    assert isinstance(q, QBRatings) and q.ratings == {} and q.delta("AAA", "x") == 0.0


# ── game model ────────────────────────────────────────────────────────────────

def _g(**kw):
    base = {"roof": "outdoors", "wind": 12.0}
    base.update(kw)
    return pd.Series(base)


def test_context_features():
    assert context_features(_g(roof="dome", wind=20.0)) == {"dome": 1.0, "wind": 0.0}
    assert context_features(_g(roof="closed")) == {"dome": 1.0, "wind": 0.0}
    assert context_features(_g()) == {"dome": 0.0, "wind": 12.0}
    assert context_features(_g(wind=float("nan")))["wind"] == MEDIAN_OUTDOOR_WIND_MPH
    assert context_features(_g(roof=""))["dome"] == 0.0


def test_fit_combiner_recovers_linear_map_and_predicts():
    rng = np.random.default_rng(1)
    rows = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    rows["points"] = 3.0 + 2.0 * rows["a"] - 0.5 * rows["b"]
    comb = fit_combiner(rows, ("a", "b"))
    assert comb.coef == pytest.approx((3.0, 2.0, -0.5))
    assert Combiner(("a",), (1.0, 2.0)).predict({"a": 3.0}) == 7.0


def test_walk_forward_is_leak_free_and_consistent():
    tg, games = _league(seasons=(2020, 2021, 2022), noise=3.0)
    cfg = GameModelConfig(first_feature_season=2020, features=("pts", "epa", "sr", "dome", "wind", "qb"))
    base = walk_forward(tg, games, [2022], cfg)
    assert len(base) == (games["season"] == 2022).sum()
    assert np.allclose(base["proj_margin"], base["proj_home"] - base["proj_away"])
    assert np.allclose(base["proj_total"], base["proj_home"] + base["proj_away"])

    # Poison one mid-season 2022 game: projections up to and including its week must not move.
    g22 = games[games["season"] == 2022].sort_values("gameday")
    target = g22.iloc[len(g22) // 2]
    tg2, games2 = tg.copy(), games.copy()
    games2.loc[games2["game_id"] == target.game_id, ["home_score", "away_score"]] = [99.0, 0.0]
    tg2.loc[tg2["game_id"] == target.game_id, ["points_for", "epa_pp"]] = [99.0, 0.99]
    after = walk_forward(tg2, games2, [2022], cfg)
    m = base.merge(after, on="game_id", suffixes=("", "_p"))
    early = m["week"] <= target.week
    assert np.allclose(m.loc[early, "proj_margin"], m.loc[early, "proj_margin_p"])
    assert not np.allclose(m.loc[~early, "proj_margin"], m.loc[~early, "proj_margin_p"])


# ── live helpers ──────────────────────────────────────────────────────────────

def test_next_week_infer_roof_latest_starters():
    games = pd.DataFrame({
        "season": [2026, 2026, 2026], "week": [4, 5, 5],
        "gameday": pd.to_datetime(["2026-10-04", "2026-10-11", "2026-10-12"]),
        "home_score": [21.0, None, None], "stadium_id": ["ATL97", "ATL97", "ATL97"],
        "roof": ["closed", "", ""]})
    assert live.next_week(games, date(2026, 10, 9)) == (2026, 5)
    with pytest.raises(ValueError):
        live.next_week(games, date(2027, 3, 1))
    assert live.infer_roof(games, "ATL97", "") == "closed"
    assert live.infer_roof(games, "ATL97", "open") == "open"
    assert live.infer_roof(games, "NEW00", "") == "outdoors"

    tg = pd.DataFrame({"team": ["AAA", "AAA", "BBB"], "first_qb_id": ["old", "new", "b1"],
                       "gameday": pd.to_datetime(["2026-09-07", "2026-09-14", "2026-09-07"])})
    assert live.latest_starters(tg, pd.Timestamp("2026-09-20")) == {"AAA": "new", "BBB": "b1"}
    assert live.latest_starters(tg, pd.Timestamp("2026-09-10")) == {"AAA": "old", "BBB": "b1"}


def test_cover_and_over_probabilities():
    p = live.home_cover_probability(3.0, -3.5)
    assert p is not None and 0.35 < p < 0.5                      # favored by 3, laying 3.5
    assert live.home_cover_probability(-3.0, 3.5) == pytest.approx(1 - p)
    assert live.home_cover_probability(7.0, -3.5) > live.home_cover_probability(3.0, -3.5) > 0
    assert live.total_over_probability(44.0, 44.0) == pytest.approx(0.5)
    assert live.total_over_probability(50.0, 44.0) > 0.6


def test_cli_line_format():
    from evmax.cli.commands.project import _nfl_line
    assert _nfl_line("DAL", "TB", 3.14) == "DAL -3.1"
    assert _nfl_line("DAL", "TB", -2.0) == "TB -2.0"
    assert _nfl_line("DAL", "TB", 0.01) == "PK"
    assert math.isfinite(len(_nfl_line("A", "B", 10.0)))


class _Stop(Exception):
    """Raised by a stub to end a heavy pipeline call once the step under test has run."""


def test_project_week_players_reuses_a_given_game_projection(monkeypatch):
    from evmax.nfl_projections import team_games

    def no_game_model(*a, **k):
        raise AssertionError("project_week must not run when game_proj is given")

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(live, "project_week", no_game_model)
    monkeypatch.setattr(team_games, "load_team_games", stop)
    with pytest.raises(_Stop):
        live.project_week_players(2026, 5, refresh=False, game_proj=pd.DataFrame())


def test_project_week_players_runs_the_game_model_without_one(monkeypatch):
    calls = []

    def game_model(season, week, *a, **k):
        calls.append((season, week))
        raise _Stop

    monkeypatch.setattr(live, "project_week", game_model)
    with pytest.raises(_Stop):
        live.project_week_players(2026, 5, refresh=False)
    assert calls == [(2026, 5)]


def test_simulate_game_reuses_given_player_projections(monkeypatch):
    from evmax.nfl_projections import player_games, simulate

    def no_player_model(*a, **k):
        raise AssertionError("project_week_players must not run when proj is given")

    monkeypatch.setattr(live, "project_week_players", no_player_model)
    monkeypatch.setattr(player_games, "load_player_games", lambda *a, **k: pd.DataFrame({"season_type": []}))
    monkeypatch.setattr(simulate, "fit_sim_params", lambda pg: simulate.SimParams())
    base = {"tgt_share": 0.3, "car_share": 0.3, "catch_rate": 0.6, "ypt": 8.0, "ypc": 4.0, "rush_xtd_share": 0.3,
            "rec_xtd_share": 0.3, "is_starting_qb": False, "exp_team_targets": 33.0, "exp_team_carries": 26.0,
            "exp_team_rush_tds": 0.9, "exp_team_rec_tds": 1.4}
    proj = pd.DataFrame([
        {**base, "game_id": "g1", "team": "AAA", "player_id": "a1"},
        {**base, "game_id": "g1", "team": "BBB", "player_id": "b1"},
        {**base, "game_id": "g2", "team": "CCC", "player_id": "c1"},
    ])
    out = live.simulate_game(2026, 5, "aaa", n=200, proj=proj)
    assert set(out) == {"AAA", "BBB"}                                    # both teams of AAA's game, not g2
    players, sims = out["AAA"]
    assert list(players["player_id"]) == ["a1"] and sims["receptions"].shape == (200, 1)
    with pytest.raises(ValueError, match="no projected game"):
        live.simulate_game(2026, 5, "ZZZ", n=10, proj=proj)
