"""Tests for the NFL ridge rating (evmax.agents.models._nfl_ridge) and its seed wiring."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from evmax.agents.models._nfl_ridge import ridge_ratings

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _plays(spec, start=date(2024, 9, 1)):
    """spec: list of (week, posteam, defteam, home_team, epa, n_plays)."""
    rows = []
    for week, pos, dfn, home, epa, n in spec:
        d = start + timedelta(weeks=week)
        rows += [dict(posteam=pos, defteam=dfn, home_team=home, epa=epa, game_date=d)] * n
    return pl.DataFrame(rows).with_columns(pl.col("game_date").cast(pl.Date))


def _round_robin(strength, weeks=6, per=40, hfa=0.0):
    """Every pair plays both ways each week; EPA = off strength - opp def strength (+hfa at home)."""
    spec = []
    for w in range(weeks):
        for i, a in enumerate(TEAMS):
            for b in TEAMS[i + 1:]:
                home = a if (w + i) % 2 == 0 else b
                for pos, dfn in ((a, b), (b, a)):
                    spec.append((w, pos, dfn, home, strength[pos] - strength[dfn] * 0 + (hfa if pos == home else 0.0), per))
    return _plays(spec)


class TestRidgeRatings:
    def test_empty_frame(self):
        assert ridge_ratings(pl.DataFrame(schema={"posteam": pl.Utf8, "defteam": pl.Utf8, "home_team": pl.Utf8,
                                                  "epa": pl.Float64, "game_date": pl.Date})) == {}

    def test_string_game_date_matches_date_column(self):
        # nflverse PBP carries game_date as a String (the weekly seed's real input).
        df = _round_robin({"AAA": 0.2, "BBB": 0.0, "CCC": 0.0, "DDD": -0.2})
        as_str = df.with_columns(pl.col("game_date").dt.strftime("%Y-%m-%d"))
        assert as_str.schema["game_date"] == pl.Utf8
        a, b = ridge_ratings(df, lam=10.0), ridge_ratings(as_str, lam=10.0)
        assert a.keys() == b.keys()
        for t in a:
            assert a[t]["net"] == pytest.approx(b[t]["net"])

    def test_ranks_stronger_offense_higher(self):
        r = ridge_ratings(_round_robin({"AAA": 0.20, "BBB": 0.05, "CCC": -0.05, "DDD": -0.20}), lam=10.0)
        nets = [r[t]["net"] for t in TEAMS]
        assert nets == sorted(nets, reverse=True)

    def test_net_is_off_minus_def(self):
        r = ridge_ratings(_round_robin({"AAA": 0.2, "BBB": 0.0, "CCC": 0.0, "DDD": -0.2}), lam=10.0)
        for v in r.values():
            assert v["net"] == pytest.approx(v["off"] - v["def"])

    def test_home_field_does_not_leak_into_ratings(self):
        # Identical teams; offense is +0.3 EPA at home. The joint home term must absorb it,
        # so no team looks better because it had more home games.
        flat = {t: 0.0 for t in TEAMS}
        r = ridge_ratings(_round_robin(flat, hfa=0.3), lam=1.0)
        assert max(abs(v["net"]) for v in r.values()) < 0.05

    def test_ridge_shrinks_toward_zero(self):
        data = _round_robin({"AAA": 0.2, "BBB": 0.0, "CCC": 0.0, "DDD": -0.2})
        loose = ridge_ratings(data, lam=1.0)["AAA"]["net"]
        tight = ridge_ratings(data, lam=1e6)["AAA"]["net"]
        assert abs(tight) < abs(loose)

    def test_recency_half_life_favours_recent_form(self):
        # AAA is bad for 8 weeks then great for 4; BBB the mirror image. With no recency
        # AAA/BBB tie-ish; with a short half-life AAA must rank above BBB.
        spec = []
        for w in range(12):
            good_a = w >= 8
            spec.append((w, "AAA", "BBB", "AAA", 0.2 if good_a else -0.2, 60))
            spec.append((w, "BBB", "AAA", "BBB", -0.2 if good_a else 0.2, 60))
        df = _plays(spec)
        flat = ridge_ratings(df, lam=1.0, week_half_life=None)
        recent = ridge_ratings(df, lam=1.0, week_half_life=3.0)
        assert recent["AAA"]["net"] - recent["BBB"]["net"] > flat["AAA"]["net"] - flat["BBB"]["net"]
        assert recent["AAA"]["net"] > recent["BBB"]["net"]


class TestSeedWiring:
    def _frame(self):
        df = _round_robin({"AAA": 0.2, "BBB": 0.0, "CCC": 0.0, "DDD": -0.2})
        return df.with_columns(
            pl.lit(2024).alias("season"),
            pl.lit(0.5).alias("success"),
            pl.lit(5.0).alias("yards_gained"),
            pl.lit(0).alias("touchdown"),
            pl.lit(50.0).alias("yardline_100"),
            pl.lit("g").alias("game_id"),
        )

    def test_team_stats_carry_ridge_fields(self, monkeypatch):
        import scripts.seed_nfl_efficiency as seed

        monkeypatch.setattr(seed, "NFL_ABBREV_TO_NAME", {t: t.lower() for t in TEAMS})
        out = seed.compute_team_stats(self._frame(), 2024)
        assert set(out) == {t.lower() for t in TEAMS}
        for v in out.values():
            assert {"ridge_net", "ridge_off", "ridge_def"} <= set(v)
        assert out["aaa"]["ridge_net"] > out["ddd"]["ridge_net"]

    def test_seed_accepts_string_game_date(self, monkeypatch):
        import scripts.seed_nfl_efficiency as seed

        monkeypatch.setattr(seed, "NFL_ABBREV_TO_NAME", {t: t.lower() for t in TEAMS})
        df = self._frame().with_columns(pl.col("game_date").dt.strftime("%Y-%m-%d"))
        out = seed.compute_team_stats(df, 2024)
        assert all("ridge_net" in v for v in out.values())

    def test_frame_without_ridge_columns_omits_fields(self, monkeypatch):
        import scripts.seed_nfl_efficiency as seed

        monkeypatch.setattr(seed, "NFL_ABBREV_TO_NAME", {t: t.lower() for t in TEAMS})
        out = seed.compute_team_stats(self._frame().drop("home_team"), 2024)
        assert out and all("ridge_net" not in v for v in out.values())
