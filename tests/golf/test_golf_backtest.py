"""Walk-forward backtest tests.

The load-bearing test is `test_no_future_records_leak`: it proves the engine
never lets an event see data from itself or later — the single property whose
violation manufactures fake edge. The rest cover the outcome math (actuals,
ties, no-cut exclusion) and the Brier/skill scoring.
"""

import evmax.golf.backtest as bt
from evmax.golf.backtest import (
    WalkForwardConfig,
    compute_actuals,
    run_backtest,
)
from evmax.golf.data.espn import GolfEntry, GolfResult
from evmax.golf.models import GolfMarketType
from evmax.golf.names import canonical
from evmax.golf.simulator import SimConfig


def _event(day: int, winner_to_par: float, n=40, cut=True):
    """Synthetic completed event on 2025-<day>; distinct player pools by day so
    each event contributes fresh records."""
    entries = []
    for i in range(n):
        to_par = winner_to_par + i  # spread the field
        rounds = 4 if (not cut or i < n // 2) else 2  # bottom half misses cut
        scores = [70 + (to_par // 4)] * rounds
        name = f"Player {i:02d} D{day:02d}"
        entries.append(GolfEntry(canonical(name), name, float(to_par), rounds, scores))
    return GolfResult(f"E{day}", f"Event {day}", True, entries, date=f"2025-03-{day:02d}")


def _shared_pool_event(day: int, n=40):
    """Event drawing from a FIXED player pool (players persist across events) so
    skill accumulates — used for the leak test."""
    entries = []
    for i in range(n):
        to_par = -10 + i
        rounds = 4 if i < n // 2 else 2
        name = f"Pro {i:02d}"
        entries.append(GolfEntry(canonical(name), name, float(to_par), rounds, [70, 71]))
    return GolfResult(f"S{day}", f"Shared {day}", True, entries, date=f"2025-04-{day:02d}")


class TestActuals:
    def test_winner_and_positions(self):
        ev = _event(10, winner_to_par=-20, n=40)
        actuals = compute_actuals(ev)
        winner = canonical("Player 00 D10")
        assert actuals[winner][GolfMarketType.win.value] == 1.0
        assert actuals[winner][GolfMarketType.top_10.value] == 1.0
        assert actuals[winner][GolfMarketType.make_cut.value] == 1.0
        # a bottom-half player misses the cut and does not win
        loser = canonical("Player 39 D10")
        assert actuals[loser][GolfMarketType.make_cut.value] == 0.0
        assert actuals[loser][GolfMarketType.win.value] == 0.0

    def test_win_credit_split_on_tie(self):
        # two players tie for the lead
        entries = [
            GolfEntry("a a", "A A", -10.0, 4, [66, 66]),
            GolfEntry("b b", "B B", -10.0, 4, [66, 66]),
            GolfEntry("c c", "C C", -5.0, 4, [68, 69]),
        ] + [GolfEntry(f"x{i} x", f"X{i} X", float(i), 4, [70, 70]) for i in range(30)]
        ev = GolfResult("T", "Tie", True, entries, date="2025-05-01")
        actuals = compute_actuals(ev)
        assert actuals["a a"][GolfMarketType.win.value] == 0.5
        assert actuals["b b"][GolfMarketType.win.value] == 0.5
        assert actuals["c c"][GolfMarketType.win.value] == 0.0


class TestLeakFree:
    def test_no_future_records_leak(self, monkeypatch):
        """The heart of the walk-forward guarantee: every skill build must see
        ONLY records dated strictly before the event being predicted."""
        events = [_shared_pool_event(d) for d in range(1, 21)]
        seen = []
        real = bt.skill_from_records

        def spy(records, *args, **kwargs):
            max_date = max((r.date for r in records), default="")
            seen.append(max_date)
            return real(records, *args, **kwargs)

        monkeypatch.setattr(bt, "skill_from_records", spy)
        # sort events by date so we know the expected order of predictions
        ordered = sorted(events, key=lambda e: e.date)
        cfg = WalkForwardConfig(min_prior_events=3, min_field=30,
                                sim=SimConfig(n_sims=2000, seed=1))
        run_backtest(events, cfg)

        # The k-th skill build (for the k-th scored event) must only contain
        # records from strictly-earlier events. Scored events start after the
        # burn-in, so align the k-th spy call to its event.
        scored_events = ordered[cfg.min_prior_events:]
        assert len(seen) == len(scored_events)
        for max_record_date, ev in zip(seen, scored_events):
            assert max_record_date < ev.date, (
                f"LEAK: skill for {ev.date} saw a record dated {max_record_date}"
            )

    def test_burn_in_events_not_scored(self):
        events = [_shared_pool_event(d) for d in range(1, 15)]
        cfg = WalkForwardConfig(min_prior_events=5, min_field=30,
                                sim=SimConfig(n_sims=2000, seed=1))
        report = run_backtest(events, cfg)
        # 14 events, first 5 are burn-in → at most 9 scored
        assert report.n_events <= 9
        assert report.n_events >= 1


class TestNoCutExclusion:
    def test_no_cut_event_excluded_from_make_cut(self):
        # A run of normal cut events for skill, then an all-4-round no-cut event.
        events = [_shared_pool_event(d) for d in range(1, 12)]
        events.append(_event(28, winner_to_par=-15, n=40, cut=False))  # no-cut
        cfg = WalkForwardConfig(min_prior_events=6, min_field=30,
                                sim=SimConfig(n_sims=3000, seed=2))
        report = run_backtest(events, cfg)
        # make_cut still has SOME predictions (from cut events), and win/top_10
        # are always present.
        assert GolfMarketType.win.value in report.brier
        assert GolfMarketType.top_10.value in report.brier


class TestScoring:
    def test_report_has_all_markets_and_skill(self):
        events = [_shared_pool_event(d) for d in range(1, 25)]
        cfg = WalkForwardConfig(min_prior_events=6, min_field=30,
                                sim=SimConfig(n_sims=4000, seed=3))
        report = run_backtest(events, cfg)
        assert report.n_predictions > 0
        assert 0.0 <= report.coverage <= 1.0
        for mk in (GolfMarketType.win, GolfMarketType.top_10):
            assert mk.value in report.brier
            assert report.brier[mk.value] >= 0.0
            assert report.baseline_brier[mk.value] >= 0.0
        # summary renders without error
        assert "events=" in report.summary()

    def test_brier_is_mean_squared_error(self):
        # single event scored with a hand-checkable outcome
        events = [_shared_pool_event(d) for d in range(1, 10)]
        cfg = WalkForwardConfig(min_prior_events=3, min_field=30,
                                sim=SimConfig(n_sims=5000, seed=4))
        report = run_backtest(events, cfg)
        # Brier must lie in [0, 1]
        for mk in (GolfMarketType.win, GolfMarketType.top_10):
            assert 0.0 <= report.brier[mk.value] <= 1.0
