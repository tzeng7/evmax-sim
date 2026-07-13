"""Walk-forward backtest for the golf skill→simulate pipeline.

Answers the calibration question directly: does a golfer the model rates a 15%
win actually win ~15% of the time? And is the model systematically
under/over-confident, or well-calibrated on its own terms?

**Point-in-time correctness is the whole game.** For each tournament we build
skill from ONLY the events that finished BEFORE it, simulate its actual field,
and score the model's win / top-10 / make-cut probabilities against what
actually happened. The current event's own result is folded into the skill pool
only AFTER it has been predicted — so there is no lookahead by construction
(the property most likely to produce a fake "edge"; see the repo's history of
PIT-replay lookahead bugs).

This backtests the ESPN **score-proxy** skill path (SG-total from scores),
because that is the only skill signal we can reconstruct leak-free at each point
in time. The live module's PGA season-SG snapshot CANNOT be backtested here — a
season aggregate already contains post-event rounds, which would leak. A
well-calibrated score-proxy is evidence the simulator + skill framework is
sound; the richer SG feed should do at least as well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from evmax.golf.data.espn import GolfResult, result_to_sg_proxy
from evmax.golf.models import GolfMarketType, SGRecord
from evmax.golf.simulator import SimConfig, simulate_field
from evmax.golf.skill import SkillConfig, skill_from_records, to_sim_inputs

_log = logging.getLogger(__name__)

# Calibration bins for reporting predicted-vs-realised frequency.
_BINS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10),
         (0.10, 0.20), (0.20, 0.40), (0.40, 1.01)]

_MARKETS = (GolfMarketType.win, GolfMarketType.top_10, GolfMarketType.make_cut)


@dataclass
class Prediction:
    event: str
    date: str
    golfer: str
    had_prior: bool  # did the golfer have any prior data (vs replacement)?
    model: dict  # market -> model prob
    actual: dict  # market -> realised outcome (0/1, fractional on ties)


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    mean_actual: float


@dataclass
class BacktestReport:
    n_events: int
    n_predictions: int
    coverage: float  # fraction of predictions with prior data
    brier: dict  # market -> model Brier (covered-only)
    baseline_brier: dict  # market -> naive-baseline Brier
    calibration: dict  # market -> list[CalibrationBin]
    winner_capture: float  # mean model win-prob assigned to the actual winner
    favorite_hit_rate: float  # how often the model's top win pick actually won
    n_favorite_events: int

    def summary(self) -> str:
        lines = [
            f"events={self.n_events}  predictions={self.n_predictions}  "
            f"coverage={self.coverage:.0%}",
            f"winner_capture (mean model P assigned to actual winner) = "
            f"{self.winner_capture:.2%}",
            f"favorite_hit_rate (model top pick wins) = {self.favorite_hit_rate:.1%} "
            f"over {self.n_favorite_events} events",
            "",
            f"{'market':10s} {'model_brier':>12s} {'base_brier':>11s} {'skill':>8s}",
        ]
        for mk in _MARKETS:
            mb = self.brier.get(mk.value)
            bb = self.baseline_brier.get(mk.value)
            if mb is None or bb is None:
                continue
            skill = (bb - mb) / bb if bb else 0.0  # Brier skill score vs baseline
            lines.append(f"{mk.value:10s} {mb:12.4f} {bb:11.4f} {skill:+7.1%}")
        for mk in _MARKETS:
            bins = self.calibration.get(mk.value, [])
            if not bins:
                continue
            lines.append(f"\ncalibration — {mk.value}  (pred → actual)")
            for b in bins:
                if b.n == 0:
                    continue
                lines.append(
                    f"  [{b.lo:.0%},{b.hi:.0%})  n={b.n:5d}  "
                    f"pred={b.mean_pred:.1%}  actual={b.mean_actual:.1%}"
                )
        return "\n".join(lines)


@dataclass
class WalkForwardConfig:
    min_prior_events: int = 8  # need enough prior events before scoring
    min_field: int = 30
    sim: SimConfig = field(default_factory=lambda: SimConfig(n_sims=10000, seed=99))
    skill: SkillConfig = field(default_factory=SkillConfig)


def compute_actuals(result: GolfResult) -> dict[str, dict]:
    """Realised outcomes per golfer from a completed event, from to-par totals.

    make_cut = played the weekend (>= 3 rounds). Finishing position is the
    competition rank (ties share the best rank) among made-cut players by
    to-par. Win credit is split on a tie for 1st (we can't see the playoff
    result from scores alone).
    """
    entries = [e for e in result.entries if e.to_par is not None and e.rounds_played > 0]
    made = [e for e in entries if e.rounds_played >= 3]
    made_sorted = sorted(made, key=lambda e: e.to_par)

    # competition rank (1-based, ties share the best position)
    pos: dict[str, int] = {}
    i = 0
    while i < len(made_sorted):
        j = i
        while j + 1 < len(made_sorted) and made_sorted[j + 1].to_par == made_sorted[i].to_par:
            j += 1
        rank = i + 1
        for k in range(i, j + 1):
            pos[made_sorted[k].player] = rank
        i = j + 1

    # tie count at the top for win-credit splitting
    top_par = made_sorted[0].to_par if made_sorted else None
    n_win_tie = sum(1 for e in made_sorted if e.to_par == top_par) if made_sorted else 1

    out: dict[str, dict] = {}
    made_players = {e.player for e in made}
    for e in entries:
        p = pos.get(e.player)
        won = (1.0 / n_win_tie) if (p == 1) else 0.0
        out[e.player] = {
            GolfMarketType.win.value: won,
            GolfMarketType.top_5.value: 1.0 if (p is not None and p <= 5) else 0.0,
            GolfMarketType.top_10.value: 1.0 if (p is not None and p <= 10) else 0.0,
            GolfMarketType.top_20.value: 1.0 if (p is not None and p <= 20) else 0.0,
            GolfMarketType.make_cut.value: 1.0 if e.player in made_players else 0.0,
        }
    return out


def collect_predictions(
    events: Sequence[GolfResult], config: Optional[WalkForwardConfig] = None
) -> tuple[list[Prediction], int, int]:
    """Walk chronologically; predict each event from strictly-prior events only.

    Returns (predictions, favorite_hits, favorite_events). This is the leak-free
    core shared by the scoring report and the calibration experiment — the
    current event's result is folded into the skill pool only AFTER it has been
    predicted.
    """
    config = config or WalkForwardConfig()
    events = [e for e in events if e.completed and len(e.entries) >= config.min_field]
    events = sorted(events, key=lambda e: (e.date, e.event_id))

    prior_records: list[SGRecord] = []
    prior_event_count = 0
    predictions: list[Prediction] = []
    fav_hits = 0
    fav_events = 0

    for ev in events:
        if prior_event_count >= config.min_prior_events:
            skills = skill_from_records(prior_records, cfg=config.skill)
            roster = ev.field_names()
            names, means, sds = to_sim_inputs(roster, skills, config.skill)
            cut = min(config.sim.cut_size, max(2, len(roster) // 2))
            sim_cfg = SimConfig(
                n_sims=config.sim.n_sims, chunk=config.sim.chunk,
                round_corr=config.sim.round_corr,
                course_shock_sd=config.sim.course_shock_sd,
                cut_size=cut, seed=config.sim.seed,
            )
            results = simulate_field(names, means, sds, sim_cfg)
            sims = {r.player: r for r in results}
            actuals = compute_actuals(ev)

            # No-cut events (signature events, Tour Championship: the whole
            # field plays 4 rounds) can't score the make-cut market — everyone
            # "makes the cut" by construction, which would falsely inflate the
            # model's apparent under-confidence. Detect and exclude make_cut for
            # those; win/top-10 are unaffected.
            played_weekend = sum(1 for e in ev.entries if e.rounds_played >= 3)
            no_cut = ev.entries and played_weekend / len(ev.entries) > 0.90

            for r in results:
                a = actuals.get(r.player)
                if a is None:
                    continue
                model = {
                    GolfMarketType.win.value: r.win,
                    GolfMarketType.top_10.value: r.top_10,
                    GolfMarketType.make_cut.value: None if no_cut else r.make_cut,
                }
                if no_cut:
                    a = dict(a)
                    a[GolfMarketType.make_cut.value] = None
                predictions.append(Prediction(
                    event=ev.event, date=ev.date, golfer=r.player,
                    had_prior=r.player in skills, model=model, actual=a,
                ))
            # favorite hit rate: did the model's highest win-prob pick actually win?
            best_pick = max(results, key=lambda r: r.win) if results else None
            if best_pick is not None:
                fav_events += 1
                if actuals.get(best_pick.player, {}).get(GolfMarketType.win.value, 0.0) > 0:
                    fav_hits += 1

        # fold THIS event in AFTER predicting it (no lookahead)
        recs = result_to_sg_proxy(ev, min_field=config.min_field)
        if recs:
            prior_records.extend(recs)
            prior_event_count += 1

    return predictions, fav_hits, fav_events


def run_backtest(
    events: Sequence[GolfResult], config: Optional[WalkForwardConfig] = None
) -> BacktestReport:
    """Walk-forward backtest → scored calibration report."""
    predictions, fav_hits, fav_events = collect_predictions(events, config)
    return _assemble_report(events, predictions, fav_hits, fav_events)


def _assemble_report(events, predictions, fav_hits, fav_events) -> BacktestReport:
    covered = [p for p in predictions if p.had_prior]
    n = len(predictions)
    coverage = (len(covered) / n) if n else 0.0

    brier: dict = {}
    baseline: dict = {}
    calibration: dict = {}
    # per-event base rates for the naive baseline
    for mk in _MARKETS:
        key = mk.value
        preds = [
            (p.model[key], p.actual[key])
            for p in covered
            if p.model.get(key) is not None and p.actual.get(key) is not None
        ]
        if not preds:
            continue
        brier[key] = sum((m - a) ** 2 for m, a in preds) / len(preds)
        base_rate = sum(a for _, a in preds) / len(preds)
        baseline[key] = sum((base_rate - a) ** 2 for _, a in preds) / len(preds)
        bins = []
        for lo, hi in _BINS:
            sel = [(m, a) for m, a in preds if lo <= m < hi]
            if sel:
                bins.append(CalibrationBin(
                    lo, hi, len(sel),
                    sum(m for m, _ in sel) / len(sel),
                    sum(a for _, a in sel) / len(sel),
                ))
            else:
                bins.append(CalibrationBin(lo, hi, 0, 0.0, 0.0))
        calibration[key] = bins

    # winner capture: mean model win prob assigned to the actual winner, per event
    by_event: dict = {}
    for p in covered:
        by_event.setdefault(p.event, []).append(p)
    caps = []
    for ev_name, ps in by_event.items():
        winners = [p for p in ps if p.actual[GolfMarketType.win.value] > 0]
        for w in winners:
            caps.append(w.model[GolfMarketType.win.value])
    winner_capture = sum(caps) / len(caps) if caps else 0.0

    return BacktestReport(
        n_events=len({p.event for p in predictions}),
        n_predictions=n,
        coverage=coverage,
        brier=brier,
        baseline_brier=baseline,
        calibration=calibration,
        winner_capture=winner_capture,
        favorite_hit_rate=(fav_hits / fav_events) if fav_events else 0.0,
        n_favorite_events=fav_events,
    )
