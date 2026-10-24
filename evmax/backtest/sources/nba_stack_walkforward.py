"""Point-in-time walk-forward for the NBA stat stack.

The generic ``run_walkforward("nba")`` cannot score ``efficiency``,
``possession_sim``, ``shot_quality`` or ``matchup``. Those agents price from a
single season-to-date snapshot read from ``data/models/*_state.json`` (or fetched
live from stats.nba.com), so a historical replay hits three walls:

1. The snapshot is TODAY's, stamped for the current season. A 2024-25 game fails
   ``season_matches`` and all four agents abstain (0 of 192 December-2024 games
   fired when probed on 2026-10-23).
2. Even when the stamp matches, the snapshot would be end-of-window data, which
   leaks the future into every earlier game.
3. ``possession_sim.predict_pair`` also calls ``state_is_fresh`` (a live ESPN
   probe), and ``matchup`` shifts a Pinnacle probability that history lacks.

This module removes the walls without re-implementing any model. For each game
it builds the agents' own state from stats.nba.com restricted to games BEFORE the
game date (``DateTo``), then calls the production code paths:

* ``EfficiencyModelAgent._fetch_from_nba_api`` / ``predict_pair``
* ``ShotQualityAgent._fetch_shooting_data`` / ``predict_pair``
* ``MatchupAgent._fetch_data`` / ``predict_pair``
* ``PossessionSimAgent.simulate_matchup`` (``predict_pair`` minus the live probe)

Only the ``nba_api`` endpoint classes are wrapped, to inject ``DateTo`` and a
disk cache. Nothing here writes to ``data/models``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import structlog

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "backtest" / "nba_stack_cache"
RESULTS_DIR = REPO_ROOT / "data" / "backtest" / "nba_stack"

# Mirrors SECTOR_WEIGHT_OVERRIDES["nba"] in ensemble_agent.py (also
# NBA_WEIGHT_OVERRIDES in espn_walkforward.py).
LIVE_NBA_WEIGHTS: dict[str, float] = {
    "efficiency": 0.30,
    "possession_sim": 0.30,
    "elo": 0.10,
    "form": 0.10,
    "shot_quality": 0.10,
    "matchup": 0.10,
    "poisson": 0.0,
}
# The generic walk-forward's default blend (what the 0.2199 figure measured).
GENERIC_WEIGHTS: dict[str, float] = {"elo": 0.35, "form": 0.25, "poisson": 0.30}

STACK_MODELS = ("efficiency", "possession_sim", "shot_quality", "matchup")
ENDPOINT_NAMES = ("LeagueDashTeamStats", "LeagueDashTeamShotLocations")

# Opening night per season. ESPN's month feed also returns preseason games
# (live scoring drops season_type == 1); they must neither train elo/form nor be
# scored, so the walk skips everything before these dates.
SEASON_START: dict[str, date] = {
    "2324": date(2023, 10, 24),
    "2425": date(2024, 10, 22),
    "2526": date(2025, 10, 21),
}

# Regular season ends mid-April; the sim has no playoff flag from ESPN's
# scoreboard, so headline numbers cut at this month/day of the end year.
REGULAR_SEASON_END = (4, 15)


# ---------------------------------------------------------------------------
# Snapshot schedule (no-leak rule)
# ---------------------------------------------------------------------------


def snapshot_schedule(
    game_dates: list[date], snapshot_every: int = 3
) -> dict[date, Optional[date]]:
    """Map each game date to the ``as_of`` date of the stats snapshot that prices it.

    The grid is every ``snapshot_every``-th distinct game day. A game on day G
    uses the latest grid day STRICTLY BEFORE G, so a snapshot never contains the
    game it prices or any later game. ``snapshot_every=1`` is the live cadence
    (stats refreshed after every completed game day). The season's first grid
    day has no earlier snapshot, so its games map to ``None`` (the stack abstains
    exactly as it does on opening night).
    """
    if snapshot_every < 1:
        raise ValueError("snapshot_every must be >= 1")
    days = sorted(set(game_dates))
    grid = [d for i, d in enumerate(days) if i % snapshot_every == 0]
    out: dict[date, Optional[date]] = {}
    gi = 0
    latest: Optional[date] = None
    for d in days:
        while gi < len(grid) and grid[gi] < d:
            latest = grid[gi]
            gi += 1
        out[d] = latest
    return out


def season_code_label(code: str) -> str:
    """'2425' -> '2024-25' (nba_api label)."""
    return f"{2000 + int(code[:2])}-{code[2:]}"


def drop_preseason(games: list[dict], code: str) -> list[dict]:
    """Remove exhibition games played before the season's opening night."""
    start = SEASON_START.get(code)
    return [g for g in games if start is None or g["date"] >= start]


def is_regular_season(game_date: date, code: str) -> bool:
    end_year = 2000 + int(code[2:])
    return game_date <= date(end_year, *REGULAR_SEASON_END)


# ---------------------------------------------------------------------------
# Date-bounded, disk-cached nba_api endpoints
# ---------------------------------------------------------------------------


@dataclass
class FetchStats:
    cache_hits: int = 0
    network_calls: int = 0
    failures: int = 0


class _DateBoundEndpoint:
    """Stand-in for an nba_api endpoint: ``get_dict()`` with ``DateTo`` injected."""

    def __init__(
        self,
        cls: Any,
        name: str,
        kwargs: dict,
        as_of: date,
        stats: FetchStats,
        cache_dir: Path,
        offline: bool,
        pause_s: float,
    ) -> None:
        self._cls = cls
        self._name = name
        self._kwargs = dict(kwargs)
        self._as_of = as_of
        self._stats = stats
        self._cache_dir = cache_dir
        self._offline = offline
        self._pause_s = pause_s

    def _cache_path(self) -> Path:
        key = json.dumps(
            {"ep": self._name, "kw": self._kwargs, "as_of": self._as_of.isoformat()},
            sort_keys=True,
            default=str,
        )
        return self._cache_dir / f"{self._name}_{hashlib.sha1(key.encode()).hexdigest()[:20]}.json"

    def get_dict(self) -> dict:
        path = self._cache_path()
        if path.exists():
            self._stats.cache_hits += 1
            return json.loads(path.read_text())
        if self._offline:
            self._stats.failures += 1
            raise RuntimeError(f"offline and no cached payload for {self._name} as_of={self._as_of}")
        kwargs = dict(self._kwargs)
        kwargs["date_to_nullable"] = self._as_of.strftime("%m/%d/%Y")
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                time.sleep(self._pause_s)
                payload = self._cls(**kwargs).get_dict()
                self._stats.network_calls += 1
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))
                return payload
            except Exception as exc:  # noqa: BLE001 — retried, then surfaced
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
        self._stats.failures += 1
        raise RuntimeError(f"{self._name} as_of={self._as_of} failed: {last_exc}")


@contextmanager
def date_bound_nba_api(
    as_of: date,
    stats: FetchStats,
    *,
    cache_dir: Path = CACHE_DIR,
    offline: bool = False,
    pause_s: float = 0.6,
    module: Any = None,
) -> Iterator[None]:
    """Patch the nba_api endpoint classes the NBA agents import lazily.

    The agents do ``from nba_api.stats.endpoints import LeagueDashTeamStats``
    inside their fetch methods, so replacing the module attributes for the
    duration of the ``with`` block reroutes them. ``module`` is injectable for
    tests.
    """
    if module is None:
        import nba_api.stats.endpoints as module  # type: ignore[no-redef]

    originals = {n: getattr(module, n) for n in ENDPOINT_NAMES}

    def make(name: str) -> Callable[..., _DateBoundEndpoint]:
        cls = originals[name]

        def factory(**kwargs: Any) -> _DateBoundEndpoint:
            return _DateBoundEndpoint(cls, name, kwargs, as_of, stats, cache_dir, offline, pause_s)

        return factory

    try:
        for n in ENDPOINT_NAMES:
            setattr(module, n, make(n))
        yield
    finally:
        for n, orig in originals.items():
            setattr(module, n, orig)


# ---------------------------------------------------------------------------
# Production agents loaded with a point-in-time snapshot
# ---------------------------------------------------------------------------


class _FrozenDate(date):
    """``date`` whose ``today()`` is fixed.

    ``PossessionSimAgent.simulate_matchup`` seeds its RNG from
    ``hash(teams + date.today())``. Freezing ``today`` (with PYTHONHASHSEED=0,
    set by the script) makes a rerun on another day reproduce the same sims.
    """

    @classmethod
    def today(cls) -> "_FrozenDate":  # type: ignore[override]
        return cls(2000, 1, 1)


@contextmanager
def frozen_sim_seed() -> Iterator[None]:
    import evmax.agents.models.possession_sim_agent as psa

    original = psa.date
    psa.date = _FrozenDate  # type: ignore[misc]
    try:
        yield
    finally:
        psa.date = original  # type: ignore[misc]


def _noop() -> None:
    return None


async def _return(value: Any) -> Any:
    return value


@dataclass
class StackSnapshot:
    as_of: date
    efficiency: Any = None
    possession_sim: Any = None
    shot_quality: Any = None
    matchup: Any = None
    efficiency_state: dict = field(default_factory=dict)


async def build_snapshot(as_of: date, season: str) -> StackSnapshot:
    """Fetch each agent's inputs as of ``as_of`` through its own fetch code.

    An agent whose fetch raises (opening weeks: no team has a game yet) is left
    as ``None`` and abstains, matching live behaviour. Must run inside
    ``date_bound_nba_api``.
    """
    from evmax.agents.models.efficiency_agent import EfficiencyModelAgent
    from evmax.agents.models.matchup_agent import MatchupAgent
    from evmax.agents.models.possession_sim_agent import PossessionSimAgent
    from evmax.agents.models.shot_quality_agent import ShotQualityAgent

    snap = StackSnapshot(as_of=as_of)

    eff = EfficiencyModelAgent()
    eff.save_state = _noop  # never touch data/models
    try:
        state = await eff._fetch_from_nba_api(season)
    except Exception as exc:  # noqa: BLE001
        logger.info("nba_stack_efficiency_unavailable", as_of=as_of.isoformat(), error=str(exc))
        state = {}
    if state.get("teams"):
        eff._state["nba"] = state
        eff._ensure_stats = lambda sector, _s=state: _return(_s)  # type: ignore[method-assign]
        snap.efficiency = eff
        snap.efficiency_state = state
        sim = PossessionSimAgent()
        sim._load_efficiency_state = lambda _s=state: _s  # type: ignore[method-assign]
        snap.possession_sim = sim

    sq = ShotQualityAgent()
    sq.save_state = _noop
    try:
        await sq._fetch_shooting_data(season)
        sq._ensure_data = lambda: _return(None)  # type: ignore[method-assign]
        snap.shot_quality = sq
    except Exception as exc:  # noqa: BLE001
        logger.info("nba_stack_shot_quality_unavailable", as_of=as_of.isoformat(), error=str(exc))

    mu = MatchupAgent()
    mu.save_state = _noop
    try:
        await mu._fetch_data(season)
        mu._ensure_data = lambda: _return(None)  # type: ignore[method-assign]
        snap.matchup = mu
    except Exception as exc:  # noqa: BLE001
        logger.info("nba_stack_matchup_unavailable", as_of=as_of.isoformat(), error=str(exc))

    return snap


def _market(home: str, away: str, game_date: date, base: float):
    from evmax.models.market import MarketSource, PredictionMarket
    from evmax.models.odds import SharpBook, SharpOdds

    eid = f"nbastack_{game_date.isoformat()}_{home}_{away}"
    market = PredictionMarket(
        id=eid, market_id=eid, event_id=eid, sector="nba",
        team_home=home, team_away=away,
        # Noon UTC, the same anchor Kalshi tickers use, so the agents'
        # season_matches() sees the GAME's season, not today's.
        event_date=datetime(game_date.year, game_date.month, game_date.day, 12),
        source=MarketSource.kalshi, yes_price=base, no_price=1.0 - base,
    )
    sharp = SharpOdds(
        event_id=eid, book=SharpBook.pinnacle, sector="nba",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1.0 / base, outcome_b_decimal=1.0 / (1.0 - base),
        true_prob_a=base, true_prob_b=1.0 - base,
    )
    return market, sharp


def _prob(pred: Any) -> Optional[float]:
    return None if pred is None else float(pred.true_prob_a)


def weighted_blend(probs: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    """Weight-normalised mean of the models that fired (weight 0 or None skips)."""
    tw = tp = 0.0
    for name, w in weights.items():
        p = probs.get(name)
        if p is not None and w > 0:
            tw += w
            tp += w * p
    return tp / tw if tw else None


async def predict_stack(
    snap: Optional[StackSnapshot],
    home: str,
    away: str,
    game_date: date,
    other_probs: dict[str, Optional[float]],
) -> dict[str, Optional[float]]:
    """Run the four production agents for one game against ``snap``.

    ``matchup`` is a nudge on the sharp probability (``base + shift``). History
    has no Pinnacle price, so ``base`` is the live-weighted blend of every other
    model that fired (0.5 when none did). The returned ``matchup`` is
    ``base + shift`` and ``matchup_shift`` is the shift alone.
    """
    out: dict[str, Optional[float]] = {m: None for m in STACK_MODELS}
    out["matchup_shift"] = None
    if snap is None:
        return out

    market, sharp = _market(home, away, game_date, 0.55)

    if snap.efficiency is not None:
        out["efficiency"] = _prob(await snap.efficiency.predict_pair(market, sharp))
    if snap.possession_sim is not None:
        sim = snap.possession_sim.simulate_matchup(
            home, away, event_id=sharp.event_id, game_date=market.event_date,
        )
        if sim is not None:
            out["possession_sim"] = float(sim["prob_a"])
    if snap.shot_quality is not None:
        out["shot_quality"] = _prob(await snap.shot_quality.predict_pair(market, sharp))

    if snap.matchup is not None:
        known = {**other_probs, **{k: out[k] for k in STACK_MODELS if k != "matchup"}}
        base = weighted_blend(known, LIVE_NBA_WEIGHTS)
        base = 0.5 if base is None else min(0.95, max(0.05, base))
        market_m, sharp_m = _market(home, away, game_date, base)
        pm = _prob(await snap.matchup.predict_pair(market_m, sharp_m))
        if pm is not None:
            out["matchup"] = pm
            out["matchup_shift"] = pm - base
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    date: str
    season: str
    home: str
    away: str
    home_won: int
    regular_season: bool
    probs: dict[str, Optional[float]]
    blends: dict[str, Optional[float]]


def compute_blends(probs: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    return {
        "generic": weighted_blend(probs, GENERIC_WEIGHTS),
        # What the live NBA blend reduces to when the stack abstains.
        "live_no_stack": weighted_blend(
            probs, {k: LIVE_NBA_WEIGHTS[k] for k in ("elo", "form")}
        ),
        "live_full": weighted_blend(probs, LIVE_NBA_WEIGHTS),
    }


def _brier(p: float, y: int) -> float:
    return (p - y) ** 2


def summarize(
    records: list[GameRecord],
    *,
    regular_only: bool = True,
    score_seasons: Optional[list[str]] = None,
) -> dict:
    """Per-model and blend Brier plus paired full-vs-no-stack deltas."""
    rs = [
        r for r in records
        if (not regular_only or r.regular_season)
        and (score_seasons is None or r.season in score_seasons)
    ]
    summary: dict[str, Any] = {"n_games": len(rs), "models": {}, "blends": {}}

    def stats(values: list[tuple[float, int]]) -> dict:
        if not values:
            return {"n": 0, "brier": None, "accuracy": None}
        b = sum(_brier(p, y) for p, y in values) / len(values)
        acc = sum((p > 0.5) == bool(y) for p, y in values) / len(values)
        return {"n": len(values), "brier": round(b, 5), "accuracy": round(acc, 4)}

    for m in ("elo", "form", "poisson") + STACK_MODELS:
        summary["models"][m] = stats([(r.probs[m], r.home_won) for r in rs if r.probs.get(m) is not None])
    home_rate = sum(r.home_won for r in rs) / len(rs) if rs else 0.0
    summary["home_win_rate"] = round(home_rate, 4)
    summary["always_home_brier"] = (
        round(sum(_brier(home_rate, r.home_won) for r in rs) / len(rs), 5) if rs else None
    )
    for b in ("generic", "live_no_stack", "live_full"):
        summary["blends"][b] = stats([(r.blends[b], r.home_won) for r in rs if r.blends.get(b) is not None])

    summary["paired"] = {
        "all_games": paired_delta(rs, "live_full", "live_no_stack"),
        "stack_all_fired": paired_delta(
            [r for r in rs if all(r.probs.get(m) is not None for m in STACK_MODELS[:3])],
            "live_full", "live_no_stack",
        ),
    }
    summary["coverage"] = {
        m: round(sum(r.probs.get(m) is not None for r in rs) / len(rs), 4) if rs else 0.0
        for m in STACK_MODELS
    }
    return summary


def paired_delta(records: list[GameRecord], a: str, b: str) -> dict:
    """Brier(a) - Brier(b) per 1000 over games where both blends exist.

    Negative = ``a`` is better. ``z`` is the mean over its standard error (games
    treated as independent).
    """
    diffs = [
        _brier(r.blends[a], r.home_won) - _brier(r.blends[b], r.home_won)
        for r in records
        if r.blends.get(a) is not None and r.blends.get(b) is not None
    ]
    n = len(diffs)
    if n < 2:
        return {"n": n, "delta_per_1000": None, "z": None}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return {
        "n": n,
        "delta_per_1000": round(mean * 1000, 3),
        "z": round(mean / se, 2) if se > 0 else None,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _regress_elo(elo: Any, keep: float = 0.5) -> None:
    """Offseason regression toward 1500, as scripts/offseason_regress.py applies live."""
    from evmax.agents.models.elo_agent import DEFAULT_ELO

    ratings = elo._state.get("nba", {}).get("ratings", {})
    for k, r in list(ratings.items()):
        ratings[k] = DEFAULT_ELO + keep * (r - DEFAULT_ELO)


def run_nba_stack_walkforward(
    warmup_seasons: list[str],
    score_seasons: list[str],
    *,
    snapshot_every: int = 3,
    offline: bool = False,
    pause_s: float = 0.6,
    cache_dir: Path = CACHE_DIR,
    elo_keep: float = 0.5,
) -> tuple[list[GameRecord], FetchStats]:
    """Walk NBA games chronologically, scoring only ``score_seasons``.

    ``warmup_seasons`` train elo/form/poisson (no stack fetch, no scoring). The
    stack is evaluated only in ``score_seasons``.
    """
    from evmax.agents.models.elo_agent import EloModelAgent
    from evmax.agents.models.form_agent import FormModelAgent
    from evmax.agents.models.poisson_agent import PoissonModelAgent
    from evmax.matching.normalizer import NameNormalizer

    elo = EloModelAgent(); elo._state = {}
    form = FormModelAgent(); form._state = {}
    poisson = PoissonModelAgent(); poisson._state = {}
    norm = NameNormalizer("nba")
    fstats = FetchStats()
    records: list[GameRecord] = []
    with frozen_sim_seed():
        _walk(
            warmup_seasons, score_seasons, snapshot_every, offline, pause_s, cache_dir,
            elo_keep, elo, form, poisson, norm, fstats, records,
        )
    return records, fstats


def _walk(
    warmup_seasons, score_seasons, snapshot_every, offline, pause_s, cache_dir,
    elo_keep, elo, form, poisson, norm, fstats, records,
) -> None:
    from evmax.backtest.engine import WALKFORWARD_MONTHS
    from evmax.backtest.sources.espn_walkforward import (
        _elo_predict, _elo_update, _form_predict, _form_update,
        _poisson_predict, _poisson_update, fetch_espn_games,
    )

    for idx, code in enumerate(warmup_seasons + score_seasons):
        scoring = code in score_seasons
        months = WALKFORWARD_MONTHS[code]["nba"]
        games = drop_preseason(fetch_espn_games("nba", months), code)
        if idx > 0:
            _regress_elo(elo, elo_keep)
        schedule = snapshot_schedule([g["date"] for g in games], snapshot_every) if scoring else {}
        label = season_code_label(code)
        snapshots: dict[date, Optional[StackSnapshot]] = {}

        for g in games:
            home, away = norm.normalize(g["home"]), norm.normalize(g["away"])
            if not home or not away:
                continue
            gd: date = g["date"]
            home_won = g["home_score"] > g["away_score"]

            if scoring:
                probs: dict[str, Optional[float]] = {
                    "elo": _elo_predict(elo, "nba", home, away),
                    "form": _form_predict(form, "nba", home, away, gd),
                    "poisson": _poisson_predict(poisson, "nba", home, away),
                }
                as_of = schedule.get(gd)
                snap: Optional[StackSnapshot] = None
                if as_of is not None:
                    if as_of not in snapshots:
                        with date_bound_nba_api(
                            as_of, fstats, cache_dir=cache_dir, offline=offline, pause_s=pause_s,
                        ):
                            snapshots[as_of] = asyncio.run(build_snapshot(as_of, label))
                    snap = snapshots[as_of]
                probs.update(asyncio.run(predict_stack(snap, home, away, gd, probs)))
                records.append(GameRecord(
                    date=gd.isoformat(), season=code, home=home, away=away,
                    home_won=int(home_won), regular_season=is_regular_season(gd, code),
                    probs=probs, blends=compute_blends(probs),
                ))

            _elo_update(elo, "nba", home, away, home_won, gd)
            _form_update(form, "nba", home, away, home_won, gd, g["home_score"], g["away_score"])
            _poisson_update(poisson, "nba", home, away, g["home_score"], g["away_score"])


def save_records(records: list[GameRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.__dict__ for r in records]))


def load_records(path: Path) -> list[GameRecord]:
    return [GameRecord(**d) for d in json.loads(path.read_text())]
