"""Automatic per-sector devig-method selection.

The weekly integrity sweep runs :func:`recommend_devig_methods`, which takes the
closing pre-tip Pinnacle game-winner line of every recently-resolved game,
re-devigs it three ways (power / shin / multiplicative) and scores each against
who actually won (:func:`collect_devig_observations`). When a non-power
method beats power by a SIGNIFICANT and material margin on an adequate sample,
the sweep surfaces a one-line recommendation with the exact one-tap command:

    evmax cleanup devig promote <sector>

which writes the choice to ``devig_method_state.json`` (read by
``resolve_devig_method``) — no code edit, no commit. So the operator never
investigates on their own bandwidth; the system flags a winner and applying it
is a single confirmation.

Why Brier-vs-outcome, not CLV: devigging EXTRACTS the sharp book's own implied
probability from its vigged odds. Its quality is how well that extracted
probability predicts the outcome — a calibration question whose ground truth is
the result. That is distinct from MODEL promotion, which asks whether a forecast
beats the market and is judged on CLV (the tennis lesson). The tennis lesson
still applies as the SIGNIFICANCE guard here: a marginal Brier edge below the
noise floor never triggers a flip — the gate requires a paired z as well as a
material delta and a large n. Power stays sticky; ties go to power.

The sample is ONE observation per resolved game, on game-winner markets only.
Until 2026-10-07 the recommender scored every archived snapshot of every record
that shared an event_id with a resolved market. That produced a false NHL
"multiplicative" recommendation (z=32.0, n=1783 snapshots from 47 events):

* 1753 of the 1783 rows were ``::spread`` snapshots. The archived ``::spread``
  record is Pinnacle's MAIN line. Its rung and side change between snapshots
  (TOR −1.5 → MTL −1.5). The ev_outcomes rows that share its event_id settle
  other rungs and sides (MTL +1.5). The pairing scored a mean side-A
  probability of 0.355 against a side-A "win" rate of 0.795. Multiplicative
  gives the longshot side the most probability, so it "won".
* ~38 snapshots per event were counted as independent lines. That inflated n
  and the paired z by roughly the square root of the snapshot count.
* In-play snapshots (Pinnacle quotes during the game) were scored as well.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from evmax.ev.devig import (
    DEVIG_METHOD_STATE_PATH,
    DEVIG_METHODS,
    devig,
    invalidate_selected_methods_cache,
    resolve_devig_method,
)

# Gate constants — a challenger must clear ALL THREE to be recommended.
DEVIG_SELECT_MIN_N = 200          # resolved GAMES for the sector in the window
DEVIG_SELECT_MIN_BRIER_DELTA = 0.002   # >= 2/1000 mean paired Brier improvement
DEVIG_SELECT_MIN_Z = 1.64         # one-sided 95% on the paired per-game diff

# A game-winner event_id is ``sector::date::teams``. Every other record carries a
# market suffix (``::spread``, ``::total::<line>``, ``::advance``, ``::prop::…``)
# whose settlement is NOT "who won the game", so it never enters the A/B.
_GAME_WINNER_ID_PARTS = 3
_DRAW_LABELS = frozenset({"draw", "tie", "x"})


@dataclass
class DevigRecommendation:
    sector: str
    current_method: str      # what resolve_devig_method uses today
    best_method: str         # best-Brier method over the window
    power_brier: float
    best_brier: float
    delta: float             # power_brier - best_brier (>0 = challenger better)
    z: float                 # paired z of the per-game squared-error diff
    n: int
    clears_gate: bool        # best is a non-power method AND clears all gates

    @property
    def is_actionable(self) -> bool:
        """A real, un-applied recommendation the sweep should surface."""
        return self.clears_gate and self.best_method != self.current_method


def evaluate_sector(
    sector: str,
    pairs_by_method: dict[str, list[tuple[float, int]]],
    current_method: Optional[str] = None,
) -> Optional[DevigRecommendation]:
    """Gate one sector from per-method (prob, outcome) pairs (one per game).

    ``pairs_by_method`` must carry the SAME games under every method, in the
    same order — the paired significance test differences them per game. Returns
    None when power isn't present or samples are empty; otherwise a
    DevigRecommendation whose ``clears_gate`` reflects the three-part gate.
    """
    power = pairs_by_method.get("power")
    if not power:
        return None
    n = len(power)
    if current_method is None:
        current_method = resolve_devig_method(sector)

    def _brier(pairs: list[tuple[float, int]]) -> float:
        return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else float("nan")

    power_brier = _brier(power)

    # Best non-power challenger by Brier.
    best_method, best_brier, best_pairs = "power", power_brier, power
    for m in DEVIG_METHODS:
        if m == "power":
            continue
        pairs = pairs_by_method.get(m)
        if not pairs or len(pairs) != n:
            continue
        b = _brier(pairs)
        if b < best_brier:
            best_method, best_brier, best_pairs = m, b, pairs

    delta = power_brier - best_brier

    # Paired significance of the per-game squared-error improvement.
    z = 0.0
    if best_method != "power" and n >= 2:
        diffs = [
            (pp - po) ** 2 - (cp - co) ** 2
            for (pp, po), (cp, co) in zip(power, best_pairs)
        ]
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        z = mean / se if se > 0 else 0.0

    clears = (
        best_method != "power"
        and n >= DEVIG_SELECT_MIN_N
        and delta >= DEVIG_SELECT_MIN_BRIER_DELTA
        and z >= DEVIG_SELECT_MIN_Z
    )
    return DevigRecommendation(
        sector=sector,
        current_method=current_method,
        best_method=best_method,
        power_brier=power_brier,
        best_brier=best_brier,
        delta=delta,
        z=z,
        n=n,
        clears_gate=clears,
    )


def save_selected_method(sector: str, method: str) -> None:
    """Persist the one-tap selection and refresh the runtime cache."""
    m = method.lower()
    if m not in DEVIG_METHODS:
        raise ValueError(f"unknown devig method: {method!r} (choices: {DEVIG_METHODS})")
    DEVIG_METHOD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(DEVIG_METHOD_STATE_PATH.read_text())
        methods = state.get("methods") if isinstance(state, dict) else None
        methods = dict(methods) if isinstance(methods, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        methods = {}
    if m == "power":
        methods.pop(sector.lower(), None)  # power is the default — store nothing
    else:
        methods[sector.lower()] = m
    DEVIG_METHOD_STATE_PATH.write_text(json.dumps({"methods": methods}, indent=2, sort_keys=True))
    invalidate_selected_methods_cache()


def clear_selected_method(sector: str) -> None:
    """Revert a sector to power (removes its persisted entry)."""
    save_selected_method(sector, "power")


@dataclass(frozen=True)
class DevigObservation:
    """One resolved game: its closing Pinnacle game-winner line, devigged every way."""

    sector: str
    event_id: str
    three_way: bool          # the line carried a draw price (soccer / worldcup)
    contract: str            # scored outcome: "a" / "b" (Pinnacle's sides) or "draw"
    won: int                 # 1 = the scored outcome happened
    probs: dict[str, float]  # devig method -> devigged probability of ``contract``


# Index of each contract in ``devig([a, b, draw]).true_probs``.
_CONTRACT_INDEX = {"a": 0, "b": 1, "draw": 2}


def is_game_winner_event_id(event_id: Optional[str]) -> bool:
    """True for a moneyline / 3-way event_id (``sector::date::teams``)."""
    return bool(event_id) and len(event_id.split("::")) == _GAME_WINNER_ID_PARTS


def scored_contract(
    results: Iterable[tuple[Optional[str], int]],
    a_label: Optional[str],
    b_label: Optional[str],
    *,
    three_way: bool,
    normalizer=None,
) -> Optional[tuple[str, int]]:
    """Pick the one contract to score for a game, and whether it happened.

    ``results`` holds ``(yes_team, outcome)`` for each resolved market that
    shares the game's event_id (both teams, both venues, the TIE market). Each
    row maps to the contract it settles: Pinnacle side ``"a"``, side ``"b"``,
    or ``"draw"`` (3-way lines only). Team labels map through the strict
    closed-world rule shared with outcome resolution and close alignment
    (``alignment.resolve_team``). A label that maps to neither side, or to
    both, is ignored.

    The scored contract is the first of a → b → draw that a market settles.
    The choice depends only on WHICH markets were logged (a pre-game scan
    decision), never on the result. Deriving side A's result from other rows
    would not be outcome-neutral on a 3-way line: "B won" or "draw won" proves
    "A lost", but "B lost" leaves "A won" and "draw" open, so A's wins would be
    under-sampled. On a 2-way line scoring "b" has the same squared error as
    scoring "a" with the result flipped.

    Returns ``(contract, won)``. Returns None when no row maps to a contract or
    when the rows contradict each other (one contract both won and lost, two
    winners, or every contract lost).
    """
    from evmax.matching.alignment import YesOutcome, resolve_team

    seen: dict[str, set[int]] = {}
    for yes_team, outcome in results:
        name = (yes_team or "").strip().lower()
        if not name or outcome is None:
            continue
        if name in _DRAW_LABELS:
            if not three_way:
                continue  # no draw price to score on a 2-way line
            contract = "draw"
        else:
            side = resolve_team(name, a_label, b_label, normalizer, allow_codes=True)
            if side is None:
                continue
            contract = "a" if side[0] is YesOutcome.A else "b"
        seen.setdefault(contract, set()).add(int(outcome))

    if any(len(v) > 1 for v in seen.values()):
        return None
    winners = [c for c, v in seen.items() if 1 in v]
    n_contracts = 3 if three_way else 2
    if len(winners) > 1 or (not winners and len(seen) == n_contracts):
        return None
    for contract in ("a", "b", "draw"):
        if contract in seen:
            return contract, next(iter(seen[contract]))
    return None


def _sector_normalizer(sector: str):
    """The sector's NameNormalizer (alias maps), or None when unavailable."""
    try:
        from evmax.matching.normalizer import NameNormalizer

        return NameNormalizer(sector)
    except Exception:  # noqa: BLE001 — the A/B must never die on a registry quirk
        return None


# The closing line: the last snapshot strictly before the scheduled start.
# Pinnacle keeps quoting in-play, and in-play prices trend to 0/1. Game-winner
# records only — the spread/total line columns stay NULL on them.
_CLOSE_SQL = """
    SELECT sector, outcome_a_label, outcome_b_label,
           outcome_a_decimal, outcome_b_decimal, outcome_draw_decimal
    FROM archived_sharp_odds
    WHERE event_id = ?
      AND spread_line IS NULL
      AND total_line IS NULL
      AND outcome_a_decimal IS NOT NULL
      AND outcome_b_decimal IS NOT NULL
      AND event_date IS NOT NULL
      AND fetched_at < event_date
    ORDER BY fetched_at DESC
    LIMIT 1
"""

_OUTCOMES_SQL = """
    SELECT event_id, yes_team, outcome
    FROM ev_outcomes
    WHERE outcome IS NOT NULL AND event_id IS NOT NULL
      AND resolved_at >= datetime('now', ?)
"""


def collect_devig_observations(
    days: int = 120,
    *,
    pred_conn=None,
    archive_path: Optional[Path] = None,
) -> list[DevigObservation]:
    """One observation per resolved game: its closing line, devigged every way.

    1. Read the resolved ev_outcomes rows from the last ``days`` days. Keep only
       game-winner event_ids and group the rows by event_id. Many markets
       (Kalshi + PolyUS, both teams, TIE) settle the same game.
    2. For each game, read the closing pre-tip game-winner snapshot from
       archive.db (:data:`_CLOSE_SQL`).
    3. Pick the contract to score and its result from the game's rows
       (:func:`scored_contract`). Skip a game with no mappable row or with
       contradictory rows.
    4. Devig the closing decimals with every method. Skip the game if any
       method fails, so every method scores the same games.

    ``pred_conn`` (an open predictions.db connection) and ``archive_path``
    exist for tests; by default the prod predictions.db and
    ``evmax.archiver.DB_PATH`` are read. archive.db is opened read-only.
    Degrades to ``[]`` off the prod box (no archive.db / no rows).
    """
    import sqlite3
    from collections import defaultdict

    params = (f"-{days} days",)
    try:
        if pred_conn is not None:
            outcome_rows = pred_conn.execute(_OUTCOMES_SQL, params).fetchall()
        else:
            from evmax.agents.cleanup.db import get_connection

            with get_connection() as conn:
                outcome_rows = conn.execute(_OUTCOMES_SQL, params).fetchall()
    except Exception:  # noqa: BLE001
        return []

    results_by_event: dict[str, list[tuple[Optional[str], int]]] = defaultdict(list)
    for event_id, yes_team, outcome in outcome_rows:
        if is_game_winner_event_id(event_id):
            results_by_event[event_id].append((yes_team, int(outcome)))
    if not results_by_event:
        return []

    if archive_path is None:
        try:
            from evmax.archiver import DB_PATH as archive_path
        except Exception:  # noqa: BLE001
            return []
    archive_path = Path(archive_path).resolve()
    if not archive_path.exists():
        return []

    normalizers: dict[str, object] = {}
    observations: list[DevigObservation] = []
    arc = sqlite3.connect(f"{archive_path.as_uri()}?mode=ro", uri=True)
    try:
        for event_id, results in results_by_event.items():
            row = arc.execute(_CLOSE_SQL, (event_id,)).fetchone()
            if row is None:
                continue
            sector_col, a_label, b_label, a_dec, b_dec, draw_dec = row
            sector = (sector_col or event_id.split("::", 1)[0]).lower()
            three_way = bool(draw_dec)
            if sector not in normalizers:
                normalizers[sector] = _sector_normalizer(sector)
            scored = scored_contract(
                results, a_label, b_label,
                three_way=three_way, normalizer=normalizers[sector],
            )
            if scored is None:
                continue
            contract, won = scored
            idx = _CONTRACT_INDEX[contract]
            decimals = [a_dec, b_dec, draw_dec] if three_way else [a_dec, b_dec]
            try:
                probs = {
                    m: float(devig(decimals, method=m).true_probs[idx]) for m in DEVIG_METHODS
                }
            except Exception:  # noqa: BLE001
                continue
            observations.append(DevigObservation(
                sector=sector,
                event_id=event_id,
                three_way=three_way,
                contract=contract,
                won=won,
                probs=probs,
            ))
    finally:
        arc.close()
    return observations


def recommend_devig_methods(
    days: int = 120,
    *,
    pred_conn=None,
    archive_path: Optional[Path] = None,
) -> list[DevigRecommendation]:
    """Prod-only: gate each sector on its resolved games' closing lines.

    The sample is :func:`collect_devig_observations` — one closing pre-tip
    game-winner line per resolved game — so ``n`` counts games. Returns a
    recommendation per sector. Degrades to ``[]`` off the prod box.
    ``scripts/backtest_devig_ab.py`` reads the same observations.
    """
    from collections import defaultdict

    pairs: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    for obs in collect_devig_observations(days, pred_conn=pred_conn, archive_path=archive_path):
        for method, prob in obs.probs.items():
            pairs[obs.sector][method].append((prob, obs.won))

    recs: list[DevigRecommendation] = []
    for sec, by_method in pairs.items():
        rec = evaluate_sector(sec, by_method)
        if rec is not None:
            recs.append(rec)
    return recs
