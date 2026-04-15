"""Sackmann / tennis-data.co.uk xlsx replay validation for MODEL-1.

Iterates every row in ``data/backtest/tennis/{2024,2025}.xlsx``, feeds a
synthesized Kalshi-shaped ``"{ATP|WTA} {Tournament}"`` competition string
through ``TennisModelAgent._resolve_surface``, and asserts the returned
surface matches the xlsx's ``Surface`` column in at least 95% of rows.

**This is designed as a coverage-expansion loop, not a one-shot gate.**
The first run after any dict change may miss small-tour tournaments. When
it fails, the printed miss table drives iteration: the implementer adds
the missing cities to ``CITY_TO_SURFACE`` in
``evmax/agents/models/tennis_model_agent.py`` and re-runs until accuracy
clears the 95% floor. Once stable, this is a regression gate.

Indoor-hard tournaments are NOT validated here: the xlsx tracks
``Surface`` and ``Court`` as orthogonal columns, and all indoor-hard
events are labelled Surface=Hard (Court=Indoor). Our resolver correctly
maps these to surface=hard and is_indoor=True. The ``is_indoor`` seam is
covered separately in ``tests/test_tennis_kalshi_fixtures.py``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from evmax.agents.models.tennis_model_agent import TennisModelAgent

XLSX_DIR = Path(__file__).parent.parent / "data" / "backtest" / "tennis"
ACCURACY_FLOOR = 0.95


def _load_rows(year: int):
    """Yield (tournament, expected_surface) pairs from the xlsx for a given year."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    path = XLSX_DIR / f"{year}.xlsx"
    if not path.exists():
        pytest.skip(f"{path} not found")

    df = pd.read_excel(path)
    if "Tournament" not in df.columns or "Surface" not in df.columns:
        pytest.skip(f"{path} missing required columns")

    for _, row in df.iterrows():
        tournament = str(row["Tournament"]).strip()
        surface = str(row["Surface"]).strip().lower()
        if not tournament or not surface or surface == "nan":
            continue
        if surface not in {"hard", "clay", "grass"}:
            # Carpet or other legacy values — skip
            continue
        yield tournament, surface


def _run_replay(year: int) -> tuple[int, int, dict]:
    """Replay all rows for a year. Returns (correct, total, miss_summary)."""
    correct = 0
    total = 0
    miss_tournaments: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "expected": None, "got": Counter()}
    )

    for tournament, expected in _load_rows(year):
        # Synthesize the Kalshi competition format. xlsx rows don't carry
        # the tour, so we try both ATP and WTA prefixes — the resolver
        # strips them anyway. ATP is arbitrary here; both produce the
        # same result.
        competition = f"ATP {tournament}"
        got, _ = TennisModelAgent._resolve_surface(competition=competition)

        total += 1
        if got == expected:
            correct += 1
        else:
            miss_tournaments[tournament]["count"] += 1
            miss_tournaments[tournament]["expected"] = expected
            miss_tournaments[tournament]["got"][got] += 1

    return correct, total, miss_tournaments


def _format_miss_report(year: int, miss_summary: dict, total: int, correct: int) -> str:
    """Format a human-readable miss report to drive CITY_TO_SURFACE expansion."""
    if not miss_summary:
        return f"{year}: {correct}/{total} (100.0%) — no misses"

    lines = [
        f"{year}: {correct}/{total} ({100*correct/total:.1f}%) — "
        f"{len(miss_summary)} unique tournaments missed",
        "",
        "Misses grouped by tournament (add missing cities to CITY_TO_SURFACE):",
    ]
    for tournament, info in sorted(
        miss_summary.items(), key=lambda kv: -kv[1]["count"]
    ):
        got_str = ", ".join(f"{g}×{n}" for g, n in info["got"].most_common())
        lines.append(
            f"  {tournament!r:<45} "
            f"expected={info['expected']:<5} "
            f"got=[{got_str}] "
            f"({info['count']} matches)"
        )
    return "\n".join(lines)


class TestSackmannReplay:
    """Coverage-expansion gate: ≥95% accuracy against historical xlsx."""

    def test_2025_accuracy(self, capsys):
        correct, total, misses = _run_replay(2025)
        report = _format_miss_report(2025, misses, total, correct)
        print(report)  # always visible with -s or on failure
        accuracy = correct / total if total else 0.0
        assert accuracy >= ACCURACY_FLOOR, (
            f"2025 accuracy {accuracy:.3f} < floor {ACCURACY_FLOOR}\n\n{report}"
        )

    def test_2024_accuracy(self, capsys):
        correct, total, misses = _run_replay(2024)
        report = _format_miss_report(2024, misses, total, correct)
        print(report)
        accuracy = correct / total if total else 0.0
        assert accuracy >= ACCURACY_FLOOR, (
            f"2024 accuracy {accuracy:.3f} < floor {ACCURACY_FLOOR}\n\n{report}"
        )
