"""Offline tests for the Kalshi fee-schedule watcher (scripts/check_kalshi_fees.py).

These exercise the PURE parse/classify core against a committed text fixture —
no network, no pypdf — so they run in the normal PR CI. The scheduled workflow
supplies the network + pypdf layer.

The load-bearing guard is ``test_snapshot_agrees_with_fees_constants``: it ties
the committed snapshot to the live evmax.fees constants, so the two can never
silently diverge without a red test.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import check_kalshi_fees as cfk  # noqa: E402
from evmax.fees import KALSHI_MAKER_RATE_MULT, KALSHI_TAKER_RATE  # noqa: E402

_FIXTURE = _REPO / "tests" / "fixtures" / "kalshi_fee_schedule_sample.txt"


@pytest.fixture
def fixture_text() -> str:
    return _FIXTURE.read_text()


class TestParseSchedule:
    def test_extracts_taker_and_maker_rates(self, fixture_text):
        parsed = cfk.parse_schedule(fixture_text)
        assert parsed["taker_rate"] == pytest.approx(0.07)
        assert parsed["maker_rate"] == pytest.approx(0.0175)
        assert parsed["maker_mult"] == pytest.approx(0.25)
        assert parsed["effective_date"] == "July 7, 2026"

    def test_extracted_rates_match_fees_constants(self, fixture_text):
        """The document's numbers must equal what the trading math uses."""
        parsed = cfk.parse_schedule(fixture_text)
        assert parsed["taker_rate"] == pytest.approx(KALSHI_TAKER_RATE)
        assert parsed["maker_mult"] == pytest.approx(KALSHI_MAKER_RATE_MULT)

    def test_missing_formula_yields_none(self):
        parsed = cfk.parse_schedule("no fee formula here, just prose about markets")
        assert parsed["taker_rate"] is None
        assert parsed["effective_date"] is None

    def test_taker_not_confused_with_maker(self, fixture_text):
        """Both formulas are present; taker must resolve to 0.07, not 0.0175."""
        parsed = cfk.parse_schedule(fixture_text)
        assert parsed["taker_rate"] != parsed["maker_rate"]


class TestContentHash:
    def test_hash_is_whitespace_insensitive(self):
        a = cfk.content_hash("round up(M x 0.07 x C x P x (1-P))")
        b = cfk.content_hash("round   up(M x 0.07\n\n x C x P x (1-P))   ")
        assert a == b

    def test_hash_changes_on_real_edit(self):
        a = cfk.content_hash("taker multiplier DEFAULT 1")
        b = cfk.content_hash("taker multiplier DEFAULT 2")
        assert a != b


class TestClassify:
    def _snapshot(self, fixture_text):
        cur = cfk.build_current(fixture_text)
        return {"extracted": cur["extracted"], "content_sha256": cur["content_sha256"]}

    def _constants(self):
        return {"taker_rate": KALSHI_TAKER_RATE, "maker_mult": KALSHI_MAKER_RATE_MULT}

    def test_match(self, fixture_text):
        cur = cfk.build_current(fixture_text)
        verdict, reasons = cfk.classify(cur, self._snapshot(fixture_text), self._constants())
        assert verdict == "match"
        assert reasons == []

    def test_hard_drift_on_taker_rate_change(self, fixture_text):
        drifted = fixture_text.replace("0.07", "0.08")
        cur = cfk.build_current(drifted)
        verdict, reasons = cfk.classify(cur, self._snapshot(fixture_text), self._constants())
        assert verdict == "hard_drift"
        assert any("taker_rate" in r for r in reasons)

    def test_hard_drift_when_code_constant_disagrees(self, fixture_text):
        """Even with a matching snapshot, a stale fees.py constant is hard drift."""
        cur = cfk.build_current(fixture_text)
        stale = {"taker_rate": 0.05, "maker_mult": 0.25}
        verdict, reasons = cfk.classify(cur, self._snapshot(fixture_text), stale)
        assert verdict == "hard_drift"

    def test_soft_drift_on_body_change_only(self, fixture_text):
        """Per-series table / wording change with identical rates → soft drift."""
        changed = fixture_text.replace("DEFAULT         1", "DEFAULT         2")
        cur = cfk.build_current(changed)
        verdict, reasons = cfk.classify(cur, self._snapshot(fixture_text), self._constants())
        assert verdict == "soft_drift"
        assert any("content hash" in r or "effective_date" in r for r in reasons)

    def test_soft_drift_on_effective_date(self, fixture_text):
        changed = fixture_text.replace("July  7,  2026", "August  1,  2026")
        cur = cfk.build_current(changed)
        verdict, reasons = cfk.classify(cur, self._snapshot(fixture_text), self._constants())
        assert verdict == "soft_drift"
        assert any("effective_date" in r for r in reasons)

    def test_inconclusive_when_rate_absent(self):
        cur = cfk.build_current("the document format changed and no formula is present")
        verdict, reasons = cfk.classify(cur, None, self._constants())
        assert verdict == "inconclusive"

    def test_no_snapshot_matches_when_rates_ok(self, fixture_text):
        cur = cfk.build_current(fixture_text)
        verdict, _ = cfk.classify(cur, None, self._constants())
        assert verdict == "match"


class TestCommittedSnapshot:
    def test_snapshot_agrees_with_fees_constants(self):
        """The committed snapshot's rates must equal the live fees.py constants.

        This is the guard that fails in PR CI if someone edits evmax/fees.py
        without re-running the watcher's --update, or vice versa.
        """
        snap = json.loads((_REPO / "data" / "kalshi_fee_schedule.snapshot.json").read_text())
        ex = snap["extracted"]
        assert ex["taker_rate"] == pytest.approx(KALSHI_TAKER_RATE)
        assert ex["maker_mult"] == pytest.approx(KALSHI_MAKER_RATE_MULT)
        assert isinstance(snap["content_sha256"], str) and len(snap["content_sha256"]) == 64

    def test_offline_cli_against_snapshot_is_match(self, fixture_text, monkeypatch, tmp_path, capsys):
        """The CLI --offline path classifies the fixture as match when the
        snapshot is derived from the same text (end-to-end wiring smoke test)."""
        # Point the module's snapshot at a temp copy built from the fixture.
        snap_path = tmp_path / "snap.json"
        cur = cfk.build_current(fixture_text)
        snap_path.write_text(json.dumps(
            {"source_url": cur["source_url"], "extracted": cur["extracted"],
             "content_sha256": cur["content_sha256"]}
        ))
        monkeypatch.setattr(cfk, "SNAPSHOT_PATH", snap_path)
        monkeypatch.setattr(cfk, "RESULT_PATH", tmp_path / "result.json")
        rc = cfk.main(["--offline", str(_FIXTURE)])
        assert rc == 0
        assert "unchanged" in capsys.readouterr().out
