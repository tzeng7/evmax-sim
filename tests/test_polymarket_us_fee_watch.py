"""Offline tests for the Polymarket US fee watcher (scripts/check_polymarket_us_fees.py).

The pure parse/classify core runs against a committed copy of the docs page, so
no network is needed. ``test_snapshot_agrees_with_fees_constants`` and
``test_fixture_table_reproduced_by_order_fee`` tie the snapshot/page to the live
evmax.fees constants: the 0.06 -> 0.0695 taker change (2026-10-01) would have
tripped both.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import check_polymarket_us_fees as cpf  # noqa: E402
from evmax.fees import (  # noqa: E402
    POLYMARKET_US_MAKER_THETA,
    POLYMARKET_US_TAKER_THETA,
    polymarket_us_order_fee,
)

_FIXTURE = _REPO / "tests" / "fixtures" / "polymarket_us_fee_schedule_sample.txt"


@pytest.fixture
def text() -> str:
    return _FIXTURE.read_text()


def _constants() -> dict:
    return cpf.get_constants()


def _snapshot(text: str) -> dict:
    cur = cpf.build_current(text)
    return {"extracted": cur["extracted"], "content_sha256": cur["content_sha256"]}


class TestParse:
    def test_extracts_thetas_date_and_rounding(self, text):
        ex = cpf.parse_schedule(text)
        assert ex["taker_theta"] == pytest.approx(0.0695)
        assert ex["maker_theta"] == pytest.approx(-0.0125)
        assert "October 1, 2026" in ex["effective"]
        assert ex["bankers_rounding"] is True

    def test_extracts_fee_by_price_table(self, text):
        table = cpf.parse_schedule(text)["fee_table"]
        assert len(table) >= 90
        assert table["0.50"] == [1.74, 0.31]

    def test_missing_page_yields_none(self):
        ex = cpf.parse_schedule("nothing about fees here")
        assert ex["taker_theta"] is None and ex["maker_theta"] is None
        assert ex["fee_table"] == {}


class TestClassify:
    def test_match(self, text):
        cur = cpf.build_current(text)
        assert cpf.classify(cur, _snapshot(text), _constants()) == ("match", [])

    def test_hard_drift_when_docs_theta_changes(self, text):
        cur = cpf.build_current(text.replace("0.0695", "0.0750"))
        verdict, reasons = cpf.classify(cur, _snapshot(text), _constants())
        assert verdict == "hard_drift"
        assert any("taker_theta" in r for r in reasons)

    def test_hard_drift_when_fees_py_is_stale(self, text):
        """The exact 2026-10-01 failure: docs say 0.0695, code still 0.06."""
        cur = cpf.build_current(text)
        stale = {**_constants(), "taker_theta": 0.06}
        verdict, reasons = cpf.classify(cur, _snapshot(text), stale)
        assert verdict == "hard_drift"
        assert any("POLYMARKET_US_TAKER_THETA" in r for r in reasons)

    def test_hard_drift_when_fee_table_disagrees_with_order_fee(self, text):
        cur = cpf.build_current(text)
        consts = {**_constants(), "order_fee": lambda p, c, maker=False: 0.0}
        verdict, reasons = cpf.classify(cur, None, consts)
        assert verdict == "hard_drift"
        assert any("@ $0.50" in r for r in reasons)

    def test_hard_drift_when_rounding_rule_changes(self, text):
        cur = cpf.build_current(text.replace("banker's rounding", "ceiling rounding"))
        verdict, reasons = cpf.classify(cur, None, _constants())
        assert verdict == "hard_drift"
        assert any("banker" in r for r in reasons)

    def test_soft_drift_on_body_change_only(self, text):
        cur = cpf.build_current(text + "\nNew promotional rebate paragraph.")
        verdict, reasons = cpf.classify(cur, _snapshot(text), _constants())
        assert verdict == "soft_drift"
        assert any("content hash" in r for r in reasons)

    def test_soft_drift_on_effective_date(self, text):
        cur = cpf.build_current(text.replace("October 1, 2026", "November 1, 2026"))
        verdict, reasons = cpf.classify(cur, _snapshot(text), _constants())
        assert verdict == "soft_drift"
        assert any("effective" in r for r in reasons)

    def test_inconclusive_when_theta_absent(self):
        cur = cpf.build_current("page format changed, no table")
        assert cpf.classify(cur, None, _constants())[0] == "inconclusive"

    def test_no_snapshot_matches_when_numbers_ok(self, text):
        assert cpf.classify(cpf.build_current(text), None, _constants()) == ("match", [])


class TestAgainstFees:
    def test_fixture_table_reproduced_by_order_fee(self, text):
        """Every published 100-lot row must be reproduced by polymarket_us_order_fee."""
        table = cpf.parse_schedule(text)["fee_table"]
        assert cpf.check_fee_table(table, polymarket_us_order_fee) == []

    def test_table_check_flags_old_theta(self, text):
        """With the pre-2026-10-01 theta the table check would have failed."""
        table = cpf.parse_schedule(text)["fee_table"]

        def old(price, contracts, maker=False):
            theta = POLYMARKET_US_MAKER_THETA if maker else 0.06
            return round(theta * contracts * price * (1 - price), 2)

        assert cpf.check_fee_table(table, old)

    def test_snapshot_agrees_with_fees_constants(self):
        snap = json.loads((_REPO / "data" / "polymarket_us_fee_schedule.snapshot.json").read_text())
        ex = snap["extracted"]
        assert ex["taker_theta"] == pytest.approx(POLYMARKET_US_TAKER_THETA)
        assert ex["maker_theta"] == pytest.approx(POLYMARKET_US_MAKER_THETA)
        assert len(snap["content_sha256"]) == 64


class TestCli:
    def test_offline_cli_against_matching_snapshot(self, text, monkeypatch, tmp_path, capsys):
        snap_path = tmp_path / "snap.json"
        cur = cpf.build_current(text)
        snap_path.write_text(json.dumps({"extracted": cur["extracted"], "content_sha256": cur["content_sha256"]}))
        monkeypatch.setattr(cpf, "SNAPSHOT_PATH", snap_path)
        monkeypatch.setattr(cpf, "RESULT_PATH", tmp_path / "result.json")
        assert cpf.main(["--offline", str(_FIXTURE)]) == 0
        assert "unchanged" in capsys.readouterr().out

    def test_offline_cli_hard_drift_exit_code(self, text, monkeypatch, tmp_path):
        drifted = tmp_path / "drifted.txt"
        drifted.write_text(text.replace("0.0695", "0.0800"))
        monkeypatch.setattr(cpf, "SNAPSHOT_PATH", tmp_path / "missing.json")
        monkeypatch.setattr(cpf, "RESULT_PATH", tmp_path / "result.json")
        assert cpf.main(["--offline", str(drifted)]) == 4
