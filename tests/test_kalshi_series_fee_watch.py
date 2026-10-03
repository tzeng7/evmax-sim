"""PURE classifier of the per-series fee canary (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_kalshi_series_fees as ck  # noqa: E402


def _s(fee_type="quadratic_with_maker_fees", mult=1):
    return {"fee_type": fee_type, "fee_multiplier": mult}


def test_matching_series():
    out = ck.classify_series_fees({"KXNFLGAME": _s(), "KXNFLSPREAD": _s(mult=1.0)})
    assert [f["status"] for f in out] == ["match", "match"]


def test_multiplier_mismatch_reports_value():
    (f,) = ck.classify_series_fees({"KXMLBGAME": _s(mult=0.5)})
    assert f["status"] == "mismatch" and f["fee_multiplier"] == 0.5


def test_fee_type_mismatch():
    (f,) = ck.classify_series_fees({"KXFOO": _s(fee_type="quadratic")})
    assert f["status"] == "mismatch" and f["fee_type"] == "quadratic"


def test_missing_or_unparseable_payloads():
    out = {f["series"]: f for f in ck.classify_series_fees(
        {"A": None, "B": {"fee_type": "quadratic_with_maker_fees", "fee_multiplier": "x"}, "C": {}})}
    assert out["A"]["status"] == "unavailable" and out["C"]["status"] == "unavailable"
    assert out["B"]["status"] == "mismatch" and out["B"]["fee_multiplier"] is None


def test_traded_series_dedupes_and_filters():
    nfl = ck.traded_series(["nfl"])
    assert nfl[:3] == ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL"]
    assert len(ck.traded_series()) == len(set(ck.traded_series()))
