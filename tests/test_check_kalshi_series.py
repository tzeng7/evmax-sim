"""Unit tests for the Kalshi series-drift classifier (no network).

``scripts/check_kalshi_series.py`` guards SECTOR_SERIES_MAP against drift: a
phantom/retired ticker returns zero markets silently (the NFL-prop typo that
motivated ARCH-12 — ``KXNFLPAS`` shipped instead of the real ``KXNFLPASSYDS``).
``classify_series`` is the pure core; ``main()`` only wraps it with the live
``/series`` fetch and reporting.

STALE detection is exact set membership (the safety-critical path). NEW
detection is informational and deliberately narrow: a live ticker is NEW only
when it *extends* one of our full-ticker prefixes — which catches the typo
shape above but not unrelated sibling market types (see
``test_sibling_market_type_is_not_new`` for the documented limitation).
"""
from __future__ import annotations

from scripts.check_kalshi_series import _sector_prefixes, classify_series


def test_ok_bucket_when_ticker_live():
    ok, stale, new = classify_series({"nba": ["KXNBAGAME"]}, {"KXNBAGAME"})
    assert ok == [("KXNBAGAME", "nba")]
    assert stale == []
    assert new == []


def test_stale_and_new_catch_the_nfl_prop_typo():
    # The motivating bug: a phantom short ticker is STALE, and the real ticker
    # that extends it surfaces as NEW so the fix is obvious.
    ok, stale, new = classify_series(
        {"nfl_props": ["KXNFLPAS"]}, {"KXNFLPASSYDS"}
    )
    assert stale == [("KXNFLPAS", "nfl_props")]
    assert ok == []
    assert new == [("KXNFLPASSYDS", "nfl_props")]


def test_new_fires_on_extension_ticker():
    ok, stale, new = classify_series(
        {"nba": ["KXNBAGAME"]}, {"KXNBAGAME", "KXNBAGAMEALT"}
    )
    assert ("KXNBAGAME", "nba") in ok
    assert new == [("KXNBAGAMEALT", "nba")]


def test_sibling_market_type_is_not_new():
    # Documented limitation: KXNBASPREAD is NOT flagged NEW because the derived
    # prefix is the full alpha ticker KXNBAGAME, not a short KXNBA stem. (Kept
    # narrow on purpose — a KXNBA stem would collide with nba_props' KXNBA*.)
    ok, stale, new = classify_series(
        {"nba": ["KXNBAGAME"]}, {"KXNBAGAME", "KXNBASPREAD"}
    )
    assert new == []


def test_unrelated_live_series_not_flagged_new():
    ok, stale, new = classify_series(
        {"nba": ["KXNBAGAME"]}, {"KXNBAGAME", "KXWNBAGAME"}
    )
    # KXWNBAGAME does not start with the KXNBAGAME prefix.
    assert new == []


def test_new_attributed_to_first_matching_sector_only():
    ok, stale, new = classify_series(
        {"nba": ["KXNBAGAME"], "wnba": ["KXWNBAGAME"]},
        {"KXNBAGAME", "KXWNBAGAME", "KXWNBAGAMEALT"},
    )
    # KXWNBAGAMEALT extends the wnba prefix (KXWNBAGAME), not nba.
    assert ("KXWNBAGAMEALT", "wnba") in new
    assert ("KXWNBAGAMEALT", "nba") not in new


def test_sector_prefixes_requires_min_length():
    # Prefixes shorter than 4 alpha chars are dropped (avoids over-broad matches).
    prefixes = _sector_prefixes({"x": ["KX1"], "nba": ["KXNBAGAME"]})
    assert prefixes["x"] == set()            # "KX" (before the digit) is too short
    assert prefixes["nba"] == {"KXNBAGAME"}  # full alpha run, no digit


def test_empty_inputs():
    assert classify_series({}, set()) == ([], [], [])


def test_real_registry_is_self_consistent():
    # When the live set equals our map, everything is OK — no spurious STALE/NEW
    # (guards the nba/nba_props and ncaab/ncaaw/ncaaf shared-stem shapes).
    from evmax.clients.kalshi import SECTOR_SERIES_MAP

    live = {t for tickers in SECTOR_SERIES_MAP.values() for t in tickers}
    ok, stale, new = classify_series(SECTOR_SERIES_MAP, live)
    assert stale == []
    assert new == []
    assert len(ok) == len(live)
