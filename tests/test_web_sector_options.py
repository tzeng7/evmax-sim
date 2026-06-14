"""The dashboard's sector options are sourced from the category registry,
not a hardcoded list — see evmax.web.app._scan_sector_specs / api_categories."""

from __future__ import annotations

from evmax.categories import all_categories
from evmax.web.app import (
    _default_scan_sectors,
    _scan_sector_specs,
    api_categories,
)


def test_scan_sector_specs_match_registry_game_categories():
    keys = {c.key for c in _scan_sector_specs()}
    expected = {
        c.key for c in all_categories() if not c.is_prop and c.mode != "disabled"
    }
    assert keys == expected
    # Sanity: known game sectors are present.
    assert {"nba", "soccer", "tennis"} <= keys


def test_scan_sector_specs_exclude_props():
    keys = {c.key for c in _scan_sector_specs()}
    assert "nba_props" not in keys
    assert "nfl_props" not in keys
    assert not any(c.is_prop for c in _scan_sector_specs())


def test_scan_sector_specs_exclude_disabled():
    assert all(c.mode != "disabled" for c in _scan_sector_specs())


def test_default_scan_sectors_is_comma_joined_keys():
    s = _default_scan_sectors()
    assert s == ",".join(c.key for c in _scan_sector_specs())
    assert "::" not in s and " " not in s
    assert "nba" in s.split(",")
    assert "nba_props" not in s.split(",")


def test_api_categories_payload_shape():
    import asyncio

    resp = asyncio.run(api_categories())
    import json

    payload = json.loads(bytes(resp.body))
    cats = payload["categories"]
    assert cats, "expected at least one sector option"
    keys = {c["key"] for c in cats}
    assert keys == {c.key for c in _scan_sector_specs()}
    for c in cats:
        assert set(c.keys()) == {"key", "display_name", "mode"}
        assert c["display_name"]
        assert c["mode"] != "disabled"
