"""Tests for the ESPN FPI client (evmax/clients/cfb_fpi.py).

Network paths are exercised only through a failing fake client (fail-soft
contract); the parsers and the centring helper are the load-bearing logic.
"""

from __future__ import annotations

import json

from evmax.clients import cfb_fpi as F


def test_parse_live_response_reads_fpi_category_only():
    data = {
        "teams": [
            {"team": {"id": "194", "displayName": "Ohio State Buckeyes"},
             "categories": [{"name": "resume", "values": [1.0]},
                            {"name": "fpi", "values": [28.676, 1.0, 0.0]}]},
            {"team": {"id": "5", "displayName": "No FPI U"},
             "categories": [{"name": "resume", "values": [2.0]}]},
            {"team": {"id": "6"}, "categories": [{"name": "fpi", "values": ["bad"]}]},
        ]
    }
    out = F.parse_live_response(data)
    assert out == {"194": {"fpi": 28.676, "name": "Ohio State Buckeyes"}}


def _fitt_html(rows):
    blob = {"page": {"content": {"table": {"stats": rows}}}}
    return "<html><script>var x=1;</script><script>window['__espnfitt__']=" + json.dumps(blob) + ";</script></html>"


def test_parse_fitt_page_extracts_team_id_and_fpi():
    html = _fitt_html([
        {"team": {"id": "61", "displayName": "Georgia Bulldogs"},
         "stats": [{"name": "numwins", "value": "0-0"}, {"name": "fpi", "value": "26.8"}]},
        {"team": {"id": "99", "displayName": "No Value"}, "stats": [{"name": "fpirank", "value": "3"}]},
    ])
    out = F.parse_fitt_page(html)
    assert out == {"61": {"fpi": 26.8, "name": "Georgia Bulldogs"}}


def test_parse_fitt_page_malformed_is_empty():
    assert F.parse_fitt_page("<html>no blob</html>") == {}
    assert F.parse_fitt_page("<script>window['__espnfitt__']={\"page\":{}};</script>") == {}
    assert F.parse_fitt_page("") == {}


def test_centre_fpi_relative_to_rated_members_only():
    ratings = {"1": {"fpi": 10.0}, "2": {"fpi": 0.0}, "3": {"fpi": -10.0}}
    # "3" is not in the universe, "4" has no rating -> mean over {1,2} = 5
    out = F.centre_fpi(ratings, {"1", "2", "4"})
    assert out == {"1": 5.0, "2": -5.0}
    assert F.centre_fpi({}, {"1"}) == {}
    assert F.centre_fpi(ratings, set()) == {}


class _BadClient:
    def get(self, *a, **k):
        raise RuntimeError("boom")


def test_fetch_fpi_fail_soft():
    assert F.fetch_fpi(2026, client=_BadClient()) == {}


def test_wayback_helpers_fail_soft():
    assert F.wayback_snapshot_url("20240823", client=_BadClient()) is None
    assert F.fetch_preseason_fpi_wayback("20240823", client=_BadClient()) == ({}, None)
