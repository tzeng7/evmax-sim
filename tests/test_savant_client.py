"""Tests for the Baseball Savant xERA client (WS2a).

parse_expected_stats_csv is exercised against a fixture mirroring the real
export — including the UTF-8 BOM regression: attached to the opening quote
of the first header, the BOM breaks csv quote detection and shifts every
column by one (the bug that made the first live fetch parse 1 pitcher).
"""

from __future__ import annotations

import json
import time

import pytest

from evmax.clients import savant
from evmax.clients.savant import parse_expected_stats_csv

_CSV = (
    '﻿"last_name, first_name","player_id","year","pa","bip","ba","est_ba",'
    '"est_ba_minus_ba_diff","slg","est_slg","est_slg_minus_slg_diff","woba",'
    '"est_woba","est_woba_minus_woba_diff","era","xera","era_minus_xera_diff"\n'
    '"Webb, Logan","657277","2025","856","579",0.264,0.246,0.018,0.386,0.385,'
    '0.001,0.303,0.295,0.008,"3.22","3.58","-0.363"\n'
    '"Sánchez, Cristopher","650911","2025","800","520",0.240,0.235,0.005,0.360,'
    '0.355,0.005,0.290,0.288,0.002,"2.95","3.10","-0.15"\n'
    '"Broken, Row","not-an-id","2025","10","5",0.2,0.2,0,0.3,0.3,0,0.25,0.25,0,'
    '"4.00","4.10","-0.1"\n'
    '"NoXera, Guy","999999","2025","50","30",0.2,0.2,0,0.3,0.3,0,0.25,0.25,0,'
    '"4.00","",""\n'
)


class TestParse:
    def test_bom_does_not_shift_columns(self):
        out = parse_expected_stats_csv(_CSV)
        webb = out[657277]
        # Pre-fix, player_id parsed as the year and xera as the diff column.
        assert webb["xera"] == pytest.approx(3.58)
        assert webb["era"] == pytest.approx(3.22)
        assert webb["pa"] == 856
        assert webb["bip"] == 579

    def test_name_reordered_and_folded(self):
        out = parse_expected_stats_csv(_CSV)
        assert out[657277]["name"] == "Logan Webb"
        assert out[657277]["name_key"] == "logan webb"
        # Accent folding for the join against the BR-seeded pitcher DB.
        assert out[650911]["name_key"] == "cristopher sanchez"

    def test_bad_rows_skipped(self):
        out = parse_expected_stats_csv(_CSV)
        assert 999999 not in out          # empty xERA
        assert len(out) == 2              # non-integer player_id dropped too


class TestFetchCache:
    @pytest.mark.asyncio
    async def test_past_season_served_from_cache_without_network(
        self, tmp_path, monkeypatch
    ):
        cache = {
            "2024": {
                "fetched_at": time.time() - 999_999,  # ancient — past seasons never expire
                "players": {"657277": {"name": "Logan Webb", "name_key": "logan webb",
                                        "pa": 800, "bip": 550, "era": 3.4, "xera": 3.6}},
            }
        }
        cache_path = tmp_path / "savant_xera_cache.json"
        cache_path.write_text(json.dumps(cache))
        monkeypatch.setattr(savant, "CACHE_PATH", cache_path)

        def _boom(*a, **k):
            raise AssertionError("network must not be touched for cached past seasons")

        monkeypatch.setattr("httpx.AsyncClient.get", _boom)
        out = await savant.fetch_expected_stats(2024)
        assert out[657277]["xera"] == 3.6

    @pytest.mark.asyncio
    async def test_total_failure_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(savant, "CACHE_PATH", tmp_path / "cache.json")
        monkeypatch.setattr(savant, "_RETRIES", 1)

        async def _fail(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr("httpx.AsyncClient.get", _fail)
        monkeypatch.setattr("asyncio.sleep", _noop_sleep)
        out = await savant.fetch_expected_stats(2025)
        assert out == {}


async def _noop_sleep(_):
    return None
