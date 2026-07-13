"""Shared fixtures for the golf module tests. All data is captured live and
stored under tests/golf/fixtures so the suite runs fully offline."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def pga_sg_payload() -> dict:
    return _load("pga_sg_2026.json")


@pytest.fixture
def polyus_golf_payload() -> dict:
    return _load("polyus_open_winner.json")


@pytest.fixture
def kalshi_golf_markets() -> list:
    return _load("kalshi_golf_markets.json")["markets"]


@pytest.fixture
def espn_scoreboard_payload() -> dict:
    return _load("espn_scoreboard.json")
