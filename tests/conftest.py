"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_datetime():
    return datetime(2026, 2, 22, 18, 0, 0, tzinfo=timezone.utc)
