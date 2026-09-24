"""Fuzzy matching helpers using rapidfuzz."""

from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz, process

from evmax.clients.time_util import uses_et_game_day

_log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 88


def fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: int = DEFAULT_THRESHOLD,
) -> Optional[tuple[str, float]]:
    """
    Find the best fuzzy match for query in candidates.

    Args:
        query: String to search for.
        candidates: List of candidate strings.
        threshold: Minimum score to consider a match (0–100).

    Returns:
        (best_match, score) if above threshold, else None.
    """
    if not query or not candidates:
        return None

    result = process.extractOne(
        query,
        candidates,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )
    if result is None:
        return None
    match, score, _ = result
    return match, float(score)


def fuzzy_match_event_keys(
    query_key: str,
    candidate_keys: list[str],
    threshold: int = DEFAULT_THRESHOLD,
) -> Optional[tuple[str, float]]:
    """
    Fuzzy match an event key against a list of sharp event keys.

    The key format is "{sector}::{date}::{team_a}_vs_{team_b}".
    The date segment is a filter, never a fuzzy-scored string: only the team
    part is scored.

    Date filter by sector:
      - ET-dated sectors (`uses_et_game_day`: nba, wnba, nfl, baseball, nhl,
        ncaab, ncaaw, ncaaf, ufc): the candidate date must EQUAL the query
        date. Both sides are already on the ET game day, so any other date is
        another game — the next game of a series or a back-to-back.
      - Every other sector: ±1 day, then any date in the sector when nothing
        is within ±1 day. Kalshi dates these markets on a local/US day while
        Pinnacle keys the UTC day (Americas-evening soccer, World Cup 2026),
        and Kalshi tennis tickers carry a listing date, not the match day.
        Among equal team matches, the candidate nearest the query date wins.

    Args:
        query_key: Canonical event key from prediction market.
        candidate_keys: Sharp event keys to match against.
        threshold: Fuzzy threshold (default 88).

    Returns:
        (best_key, score) or None.
    """
    def extract_teams(key: str) -> str:
        parts = key.split("::")
        raw = parts[2] if len(parts) >= 3 else key
        # Replace underscores with spaces so token_sort_ratio can sort team tokens.
        # Without this, "brugge_vs_atletico" is a single token and can't be
        # sorted/compared against "atletico_vs_brugge".
        return raw.replace("_", " ")

    # Filter candidates to same sector and within date window
    query_parts = query_key.split("::")
    query_sector = query_parts[0] if query_parts else ""
    query_date = query_parts[1] if len(query_parts) > 1 else ""
    query_teams = extract_teams(query_key)

    if uses_et_game_day(query_sector):
        # Same ET game day only — no window, no any-date fallback. An unknown
        # date can't be proven to be the same game day, so it never matches.
        if query_date in ("", "unknown"):
            return None
        filtered = [
            k for k in candidate_keys
            if k.startswith(f"{query_sector}::") and _key_date(k) == query_date
        ]
        if not filtered:
            return None
    else:
        # Only match events in same sector and within ±1 day
        filtered = [
            k for k in candidate_keys
            if k.startswith(f"{query_sector}::") and _date_close(query_date, _key_date(k))
        ]

    if not filtered:
        # Try same-sector candidates regardless of date before falling back to all
        same_sector = [k for k in candidate_keys if k.startswith(f"{query_sector}::")]
        if same_sector:
            _log.debug(
                "fuzzy_date_filter_empty: falling back to %d same-sector candidates for %s",
                len(same_sector),
                query_key,
            )
            filtered = same_sector
        else:
            # No same-sector candidates at all — cross-sector match is always wrong
            _log.debug(
                "fuzzy_date_filter_empty: no same-sector candidates for %s — no match",
                query_key,
            )
            return None

    best = fuzzy_match(query_teams, [extract_teams(k) for k in filtered], threshold)
    if best is None:
        return None

    match_teams, score = best

    # Secondary check: verify each individual team token has a close counterpart.
    # This prevents "nets vs celtics" from matching "nuggets vs celtics" even though
    # the combined token_sort_ratio scores ~91% due to character overlap in "nets/nuggets".
    if not _teams_individually_match(query_teams, match_teams):
        return None

    # Recover the full key from the matched teams string. Several dates can
    # carry the same teams (a rematch inside the ±1-day window); take the one
    # nearest the query date, not the first one listed.
    same_teams = [k for k in filtered if extract_teams(k) == match_teams]
    if same_teams:
        return min(same_teams, key=lambda k: _day_gap(query_date, _key_date(k))), score

    return None


def _key_date(key: str) -> str:
    """Date segment of an event key ("" when the key has none)."""
    parts = key.split("::")
    return parts[1] if len(parts) > 1 else ""


def _day_gap(date_a: str, date_b: str) -> int:
    """Absolute day gap between two YYYY-MM-DD strings (large when unparseable)."""
    try:
        from datetime import date
        return abs((date.fromisoformat(date_a) - date.fromisoformat(date_b)).days)
    except ValueError:
        return 10**6


def _teams_individually_match(query_teams: str, match_teams: str, min_score: int = 90) -> bool:
    """
    Verify each team token in the query matches at least one token in the candidate.

    Prevents false positives like "nets vs celtics" matching "nuggets vs celtics"
    where the overall token_sort_ratio is high but individual teams don't match.
    """
    q_tokens = [t for t in query_teams.split() if t != "vs"]
    m_tokens = [t for t in match_teams.split() if t != "vs"]

    if not q_tokens or not m_tokens:
        return False  # Empty tokens — conservative reject to prevent false positives

    for qt in q_tokens:
        best = process.extractOne(qt, m_tokens, scorer=fuzz.ratio, score_cutoff=min_score)
        if best is None:
            return False  # This team has no close match in the candidate

    return True


def _date_close(date_a: str, date_b: str, max_days: int = 1) -> bool:
    """Check if two YYYY-MM-DD date strings are within max_days of each other."""
    if not date_a or not date_b or date_a == "unknown" or date_b == "unknown":
        return True  # Can't check, allow match
    try:
        from datetime import date
        a = date.fromisoformat(date_a)
        b = date.fromisoformat(date_b)
        return abs((a - b).days) <= max_days
    except ValueError:
        return True
