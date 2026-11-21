"""Fuzzy matching helpers using rapidfuzz."""

from __future__ import annotations

import logging
from datetime import date, timedelta
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
    query_sector, query_date, query_teams = _split_query(query_key)
    prefix = f"{query_sector}::"

    if uses_et_game_day(query_sector):
        # Same ET game day only — no window, no any-date fallback. An unknown
        # date can't be proven to be the same game day, so it never matches.
        if query_date in ("", "unknown"):
            return None
        filtered = [
            k for k in candidate_keys
            if k.startswith(prefix) and _key_date(k) == query_date
        ]
        if not filtered:
            return None
    else:
        # Only match events in same sector and within ±1 day
        filtered = [
            k for k in candidate_keys
            if k.startswith(prefix) and _date_close(query_date, _key_date(k))
        ]

    if not filtered:
        # Try same-sector candidates regardless of date before falling back to all
        filtered = [k for k in candidate_keys if k.startswith(prefix)]
        if not filtered:
            # No same-sector candidates at all — cross-sector match is always wrong
            _log.debug(
                "fuzzy_date_filter_empty: no same-sector candidates for %s — no match",
                query_key,
            )
            return None
        _log.debug(
            "fuzzy_date_filter_empty: falling back to %d same-sector candidates for %s",
            len(filtered),
            query_key,
        )

    return _score_candidates(
        query_teams, query_date, filtered, [_extract_teams(k) for k in filtered], threshold,
    )


_Entry = tuple[int, str, str]  # (position in the candidate list, key, team string)


class FuzzyKeyIndex:
    """Sharp event keys grouped by sector and game day, built once per batch.

    ``fuzzy_match_event_keys`` re-filters the WHOLE candidate list for every
    query (sector prefix + date parse per key): O(markets x keys) per batch,
    0.84 s for one soccer scan (1,039 markets x 1,718 Pinnacle records), run
    twice per scan. This index answers the same filter with dictionary
    lookups. Every candidate list keeps the ORIGINAL key order, because
    rapidfuzz's extractOne and the nearest-date tie-break both pick the first
    candidate among equals: same order, same answer.
    """

    def __init__(self, keys: list[str]) -> None:
        # Entries are (position, key, team string). The team string is computed
        # once here instead of once per query per candidate.
        # sector -> entries for every key whose first segment is the sector
        self._by_sector: dict[str, list[_Entry]] = {}
        # sector -> raw date segment -> entries  (ET exact-day filter)
        self._by_date_str: dict[str, dict[str, list[_Entry]]] = {}
        # sector -> parsed date -> entries       (±1-day window)
        self._by_day: dict[str, dict[date, list[_Entry]]] = {}
        # sector -> entries whose date is missing/unknown/unparseable; the ±1-day
        # filter (_date_close) always lets these through.
        self._undated: dict[str, list[_Entry]] = {}
        for pos, key in enumerate(keys):
            if "::" not in key:
                continue  # can never satisfy key.startswith(f"{sector}::")
            sector = key.split("::", 1)[0]
            entry = (pos, key, _extract_teams(key))
            self._by_sector.setdefault(sector, []).append(entry)
            raw_date = _key_date(key)
            self._by_date_str.setdefault(sector, {}).setdefault(raw_date, []).append(entry)
            day = _parse_day(raw_date)
            if day is None:
                self._undated.setdefault(sector, []).append(entry)
            else:
                self._by_day.setdefault(sector, {}).setdefault(day, []).append(entry)

    def same_day(self, sector: str, raw_date: str) -> list[_Entry]:
        """Entries of ``sector`` whose date segment equals ``raw_date`` exactly."""
        return list(self._by_date_str.get(sector, {}).get(raw_date, ()))

    def within_one_day(self, sector: str, raw_date: str) -> list[_Entry]:
        """Entries of ``sector`` that ``_date_close(raw_date, key_date)`` accepts, in order."""
        day = _parse_day(raw_date)
        if day is None:  # unknown query date: _date_close accepts every key
            return self.same_sector(sector)
        by_day = self._by_day.get(sector, {})
        entries = list(self._undated.get(sector, ()))
        for offset in (-1, 0, 1):
            entries.extend(by_day.get(day + timedelta(days=offset), ()))
        entries.sort()
        return entries

    def same_sector(self, sector: str) -> list[_Entry]:
        return list(self._by_sector.get(sector, ()))


def fuzzy_match_event_keys_indexed(
    query_key: str,
    index: FuzzyKeyIndex,
    threshold: int = DEFAULT_THRESHOLD,
) -> Optional[tuple[str, float]]:
    """``fuzzy_match_event_keys`` over a prebuilt ``FuzzyKeyIndex``.

    Same filter rules, same candidate order, same scoring — only the
    candidate filter is a lookup instead of a scan of every key.
    """
    query_sector, query_date, query_teams = _split_query(query_key)

    if uses_et_game_day(query_sector):
        if query_date in ("", "unknown"):
            return None
        filtered = index.same_day(query_sector, query_date)
        if not filtered:
            return None
    else:
        filtered = index.within_one_day(query_sector, query_date)

    if not filtered:
        filtered = index.same_sector(query_sector)
        if not filtered:
            return None

    return _score_candidates(
        query_teams, query_date,
        [k for _, k, _ in filtered], [t for _, _, t in filtered], threshold,
    )


def _extract_teams(key: str) -> str:
    parts = key.split("::")
    raw = parts[2] if len(parts) >= 3 else key
    # Replace underscores with spaces so token_sort_ratio can sort team tokens.
    # Without this, "brugge_vs_atletico" is a single token and can't be
    # sorted/compared against "atletico_vs_brugge".
    return raw.replace("_", " ")


def _split_query(query_key: str) -> tuple[str, str, str]:
    """(sector, date segment, team string) of a canonical event key."""
    query_parts = query_key.split("::")
    query_sector = query_parts[0] if query_parts else ""
    query_date = query_parts[1] if len(query_parts) > 1 else ""
    return query_sector, query_date, _extract_teams(query_key)


def _score_candidates(
    query_teams: str,
    query_date: str,
    filtered: list[str],
    filtered_teams: list[str],
    threshold: int,
) -> Optional[tuple[str, float]]:
    """Fuzzy-score the team part of the date-filtered candidates.

    ``filtered_teams[i]`` is ``_extract_teams(filtered[i])``. Each distinct
    team string is scored once (a game's spread and total records repeat its
    moneyline's teams). The score depends only on the string, and extractOne
    keeps the first of equal scores, so the winner is unchanged.
    """
    best = fuzzy_match(query_teams, list(dict.fromkeys(filtered_teams)), threshold)
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
    same_teams = [k for k, t in zip(filtered, filtered_teams) if t == match_teams]
    if same_teams:
        return min(same_teams, key=lambda k: _day_gap(query_date, _key_date(k))), score

    return None


def _parse_day(raw: str) -> Optional[date]:
    """Parsed YYYY-MM-DD, or None where ``_date_close`` treats the date as unknown."""
    if not raw or raw == "unknown":
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
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
