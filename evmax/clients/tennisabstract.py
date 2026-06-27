"""Tennis Abstract Elo ratings client.

Jeff Sackmann's GitHub data repos (``tennis_atp`` / ``tennis_wta``) went offline
in 2026, taking the match-level CSVs that ``seed_tennis_models.py`` relied on with
them. His site, **tennisabstract.com**, still publishes weekly-updated Elo
leaderboards, which are a cleaner seeding source for the surface-aware Elo model:
they are *pre-computed*, peak-weighted, surface-specific ratings rather than raw
results we would have to replay.

Pages (one row per player, sorted by overall Elo)::

    https://tennisabstract.com/reports/atp_elo_ratings.html
    https://tennisabstract.com/reports/wta_elo_ratings.html

Each row carries overall ``Elo`` plus surface blends ``hElo`` / ``cElo`` / ``gElo``
that map directly onto the model's hard / clay / grass buckets. Ratings use the
standard 400-point logistic scale (a 100-point gap implies ~64% in a best-of-three),
which is exactly the scale ``TennisModelAgent.predict_pair`` assumes, so the values
drop straight into the model without rescaling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import json

import httpx

ELO_URLS = {
    "atp": "https://tennisabstract.com/reports/atp_elo_ratings.html",
    "wta": "https://tennisabstract.com/reports/wta_elo_ratings.html",
}

# `leaders.cgi` builds its serve/return/break tables client-side from a `matchmx`
# array served in these JS files — a per-match matrix (full tour) that replaces the
# offline Sackmann match CSVs. Each tour fans out across ranking-bucket segments
# (top-50 / 51-100 / challenger); the union covers the bettable field.
LEADERSOURCE_FILES = {
    "atp": (
        "https://www.tennisabstract.com/jsmatches/leadersource.js",
        "https://www.tennisabstract.com/jsmatches/leadersource51.js",
        "https://www.tennisabstract.com/jsmatches/leadersource_chall.js",
    ),
    "wta": (
        "https://www.tennisabstract.com/jsmatches/leadersource_wta.js",
        "https://www.tennisabstract.com/jsmatches/leadersource51_wta.js",
    ),
}

# winners / unforced-error leaderboards (Match Charting Project, static HTML) — the
# one advanced-model feature matchmx lacks. Charted-match coverage is sparse, so the
# advanced agent falls back to its RPW-only reduced model for unlisted players.
WINNERS_ERRORS_URLS = {
    "atp": "https://tennisabstract.com/reports/winners_errors_leaders_men_last52.html",
    "wta": "https://tennisabstract.com/reports/winners_errors_leaders_women_last52.html",
}

# Column index of each field in a matchmx row (from the page's `var matchhead`).
# A row is one player's record of one match; the opponent columns (o*) carry the
# other side's serve stats, so a single `wl == "W"` row reconstructs the full match.
_MX = {
    "date": 0, "surf": 2, "level": 3, "wl": 4, "player": 5, "rank": 6,
    "opp": 12, "orank": 13, "time": 26,
    "pts": 29, "fwon": 31, "swon": 32, "saved": 34, "chances": 35,
    "opts": 38, "ofwon": 40, "oswon": 41, "osaved": 43, "ochances": 44,
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (evmax tennis Elo seeder)"}

_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_ROW_RE = re.compile(r"<tr[^>]*>.*?</tr>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]*>")
_UPDATE_RE = re.compile(r"Last update:\s*([\d-]+)")


@dataclass(frozen=True)
class PlayerElo:
    """One player's current Elo ratings from a Tennis Abstract leaderboard."""

    player: str
    elo: float                      # overall Elo
    hard: Optional[float] = None    # hElo
    clay: Optional[float] = None    # cElo
    grass: Optional[float] = None   # gElo
    official_rank: Optional[int] = None


def _clean(cell_html: str) -> str:
    """Strip tags/entities from a table cell, collapsing whitespace."""
    text = _TAG_RE.sub("", cell_html)
    text = text.replace("&nbsp;", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_elo_html(html: str) -> list[PlayerElo]:
    """Parse a Tennis Abstract Elo leaderboard page into ``PlayerElo`` rows.

    Locates the data header by its cell labels (``Player`` + ``hElo`` as discrete
    cells — the page's intro paragraph mentions those words in running prose but
    sits in a single-cell row, so it is naturally skipped). Rows whose Elo column
    is non-numeric (stray header/footer rows) are dropped.
    """
    rows = _ROW_RE.findall(html)
    header: Optional[list[str]] = None
    header_pos = -1
    for i, row in enumerate(rows):
        cells = [_clean(c) for c in _CELL_RE.findall(row)]
        if "Player" in cells and "hElo" in cells:
            header = cells
            header_pos = i
            break
    if header is None:
        return []

    def col(label: str) -> Optional[int]:
        return header.index(label) if label in header else None

    i_player = col("Player")
    i_elo = col("Elo")
    i_hard = col("hElo")
    i_clay = col("cElo")
    i_grass = col("gElo")
    i_rank = col("ATP Rank")
    if i_rank is None:
        i_rank = col("WTA Rank")
    if i_player is None or i_elo is None:
        return []

    out: list[PlayerElo] = []
    for row in rows[header_pos + 1:]:
        cells = [_clean(c) for c in _CELL_RE.findall(row)]
        if not cells:
            continue

        def num(idx: Optional[int]) -> Optional[float]:
            if idx is None or idx >= len(cells):
                return None
            try:
                return float(cells[idx])
            except (ValueError, TypeError):
                return None

        name = cells[i_player] if i_player < len(cells) else ""
        elo = num(i_elo)
        if not name or elo is None:
            # Non-data row (repeated header, footer, or blank) — skip.
            continue

        rank: Optional[int] = None
        if i_rank is not None and i_rank < len(cells) and cells[i_rank].isdigit():
            rank = int(cells[i_rank])

        out.append(PlayerElo(
            player=name,
            elo=elo,
            hard=num(i_hard),
            clay=num(i_clay),
            grass=num(i_grass),
            official_rank=rank,
        ))
    return out


def parse_last_update(html: str) -> Optional[str]:
    """Return the ``Last update: YYYY-MM-DD`` date stamped on the page, if present."""
    m = _UPDATE_RE.search(html)
    return m.group(1) if m else None


def fetch_elo_page(
    tour: str,
    *,
    timeout: float = 30.0,
    client: Optional[httpx.Client] = None,
) -> tuple[list[PlayerElo], Optional[str]]:
    """Fetch a tour's Elo leaderboard, returning ``(rows, last_update_date)``.

    Pass ``client`` to reuse a connection / inject a transport in tests.
    """
    key = tour.lower()
    if key not in ELO_URLS:
        raise ValueError(f"Unknown tour {tour!r}; expected one of {sorted(ELO_URLS)}")
    url = ELO_URLS[key]
    if client is not None:
        resp = client.get(url, timeout=timeout, headers=_HEADERS)
    else:
        resp = httpx.get(url, timeout=timeout, headers=_HEADERS, follow_redirects=True)
    resp.raise_for_status()
    return parse_elo_html(resp.text), parse_last_update(resp.text)


def fetch_elo_ratings(
    tour: str,
    *,
    timeout: float = 30.0,
    client: Optional[httpx.Client] = None,
) -> list[PlayerElo]:
    """Fetch and parse the current Elo leaderboard for ``atp`` or ``wta``.

    Pass ``client`` to reuse a connection / inject a transport in tests.
    """
    rows, _ = fetch_elo_page(tour, timeout=timeout, client=client)
    return rows


# ---------------------------------------------------------------------------
# matchmx — per-match serve/return matrix (replaces the offline Sackmann CSVs)
# ---------------------------------------------------------------------------

_MATCHMX_RE = re.compile(r"var\s+matchmx\s*=\s*(\[.*?\]);", re.S)


def _i(row: list, key: str):
    idx = _MX[key]
    return row[idx] if idx < len(row) else None


def parse_matchmx(js_text: str) -> list[dict]:
    """Parse a ``leadersource*.js`` body into Sackmann-shaped per-match dicts.

    The JS file assigns ``var matchmx = [[...], ...]`` where each inner array is
    one player's record of one match. We keep only ``wl == "W"`` rows (one per
    match, from the winner's side) and emit dicts using the same field names the
    Sackmann match CSVs used — so the existing ``aggregate_*`` seeders in
    ``scripts/seed_tennis_models.py`` consume them unchanged. Match string values
    never contain ``]``, so the non-greedy array capture is safe.
    """
    m = _MATCHMX_RE.search(js_text)
    if not m:
        return []
    try:
        mx = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    out: list[dict] = []
    for row in mx:
        if _i(row, "wl") != "W":
            continue
        winner = (_i(row, "player") or "").strip()
        loser = (_i(row, "opp") or "").strip()
        date = _i(row, "date")
        if not winner or not loser or not date:
            continue
        out.append({
            "tourney_date": date,
            "surface": _i(row, "surf") or "",
            "tourney_level": _i(row, "level") or "",
            "minutes": _i(row, "time"),
            "winner_name": winner,
            "loser_name": loser,
            "winner_rank": _i(row, "rank"),
            "loser_rank": _i(row, "orank"),
            # winner serves on own service games; opponent (o*) columns are the loser
            "w_svpt": _i(row, "pts"),
            "w_1stWon": _i(row, "fwon"),
            "w_2ndWon": _i(row, "swon"),
            "w_bpSaved": _i(row, "saved"),
            "w_bpFaced": _i(row, "chances"),
            "l_svpt": _i(row, "opts"),
            "l_1stWon": _i(row, "ofwon"),
            "l_2ndWon": _i(row, "oswon"),
            "l_bpSaved": _i(row, "osaved"),
            "l_bpFaced": _i(row, "ochances"),
        })
    return out


def fetch_matchmx(
    tour: str,
    *,
    timeout: float = 90.0,
    client: Optional[httpx.Client] = None,
) -> list[dict]:
    """Fetch + merge per-match records for a tour across its ranking segments.

    Unions the top-50 / 51-100 / challenger ``leadersource`` files and dedupes on
    (date, winner, loser) so a match listed under more than one segment counts once.
    """
    key = tour.lower()
    if key not in LEADERSOURCE_FILES:
        raise ValueError(f"Unknown tour {tour!r}; expected one of {sorted(LEADERSOURCE_FILES)}")

    seen: set[tuple] = set()
    merged: list[dict] = []
    for url in LEADERSOURCE_FILES[key]:
        if client is not None:
            resp = client.get(url, timeout=timeout, headers=_HEADERS)
        else:
            resp = httpx.get(url, timeout=timeout, headers=_HEADERS, follow_redirects=True)
        if resp.status_code != 200:
            continue
        for rec in parse_matchmx(resp.text):
            sig = (rec["tourney_date"], rec["winner_name"], rec["loser_name"])
            if sig in seen:
                continue
            seen.add(sig)
            merged.append(rec)
    return merged


# ---------------------------------------------------------------------------
# winners / unforced errors (advanced-stats UE feature)
# ---------------------------------------------------------------------------

def parse_winners_errors_html(html: str) -> dict[str, dict[str, int]]:
    """Parse a winners/errors leaderboard into ``{player_key: {winners, unforced}}``.

    ``player_key`` is the lowercased full name (matching the advanced seeder's keys).
    """
    rows = _ROW_RE.findall(html)
    header: Optional[list[str]] = None
    header_pos = -1
    for i, row in enumerate(rows):
        cells = [_clean(c) for c in _CELL_RE.findall(row)]
        if "Player" in cells and "Winners" in cells and "UFEs" in cells:
            header = cells
            header_pos = i
            break
    if header is None:
        return {}
    i_player = header.index("Player")
    i_win = header.index("Winners")
    i_ufe = header.index("UFEs")

    out: dict[str, dict[str, int]] = {}
    for row in rows[header_pos + 1:]:
        cells = [_clean(c) for c in _CELL_RE.findall(row)]
        if max(i_player, i_win, i_ufe) >= len(cells):
            continue
        name = cells[i_player]
        try:
            winners = int(cells[i_win])
            unforced = int(cells[i_ufe])
        except (ValueError, TypeError):
            continue
        if name:
            out[name.lower().strip()] = {"winners": winners, "unforced": unforced}
    return out


def fetch_winners_errors(
    tour: str,
    *,
    timeout: float = 30.0,
    client: Optional[httpx.Client] = None,
) -> dict[str, dict[str, int]]:
    """Fetch the winners/unforced-errors leaderboard for ``atp`` or ``wta``."""
    key = tour.lower()
    if key not in WINNERS_ERRORS_URLS:
        raise ValueError(f"Unknown tour {tour!r}; expected one of {sorted(WINNERS_ERRORS_URLS)}")
    url = WINNERS_ERRORS_URLS[key]
    if client is not None:
        resp = client.get(url, timeout=timeout, headers=_HEADERS)
    else:
        resp = httpx.get(url, timeout=timeout, headers=_HEADERS, follow_redirects=True)
    resp.raise_for_status()
    return parse_winners_errors_html(resp.text)
