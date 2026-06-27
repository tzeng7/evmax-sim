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

import httpx

ELO_URLS = {
    "atp": "https://tennisabstract.com/reports/atp_elo_ratings.html",
    "wta": "https://tennisabstract.com/reports/wta_elo_ratings.html",
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
