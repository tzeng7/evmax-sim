"""ESPN golf data adapter — FREE, unauthenticated.

ESPN's public golf API gives the tournament field, per-round scores, and
finishing state for every event on the calendar. It does NOT carry
strokes-gained, so it plays two roles here:

  * the FIELD source (who is teeing it up this week) — required for pricing,
    since we can only price golfers who are actually in the tournament;
  * a score-derived SG-TOTAL *proxy* source — the no-SG fallback that lets the
    whole skill→sim→pricing chain run with zero paid data. The proxy is
    ``field_mean_to_par_per_round - player_to_par_per_round`` (better than the
    field ⇒ positive), which is exactly SG-total-vs-field up to a constant.

Parsers are pure functions over the decoded JSON so they unit-test against the
captured ``espn_scoreboard.json`` fixture. ``fetch_scoreboard`` is the thin
live wrapper (stdlib urllib — no new dependency, and this module runs outside
the async scan loop).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from evmax.golf.models import SGRecord
from evmax.golf.names import canonical

_log = logging.getLogger(__name__)

_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/{tour}/scoreboard"
_UA = "Mozilla/5.0 (evmax-golf research)"


@dataclass
class GolfEntry:
    player: str  # canonical
    display_name: str
    to_par: Optional[float]  # tournament total relative to par (e.g. -17)
    rounds_played: int
    round_scores: list[float] = field(default_factory=list)
    position: Optional[str] = None  # ESPN display, e.g. "T4", "CUT" (often None)


@dataclass
class GolfResult:
    event_id: str
    event: str
    completed: bool
    entries: list[GolfEntry]
    date: str = ""  # ISO event date (start), for chronological ordering

    def field_names(self) -> list[str]:
        return [e.player for e in self.entries]


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        # ESPN totals come as ints, "-17", or "E"
        if isinstance(v, str) and v.strip().upper() in ("E", "EVEN"):
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_scoreboard(payload: dict) -> list[GolfResult]:
    """Parse an ESPN golf scoreboard payload into per-event results."""
    results: list[GolfResult] = []
    for e in payload.get("events", []):
        comps = e.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        entries: list[GolfEntry] = []
        for c in competitors:
            a = c.get("athlete") or {}
            name = a.get("displayName")
            if not name:
                continue
            linescores = [
                _to_float((ls or {}).get("value")) for ls in (c.get("linescores") or [])
            ]
            linescores = [x for x in linescores if x is not None]
            pos = ((c.get("status") or {}).get("position") or {}).get("displayName")
            entries.append(
                GolfEntry(
                    player=canonical(name),
                    display_name=name,
                    to_par=_to_float(c.get("score")),
                    rounds_played=len(linescores),
                    round_scores=linescores,
                    position=pos,
                )
            )
        st = ((e.get("status") or {}).get("type") or {})
        results.append(
            GolfResult(
                event_id=str(e.get("id")),
                event=e.get("name", ""),
                completed=bool(st.get("completed")),
                entries=entries,
                date=(e.get("date") or "")[:10],  # YYYY-MM-DD
            )
        )
    return results


def field_from_scoreboard(payload: dict, event_id: Optional[str] = None) -> list[str]:
    """Canonical names of the field for an event (first event if id omitted)."""
    results = parse_scoreboard(payload)
    if not results:
        return []
    if event_id is not None:
        results = [r for r in results if r.event_id == str(event_id)] or results
    return results[0].field_names()


def result_to_sg_proxy(
    result: GolfResult, cut_size: int = 65, min_field: int = 20
) -> list[SGRecord]:
    """Convert a COMPLETED event into SG-total-proxy records (one per golfer).

    SG-total proxy (per round, vs field) = field_mean_to_par_per_round −
    player_to_par_per_round. Uses tournament to-par totals normalised by rounds
    played, so par and gross scores are not needed. Missed-cut players (≤ 2
    rounds) still contribute a (noisier, 2-round) proxy.
    """
    entries = [e for e in result.entries if e.to_par is not None and e.rounds_played > 0]
    if len(entries) < min_field:
        return []

    per_round = [e.to_par / e.rounds_played for e in entries]
    field_mean = sum(per_round) / len(per_round)

    # Finishing order among those who played the weekend (>=3 rounds = made cut
    # in a standard 36-hole-cut event; unanimous-4-round no-cut events → all).
    made = [e for e in entries if e.rounds_played >= 3]
    made_sorted = sorted(made, key=lambda e: (e.to_par if e.to_par is not None else 999))
    pos_by_player = {e.player: i + 1 for i, e in enumerate(made_sorted)}

    records: list[SGRecord] = []
    for e, pr in zip(entries, per_round):
        made_cut = e.rounds_played >= 3
        records.append(
            SGRecord(
                player=e.player,
                event=result.event,
                date=result.date,
                sg_total=field_mean - pr,  # better than field ⇒ positive
                made_cut=made_cut,
                finish_pos=pos_by_player.get(e.player),
                n_rounds=e.rounds_played,
                source="espn_proxy",
            )
        )
    return records


def fetch_scoreboard(tour: str = "pga", timeout: float = 25.0) -> dict:
    """Live-fetch the ESPN scoreboard for a tour (pga/lpga/eur/liv)."""
    url = _SCOREBOARD_URL.format(tour=tour)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        return json.load(resp)
