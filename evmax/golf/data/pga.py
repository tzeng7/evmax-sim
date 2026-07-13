"""PGA Tour strokes-gained adapter — FREE via the public GraphQL orchestrator.

``orchestrator.pgatour.com/graphql`` backs pgatour.com/stats. It authenticates
with a fixed public ``x-api-key`` shipped in the site bundle (no account), and
the ``statDetails`` query returns season strokes-gained per player. The ``Avg``
stat value is SG PER ROUND — exactly the skill input we want — and separate
``statId``s give the four categories:

    02675 SG:Total   02567 SG:OTT   02568 SG:APP
    02569 SG:ARG     02564 SG:Putting

This is SEASON granularity (per-round SG averaged over the player's measured
rounds), not per-event — which is the real free ceiling (per-round shot-level
SG is DataGolf/ShotLink, paid). Season SG/round is a strong current-form skill
estimate on its own; the decay/recency machinery lives in the skill model for
the score-proxy path.

``parse_*`` are pure over the decoded payload (tested against
``pga_sg_2026.json``); ``fetch_sg_snapshot`` is the live wrapper.

NOTE: pgatour.com ToS restricts commercial use / redistribution and automated
collection. This is used for personal modelling with polite, cached access —
keep request volume low (one snapshot refresh per day is ample).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Optional

from evmax.golf.names import canonical

_log = logging.getLogger(__name__)

_GRAPHQL_URL = "https://orchestrator.pgatour.com/graphql"
# Public key shipped in the pgatour.com JS bundle. Confirm live via DevTools if
# it ever 403s (see docs). Not a secret / not user-specific.
_PUBLIC_API_KEY = "da2-gsrx5bibzbb4njvhl7t37wqyl4"
_UA = "Mozilla/5.0 (evmax-golf research)"

SG_STAT_IDS = {
    "total": "02675",
    "ott": "02567",
    "app": "02568",
    "arg": "02569",
    "putt": "02564",
}

_STAT_QUERY = (
    "query D($t:TourCode!,$s:String!,$y:Int){statDetails(tourCode:$t,statId:$s,year:$y)"
    "{statTitle rows{... on StatDetailsPlayer{playerId playerName rank "
    "stats{statName statValue}}}}}"
)


@dataclass
class PgaSgSnapshot:
    player: str  # canonical
    display_name: str
    sg_total: Optional[float] = None
    sg_ott: Optional[float] = None
    sg_app: Optional[float] = None
    sg_arg: Optional[float] = None
    sg_putt: Optional[float] = None
    measured_rounds: Optional[int] = None

    @property
    def has_categories(self) -> bool:
        return all(
            v is not None for v in (self.sg_ott, self.sg_app, self.sg_arg, self.sg_putt)
        )


def _num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_stat_rows(category_payload: dict) -> dict[str, tuple[Optional[float], Optional[int]]]:
    """One category's rows → {canonical: (avg_sg_per_round, measured_rounds)}."""
    out: dict[str, tuple[Optional[float], Optional[int]]] = {}
    rows = category_payload.get("rows", [])
    for r in rows:
        name = r.get("playerName")
        if not name:
            continue
        stats = {s.get("statName"): s.get("statValue") for s in (r.get("stats") or [])}
        avg = _num(stats.get("Avg"))
        rounds = _num(stats.get("Measured Rounds"))
        out[canonical(name)] = (avg, int(rounds) if rounds is not None else None)
    return out


def parse_sg_snapshot(multi_payload: dict) -> dict[str, PgaSgSnapshot]:
    """Combine the 5 category payloads (as saved in the fixture) into per-player
    SG snapshots keyed by canonical name.

    ``multi_payload`` shape: {category: {"rows": [...], ...}} for category in
    total/ott/app/arg/putt (missing categories are simply left None).
    """
    # Display names from the total category (fall back to any present).
    display: dict[str, str] = {}
    for cat in ("total", "ott", "app", "arg", "putt"):
        for r in multi_payload.get(cat, {}).get("rows", []):
            nm = r.get("playerName")
            if nm:
                display.setdefault(canonical(nm), nm)

    snaps: dict[str, PgaSgSnapshot] = {
        k: PgaSgSnapshot(player=k, display_name=v) for k, v in display.items()
    }
    for cat, attr in (("total", "sg_total"), ("ott", "sg_ott"), ("app", "sg_app"),
                      ("arg", "sg_arg"), ("putt", "sg_putt")):
        parsed = parse_stat_rows(multi_payload.get(cat, {}))
        for name, (avg, rounds) in parsed.items():
            snap = snaps.get(name)
            if snap is None:
                continue
            setattr(snap, attr, avg)
            if cat == "total" and rounds is not None:
                snap.measured_rounds = rounds
    return snaps


def _graphql(stat_id: str, year: int, tour: str, timeout: float) -> dict:
    body = json.dumps(
        {"query": _STAT_QUERY, "variables": {"t": tour, "s": stat_id, "y": year}}
    ).encode()
    req = urllib.request.Request(
        _GRAPHQL_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": _PUBLIC_API_KEY,
            "User-Agent": _UA,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        d = json.load(resp)
    return (d.get("data") or {}).get("statDetails") or {"rows": []}


def fetch_sg_snapshot(
    year: int, tour: str = "R", timeout: float = 25.0
) -> dict[str, PgaSgSnapshot]:
    """Live-fetch all 5 SG categories for a season and build player snapshots.

    ``tour`` is a PGA ``TourCode`` — "R" = PGA Tour. Makes 5 GraphQL calls.
    """
    multi: dict[str, dict] = {}
    for cat, sid in SG_STAT_IDS.items():
        try:
            multi[cat] = _graphql(sid, year, tour, timeout)
        except Exception as exc:  # pragma: no cover - network
            _log.warning("pga_sg_fetch_failed cat=%s err=%s", cat, exc)
            multi[cat] = {"rows": []}
    return parse_sg_snapshot(multi)
