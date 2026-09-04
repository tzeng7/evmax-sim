"""ESPN Football Power Index (FPI) for college football — preseason-prior source
for ``ncaaf_efficiency_v2``.

Why FPI: the NCAAF market's edge over an EPA-only model is concentrated in
weeks 0–8, the stretch where the model's rating is mostly its PRIOR. ESPN
play-by-play carries no offseason information (portal, draft, coaching
changes, recruiting); FPI's preseason rating does. Blending a regressed
prior-season EPA rating 50/50 with preseason FPI (both in EPA/play units) was
the only prior variant that beat the regressed-EPA prior on every held-out
season (2023/2024/2025 walk-forward, see scripts/backtest_ncaaf_v2.py) — a
market-implied prior and a two-season EPA prior both lost.

Two sources, one parser:

  * LIVE (seed time): ``site.web.api.espn.com/apis/fitt/v3/sports/football/
    college-football/powerindex?season=Y`` — no key, all ~136 FBS teams, one
    request. Used by scripts/seed_ncaaf_efficiency.py, which FREEZES the first
    fetch of a season as the preseason prior (a mid-season refetch would leak
    ESPN's in-season updates into what is meant to be a preseason prior).
  * HISTORICAL (backtest): the Wayback Machine keeps August snapshots of
    ``espn.com/college-football/fpi``; the page embeds the same table as a
    ``window['__espnfitt__']`` JSON blob. scripts/build_ncaaf_fpi_history.py
    turns those into data/backtest/ncaaf_fpi/fpi_{season}.json.

FPI is a points-vs-average scale (≈ +30 elite … −25 bottom). Consumers
centre it on the FBS mean and divide by plays-per-game to get EPA/play.

Everything is fail-soft: any fetch/parse failure returns ``{}`` and the seed
falls back to the EPA-only prior (v1 behaviour).
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

FPI_LIVE_URL = (
    "https://site.web.api.espn.com/apis/fitt/v3/sports/football/college-football/powerindex"
)
FPI_PAGE_URL = "https://www.espn.com/college-football/fpi"
WAYBACK_AVAILABLE = "http://archive.org/wayback/available"

_FITT_BLOB_RE = re.compile(r"window\['__espnfitt__'\]=(\{.*?\});?\s*</script>", re.S)


def _fpi_from_categories(categories: list[dict]) -> Optional[float]:
    """Live endpoint shape: ``categories[{name:'fpi', values:[fpi, rank, ...]}]``."""
    for cat in categories or []:
        if (cat.get("name") or "").lower() == "fpi":
            vals = cat.get("values") or []
            if vals:
                try:
                    return float(vals[0])
                except (TypeError, ValueError):
                    return None
    return None


def parse_live_response(data: dict) -> dict[str, dict]:
    """``{espn_team_id: {"fpi": float, "name": str}}`` from the fitt powerindex JSON."""
    out: dict[str, dict] = {}
    for row in data.get("teams") or []:
        team = row.get("team") or {}
        tid = str(team.get("id") or "")
        fpi = _fpi_from_categories(row.get("categories") or [])
        if not tid or fpi is None:
            continue
        out[tid] = {"fpi": fpi, "name": team.get("displayName") or team.get("location") or ""}
    return out


def parse_fitt_page(html: str) -> dict[str, dict]:
    """``{espn_team_id: {"fpi": float, "name": str}}`` from the FPI web page.

    The page (and its Wayback snapshots) embeds the table as
    ``page.content.table.stats[] = {team:{id,displayName}, stats:[{name,value}]}``.
    Returns ``{}`` when the blob is missing or malformed.
    """
    m = _FITT_BLOB_RE.search(html or "")
    if not m:
        return {}
    try:
        blob = json.loads(m.group(1))
        rows = blob["page"]["content"]["table"]["stats"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}
    out: dict[str, dict] = {}
    for row in rows or []:
        team = row.get("team") or {}
        tid = str(team.get("id") or "")
        val = next((s.get("value") for s in row.get("stats") or [] if s.get("name") == "fpi"), None)
        if not tid or val is None:
            continue
        try:
            out[tid] = {"fpi": float(val), "name": team.get("displayName") or ""}
        except (TypeError, ValueError):
            continue
    return out


def fetch_fpi(season: int, client: Optional[httpx.Client] = None, timeout: float = 30.0) -> dict[str, dict]:
    """Current FPI for every FBS team in ``season`` (fail-soft → ``{}``)."""
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.get(
            FPI_LIVE_URL,
            params={"region": "us", "lang": "en", "season": season, "limit": 1000},
            timeout=timeout,
        )
        r.raise_for_status()
        out = parse_live_response(r.json())
    except Exception as e:  # noqa: BLE001
        logger.warning("cfb_fpi_fetch_fail", season=season, error=str(e))
        return {}
    finally:
        if own:
            client.close()
    logger.info("cfb_fpi_fetched", season=season, teams=len(out))
    return out


def wayback_snapshot_url(timestamp: str, client: Optional[httpx.Client] = None) -> Optional[str]:
    """Closest Wayback snapshot of the FPI page to ``timestamp`` (YYYYMMDD[hhmmss])."""
    own = client is None
    client = client or httpx.Client()
    try:
        r = client.get(WAYBACK_AVAILABLE, params={"url": FPI_PAGE_URL, "timestamp": timestamp}, timeout=30)
        r.raise_for_status()
        closest = (r.json().get("archived_snapshots") or {}).get("closest") or {}
        return closest.get("url") if closest.get("available") else None
    except Exception as e:  # noqa: BLE001
        logger.warning("cfb_fpi_wayback_lookup_fail", timestamp=timestamp, error=str(e))
        return None
    finally:
        if own:
            client.close()


def fetch_preseason_fpi_wayback(timestamp: str, client: Optional[httpx.Client] = None) -> tuple[dict[str, dict], Optional[str]]:
    """Preseason FPI from the Wayback snapshot closest to ``timestamp``.

    Returns ``(ratings, snapshot_url)``; ``({}, None)`` on any failure. Callers
    should pick a timestamp BEFORE the season's Week-0 kickoff so the rating is
    a genuine preseason projection (the page's ``lastUpdated`` is logged).
    """
    own = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        url = wayback_snapshot_url(timestamp, client)
        if not url:
            return {}, None
        r = client.get(url, timeout=120)
        r.raise_for_status()
        out = parse_fitt_page(r.text)
        logger.info("cfb_fpi_wayback", timestamp=timestamp, url=url, teams=len(out))
        return out, url
    except Exception as e:  # noqa: BLE001
        logger.warning("cfb_fpi_wayback_fail", timestamp=timestamp, error=str(e))
        return {}, None
    finally:
        if own:
            client.close()


def centre_fpi(ratings: dict[str, dict], team_ids) -> dict[str, float]:
    """FPI points relative to the mean over ``team_ids`` that have a rating.

    Teams in ``team_ids`` without an FPI row are omitted (the consumer falls
    back to its EPA-only prior for them) — never imputed.
    """
    vals = [ratings[t]["fpi"] for t in team_ids if t in ratings]
    if not vals:
        return {}
    mean = sum(vals) / len(vals)
    return {t: round(ratings[t]["fpi"] - mean, 3) for t in team_ids if t in ratings}
