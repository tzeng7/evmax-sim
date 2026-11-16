"""Soft-book anchor for sectors the sharp book does not post.

Pinnacle posts no women's college basketball outside March, so ``ncaaw`` has
no sharp anchor. This client supplies a STAND-IN anchor from another book's
line, in two tiers:

  1. **The Odds API** (``basketball_wncaab``) when ``ODDS_API_KEY`` is set.
     One request returns moneyline, spread and total for every listed game.
     Books are taken in ``ODDS_API_BOOK_PREFERENCE`` order (sharpest first).
     Each request costs ``markets x regions`` credits, so responses are cached
     on disk for ``soft_anchor_cache_ttl_secs`` (default 1 hour).
  2. **ESPN's scoreboard** (free, no key). It carries one retail book's line
     (DraftKings) inline, for a minority of games and only near game day
     (2 of 33 games on 2026-11-16). One request per calendar day.

Every record carries the book it came from (never ``pinnacle``). The
coordinator tags every row priced off one with ``soft_anchor``;
``has_full_blend`` rejects that token, so these rows log in shadow and are
never sized. A retail line is NOT a sharp price: the row's "EV" is
venue-vs-retail-book disagreement. Judge the lane on venue CLV and realized
Brier (``cleanup shadow clv <sector> --sources-token soft_anchor``).

Record shapes mirror ``PinnacleGuestClient`` exactly (event ids, the spread
record's outcome_a = the team laying points, totals keyed ``::total::<line>``)
so matching, pricing and resolution need no special case.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from evmax.clients.time_util import kalshi_game_day
from evmax.ev.devig import american_to_decimal, devig_two_way
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds

logger = structlog.get_logger(__name__)

SOFT_ANCHOR_SECTORS: frozenset[str] = frozenset({"ncaaw"})
SOFT_ANCHOR_TOKEN = "soft_anchor"

# sector -> The Odds API sport key
ODDS_API_SPORT: dict[str, str] = {"ncaaw": "basketball_wncaab"}
# sector -> ESPN scoreboard path + query
ESPN_SCOREBOARD: dict[str, tuple[str, dict[str, Any]]] = {
    "ncaaw": ("basketball/womens-college-basketball", {"groups": 50, "limit": 500}),
}
# Sharpest first. A book missing from SharpBook is ignored.
ODDS_API_BOOK_PREFERENCE: tuple[str, ...] = (
    "pinnacle", "betonlineag", "lowvig", "draftkings", "fanduel", "betmgm",
    "williamhill", "betrivers", "bovada",
)
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
_ESPN_HTTP_UA = "curl/8.7.1"
_ESPN_BOOKS = {"draftkings": SharpBook.draftkings, "fanduel": SharpBook.fanduel,
               "betmgm": SharpBook.betmgm}
_CACHE_DIR = Path("data/cache/soft_anchor")


def soft_anchor_enabled(sector: Optional[str]) -> bool:
    return (sector or "").lower() in SOFT_ANCHOR_SECTORS


# ---------------------------------------------------------------------------
# shared record builders
# ---------------------------------------------------------------------------

def _norm(name: str, sector: str) -> str:
    """Canonical team key. A full "School Mascot" name that the alias map does
    not know is retried without its last one or two words (the mascot)."""
    from evmax.matching.normalizer import NameNormalizer
    from evmax.sectors.registry import get_handler

    normalizer = NameNormalizer(sector)
    try:
        handler = get_handler(sector)
    except Exception:  # noqa: BLE001
        handler = None
    words = (name or "").split()
    first = normalizer.normalize(name or "")
    if handler is None or handler.is_canonical(first):
        return first
    for drop in (1, 2):
        if len(words) - drop < 1:
            break
        cand = normalizer.normalize(" ".join(words[:-drop]))
        if handler.is_canonical(cand):
            return cand
    return first


def _base_event_id(sector: str, home: str, away: str, event_date: Optional[datetime]) -> str:
    h = _norm(home, sector).replace(" ", "_").replace(".", "")
    a = _norm(away, sector).replace(" ", "_").replace(".", "")
    return f"{sector}::{kalshi_game_day(event_date, sector)}::{h}_vs_{a}"


def _two_way(dec_a: Optional[float], dec_b: Optional[float]) -> Optional[tuple[float, float, float]]:
    if not dec_a or not dec_b or dec_a <= 1.0 or dec_b <= 1.0:
        return None
    try:
        return devig_two_way(dec_a, dec_b)
    except Exception:  # noqa: BLE001
        return None


def build_records(
    sector: str, book: SharpBook, home: str, away: str, event_date: Optional[datetime],
    *, ml: Optional[tuple[float, float]] = None,
    spread: Optional[tuple[float, float, float]] = None,
    total: Optional[tuple[float, float, float]] = None,
) -> list[SharpOdds]:
    """Records for one game from DECIMAL prices.

    ml     = (home decimal, away decimal)
    spread = (HOME handicap, home decimal, away decimal)
    total  = (line, over decimal, under decimal)
    """
    base = _base_event_id(sector, home, away, event_date)
    out: list[SharpOdds] = []
    if ml:
        dv = _two_way(ml[0], ml[1])
        if dv:
            out.append(SharpOdds(
                event_id=base, book=book, sector=sector,
                outcome_a_label=home, outcome_b_label=away,
                outcome_a_decimal=ml[0], outcome_b_decimal=ml[1],
                true_prob_a=dv[0], true_prob_b=dv[1], margin=dv[2], event_date=event_date,
            ))
    if spread and spread[0] is not None and spread[0] != 0:
        hdp, home_dec, away_dec = spread
        # outcome_a = the team laying points (Pinnacle client convention).
        if hdp < 0:
            cover, other, c_dec, o_dec, point = home, away, home_dec, away_dec, float(hdp)
        else:
            cover, other, c_dec, o_dec, point = away, home, away_dec, home_dec, -float(hdp)
        dv = _two_way(c_dec, o_dec)
        if dv:
            out.append(SharpOdds(
                event_id=f"{base}::spread", book=book, sector=sector,
                outcome_a_label=cover, outcome_b_label=other,
                outcome_a_decimal=c_dec, outcome_b_decimal=o_dec,
                true_prob_a=dv[0], true_prob_b=dv[1], margin=dv[2],
                spread_line=point, event_date=event_date,
            ))
    if total and total[0] and total[0] > 0:
        line, over_dec, under_dec = total
        dv = _two_way(over_dec, under_dec)
        if dv:
            out.append(SharpOdds(
                event_id=f"{base}::total::{float(line)}", book=book, sector=sector,
                outcome_a_label="over", outcome_b_label="under",
                outcome_a_decimal=over_dec, outcome_b_decimal=under_dec,
                true_prob_a=0.0, true_prob_b=0.0,
                true_prob_over=dv[0], true_prob_under=dv[1], margin=dv[2],
                total_line=float(line), event_date=event_date,
            ))
    return out


def _parse_dt(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The Odds API
# ---------------------------------------------------------------------------

def parse_odds_api_event(event: dict, sector: str, now: Optional[datetime] = None,
                         names: Optional[dict[str, str]] = None) -> list[SharpOdds]:
    """Records for one Odds API event from the most-preferred book that has it.

    Each market (h2h / spreads / totals) is taken from the first book in
    ``ODDS_API_BOOK_PREFERENCE`` that quotes it, so a game still gets a total
    when the sharpest book only posts a moneyline. A started game is skipped.
    """
    home, away = event.get("home_team"), event.get("away_team")
    start = _parse_dt(event.get("commence_time"))
    if not home or not away or start is None:
        return []
    if start <= (now or datetime.now(timezone.utc)):
        return []
    # The feed names teams "School Mascot"; ``names`` (ESPN displayName ->
    # location, from the same day's scoreboard) gives the school name the alias
    # maps and the venue use. Prices are still looked up under the feed's names.
    label_home = (names or {}).get(home, home)
    label_away = (names or {}).get(away, away)
    by_book = {b.get("key"): b for b in event.get("bookmakers") or []}
    out: list[SharpOdds] = []
    done: set[str] = set()
    for key in ODDS_API_BOOK_PREFERENCE:
        bm = by_book.get(key)
        if bm is None or key not in SharpBook.__members__:
            continue
        book = SharpBook[key]
        for market in bm.get("markets") or []:
            mkey = market.get("key")
            if mkey in done:
                continue
            oc = market.get("outcomes") or []
            price = {o.get("name"): o for o in oc}
            recs: list[SharpOdds] = []
            try:
                if mkey == "h2h" and home in price and away in price:
                    recs = build_records(sector, book, label_home, label_away, start,
                                         ml=(float(price[home]["price"]), float(price[away]["price"])))
                elif mkey == "spreads" and home in price and away in price:
                    recs = build_records(sector, book, label_home, label_away, start, spread=(
                        float(price[home]["point"]), float(price[home]["price"]),
                        float(price[away]["price"])))
                elif mkey == "totals" and "Over" in price and "Under" in price:
                    recs = build_records(sector, book, label_home, label_away, start, total=(
                        float(price["Over"]["point"]), float(price["Over"]["price"]),
                        float(price["Under"]["price"])))
            except (KeyError, TypeError, ValueError):
                recs = []
            if recs:
                out.extend(recs)
                done.add(mkey)
    return out


async def _fetch_odds_api(sector: str, api_key: str, ttl: int) -> Optional[list[dict]]:
    sport = ODDS_API_SPORT.get(sector)
    if not sport:
        return None
    cache = _CACHE_DIR / f"odds_api_{sport}.json"
    if ttl > 0 and cache.exists() and time.time() - cache.stat().st_mtime < ttl:
        try:
            return json.loads(cache.read_text())
        except (OSError, ValueError):
            pass
    import httpx

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(ODDS_API_URL.format(sport=sport), params={
                "apiKey": api_key, "regions": "us,eu", "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal", "dateFormat": "iso",
            })
        if r.status_code != 200:
            logger.warning("soft_anchor_odds_api_error", sector=sector, status=r.status_code)
            return None
        data = r.json()
        logger.info("soft_anchor_odds_api_fetched", sector=sector, events=len(data),
                    credits_remaining=r.headers.get("x-requests-remaining"))
    except Exception as e:  # noqa: BLE001
        logger.warning("soft_anchor_odds_api_failed", sector=sector, error=str(e))
        return None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    except OSError:
        pass
    return data


# ---------------------------------------------------------------------------
# ESPN scoreboard
# ---------------------------------------------------------------------------

def _american(value: Any) -> Optional[float]:
    """ESPN price string ('-110', '+250', 'EVEN', 'OFF') -> decimal, else None."""
    s = str(value if value is not None else "").strip().upper()
    if s in ("EVEN", "EV"):
        return 2.0
    try:
        n = int(float(s))
    except ValueError:
        return None
    if n == 0 or abs(n) < 100:
        return None
    try:
        return american_to_decimal(n)
    except Exception:  # noqa: BLE001
        return None


def _close(node: Any, side: str) -> dict:
    side_node = (node or {}).get(side) or {}
    return side_node.get("close") or side_node.get("current") or side_node.get("open") or {}


def parse_espn_event(event: dict, sector: str,
                     now: Optional[datetime] = None) -> list[SharpOdds]:
    """Records for one ESPN scoreboard event that carries an inline line.

    Only a game still in the pre state is used: once it starts ESPN keeps
    showing the frozen (or a live) line.
    """
    comp = (event.get("competitions") or [{}])[0]
    state = (((event.get("status") or {}).get("type") or {}).get("state") or "").lower()
    start = _parse_dt(event.get("date"))
    if state != "pre" or start is None or start <= (now or datetime.now(timezone.utc)):
        return []
    teams = {c.get("homeAway"): ((c.get("team") or {}).get("location")
                                 or (c.get("team") or {}).get("displayName"))
             for c in comp.get("competitors") or []}
    home, away = teams.get("home"), teams.get("away")
    if not home or not away:
        return []
    for item in comp.get("odds") or []:
        provider = str((item.get("provider") or {}).get("name") or "")
        if "live" in provider.lower():
            continue
        book = _ESPN_BOOKS.get(provider.lower().replace(" ", ""))
        if book is None:
            continue
        ml = spread = total = None
        h, a = _american(_close(item.get("moneyline"), "home").get("odds")), \
            _american(_close(item.get("moneyline"), "away").get("odds"))
        if h and a:
            ml = (h, a)
        ps_h, ps_a = _close(item.get("pointSpread"), "home"), _close(item.get("pointSpread"), "away")
        try:
            hdp = float(str(ps_h.get("line")).replace("+", ""))
        except (TypeError, ValueError):
            hdp = None
        if hdp is not None and _american(ps_h.get("odds")) and _american(ps_a.get("odds")):
            spread = (hdp, _american(ps_h.get("odds")), _american(ps_a.get("odds")))
        t_o, t_u = _close(item.get("total"), "over"), _close(item.get("total"), "under")
        try:
            line = float(item.get("overUnder"))
        except (TypeError, ValueError):
            line = None
        if line and _american(t_o.get("odds")) and _american(t_u.get("odds")):
            total = (line, _american(t_o.get("odds")), _american(t_u.get("odds")))
        recs = build_records(sector, book, home, away, start, ml=ml, spread=spread, total=total)
        if recs:
            return recs
    return []


def espn_display_names(events: list[dict]) -> dict[str, str]:
    """{ESPN displayName ("Troy Trojans"): location ("Troy")} for a day's games."""
    out: dict[str, str] = {}
    for event in events:
        for c in ((event.get("competitions") or [{}])[0].get("competitors") or []):
            team = c.get("team") or {}
            if team.get("displayName") and team.get("location"):
                out[team["displayName"]] = team["location"]
    return out


async def _fetch_espn(sector: str, days: list[str]) -> list[dict]:
    spec = ESPN_SCOREBOARD.get(sector)
    if not spec:
        return []
    import httpx

    path, extra = spec
    events: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _ESPN_HTTP_UA}) as client:
            for day in days:
                r = await client.get(ESPN_URL.format(path=path), params={**extra, "dates": day})
                if r.status_code == 200:
                    events.extend(r.json().get("events") or [])
    except Exception as e:  # noqa: BLE001
        logger.warning("soft_anchor_espn_failed", sector=sector, error=str(e))
    return events


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _market_days(markets: list[PredictionMarket], sector: str) -> list[str]:
    """ESPN `dates` values (YYYYMMDD, ET game day) covering the listed markets,
    today and tomorrow only — ESPN shows no line further out."""
    now = datetime.now(timezone.utc)
    allowed = {kalshi_game_day(now + timedelta(days=d), sector) for d in (0, 1)}
    days = {kalshi_game_day(m.event_date, sector) for m in markets if m.event_date is not None}
    return sorted(d.replace("-", "") for d in (days & allowed))


async def soft_anchor_odds(sector: str, markets: list[PredictionMarket]) -> list[SharpOdds]:
    """Stand-in anchor records for ``sector``. Any failure returns []."""
    if not soft_anchor_enabled(sector) or not markets:
        return []
    from evmax.settings import get_settings

    settings = get_settings()
    out: list[SharpOdds] = []
    source = "none"
    key = (getattr(settings, "odds_api_key", "") or "").strip()
    if key:
        data = await _fetch_odds_api(sector, key, int(getattr(settings, "soft_anchor_cache_ttl_secs", 3600)))
        names = espn_display_names(await _fetch_espn(sector, _market_days(markets, sector))) if data else {}
        for event in data or []:
            out.extend(parse_odds_api_event(event, sector, names=names))
        if out:
            source = "odds_api"
    if not out:
        for event in await _fetch_espn(sector, _market_days(markets, sector)):
            out.extend(parse_espn_event(event, sector))
        if out:
            source = "espn"
    # One record per event id (a rematch listed twice keeps the first).
    seen: set[str] = set()
    unique = [r for r in out if not (r.event_id in seen or seen.add(r.event_id))]
    logger.info("soft_anchor_fetched", sector=sector, source=source, records=len(unique),
                games=len({r.event_id.split("::spread")[0].split("::total")[0] for r in unique}))
    return unique
