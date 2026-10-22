"""Read-only explanation of effective modes: WHAT each (category, market type,
venue) resolves to and WHY.

Every mode value comes from ``evmax.modes.get_mode`` and the venue firewall from
``settings.venue_sector_live``; this module never re-implements resolution. The
reason is attributed by diffing successive ``get_mode`` calls (sector-level ->
market-type -> venue), so it cannot drift from the real rules.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from evmax.categories import all_categories
from evmax.modes import get_mode, override_source

# Reason codes, strongest layer first.
RUNTIME_OVERRIDE = "runtime_override"
ENV_OVERRIDE = "env_override"
OFF_SEASON = "off_season_downgrade"
MARKET_TYPE_DISABLED = "market_type_disabled"
MARKET_TYPE_SHADOW = "market_type_shadow"
VENUE_SHADOW = "venue_market_type_shadow"
VENUE_FIREWALL = "venue_firewall"
YAML_BASE = "yaml_base"


def _venues_for(spec: Any) -> list[str]:
    from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP

    venues = ["kalshi"]
    if not spec.is_prop and spec.key in POLYMARKET_US_LEAGUE_MAP:
        venues.append("polymarket_us")
    return venues


def explain_mode(
    spec: Any, market_type: str, venue: str, today: Optional[date] = None
) -> dict[str, Any]:
    """Effective mode + reason for one (category, market_type, venue)."""
    from evmax.settings import get_settings

    key = spec.key
    sector_mode = get_mode(key, today=today)
    type_mode = get_mode(key, market_type, today=today)
    venue_mode = get_mode(key, market_type, today=today, venue=venue)

    src = override_source(key)
    if src:
        reason = RUNTIME_OVERRIDE if src == "runtime" else ENV_OVERRIDE
    elif sector_mode != spec.mode:
        reason = OFF_SEASON
    elif type_mode != sector_mode:
        reason = MARKET_TYPE_DISABLED if type_mode == "disabled" else MARKET_TYPE_SHADOW
    elif venue_mode != type_mode:
        reason = VENUE_SHADOW
    else:
        reason = YAML_BASE

    final = venue_mode
    firewall_closed = venue_mode == "live" and not get_settings().venue_sector_live(venue, key)
    if firewall_closed:
        final = "shadow"
        reason = VENUE_FIREWALL
    return {
        "category": key,
        "display_name": spec.display_name,
        "market_type": market_type,
        "venue": venue,
        "yaml_mode": spec.mode,
        "mode": final,
        "reason": reason,
    }


def mode_report(today: Optional[date] = None) -> dict[str, Any]:
    """Full report: one row per (category, market_type, venue) plus context."""
    from evmax.sectors.soccer_tiers import shadow_leagues
    from evmax.settings import get_settings

    cfg = get_settings()
    rows: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    for spec in all_categories():
        in_season = spec.is_in_season(today)
        for mt in spec.market_types:
            for venue in _venues_for(spec):
                rows.append(explain_mode(spec, mt.value, venue, today))
        categories.append({
            "key": spec.key,
            "display_name": spec.display_name,
            "yaml_mode": spec.mode,
            "mode": get_mode(spec.key, today=today),
            "override": override_source(spec.key),
            "in_season": in_season,
            "season_window": spec.season_window is not None,
        })
    return {
        "today": (today or date.today()).isoformat(),
        "categories": categories,
        "rows": rows,
        "shadow_leagues": sorted(shadow_leagues()),
        "polymarket_us": {
            "enabled": cfg.polymarket_us_enabled,
            "master_live": cfg.polymarket_us_live,
            "live_sectors": sorted(cfg.polymarket_us_live_sector_set()),
        },
    }
