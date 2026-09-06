"""Soccer league identity — the per-league dimension INSIDE the one `soccer` sector.

`soccer` stays a single sector (UCL/UEL pair clubs from different domestic
leagues, so Elo/Poisson/xG need one shared club pool, and the registry,
resolvers, alias maps and promotion gates are all sector-keyed). What differs
by league is pricing policy (tier sharp_weight, disagreement ramp) and
measurement (per-league CLV / Brier readouts). This module is the single
place that says WHICH league a market belongs to, so every consumer — the
coordinator's tier lookup, the logger's `league` column, the shadow CLI's
`--league` filter — agrees.

Canonical league keys are short lowercase tokens. Two venue-specific
spellings map onto them:

- Kalshi: the ticker SERIES prefix (`KXEPLGAME-26APR24MANCHE-CHE` → `epl`).
- Polymarket US: the league SLUG the market was fetched under
  (`/v2/leagues/{slug}/events`; `lal` → `laliga`). PolyUS market slugs also
  embed that slug as their SECOND dash token (`atc-mls-nas-atl-2026-07-17-nas`,
  `tsc-epl-bre-sun-2026-09-05-2pt5`), which is how pre-column PolyUS rows are
  backfilled. Only that token position is trusted — a team code can collide
  with a league slug (`sea` = Seattle AND Serie A).
"""

from __future__ import annotations

from typing import Optional

# Display order for per-league readouts (top-5 domestic, UEFA cups, MLS).
SOCCER_LEAGUES: tuple[str, ...] = (
    "epl", "laliga", "bundesliga", "seriea", "ligue1", "ucl", "uel", "mls",
    "ligamx", "jleague", "eredivisie", "brasileirao", "championship",
)

LEAGUE_DISPLAY: dict[str, str] = {
    "epl": "Premier League",
    "laliga": "La Liga",
    "bundesliga": "Bundesliga",
    "seriea": "Serie A",
    "ligue1": "Ligue 1",
    "ucl": "Champions League",
    "uel": "Europa League",
    "mls": "MLS",
    "ligamx": "Liga MX",
    "jleague": "J League",
    "eredivisie": "Eredivisie",
    "brasileirao": "Brasileirão",
    "championship": "Championship",
}

# Kalshi series prefix → league. Must stay in step with SECTOR_SERIES_MAP
# ["soccer"] in evmax/clients/kalshi.py (tests/test_soccer_leagues.py checks).
KALSHI_SERIES_LEAGUE: dict[str, str] = {
    "KXEPLGAME": "epl",
    "KXLALIGAGAME": "laliga",
    "KXBUNDESLIGAGAME": "bundesliga",
    "KXSERIEAGAME": "seriea",
    "KXLIGUE1GAME": "ligue1",
    "KXUCLGAME": "ucl",
    "KXUELGAME": "uel",
    "KXMLSGAME": "mls",
    "KXLIGAMXGAME": "ligamx",
    "KXJLEAGUEGAME": "jleague",
    "KXEREDIVISIEGAME": "eredivisie",
    "KXBRASILEIROGAME": "brasileirao",
    "KXEFLCHAMPIONSHIPGAME": "championship",
}

# Polymarket US league slug → league. Covers the betting map (epl/ucl/mls)
# and the extra arb-only slugs so an arb-scanner market still carries its
# league; slugs not listed (non-soccer sports) map to None.
POLYMARKET_US_SLUG_LEAGUE: dict[str, str] = {
    "epl": "epl",
    "ucl": "ucl",
    "mls": "mls",
    "lal": "laliga",
    "bun": "bundesliga",
    "sea": "seriea",
    "uefa": "uel",
    "lmx": "ligamx",
    "bra": "brasileirao",
    "eflch": "championship",
}


def _series_prefix(ticker: str) -> str:
    t = ticker.split(":", 1)[-1] if ":" in ticker else ticker
    return t.split("-", 1)[0].upper()


def league_for_ticker(ticker: Optional[str]) -> Optional[str]:
    """League for a Kalshi ticker (optionally `kalshi:`-prefixed), else None."""
    if not ticker:
        return None
    return KALSHI_SERIES_LEAGUE.get(_series_prefix(ticker))


def league_for_polymarket_slug(slug: Optional[str]) -> Optional[str]:
    """League for a Polymarket US league slug, else None."""
    if not slug:
        return None
    return POLYMARKET_US_SLUG_LEAGUE.get(slug.lower())


def league_for_polymarket_market_id(market_id: Optional[str]) -> Optional[str]:
    """League from a PolyUS market id's slug (`polymarket_us:{prefix}-{league}-...`).

    Reads ONLY the second dash token, so a team code that happens to equal a
    league slug elsewhere in the slug (`...-aus-sea-...`) can't mislead it.
    """
    if not market_id or not market_id.startswith("polymarket_us:"):
        return None
    slug = market_id.split(":", 2)[1]
    parts = slug.split("-")
    if len(parts) < 2:
        return None
    return league_for_polymarket_slug(parts[1])


def league_for_market_id(market_id: Optional[str]) -> Optional[str]:
    """Derive the league from a persisted market id.

    Kalshi ids (`kalshi:KXEPLGAME-...`, optionally `:no`-suffixed) resolve via
    the series prefix; Polymarket US ids via the slug's league token. This is
    the backfill path for rows logged before the `league` column existed.
    """
    if not market_id:
        return None
    if market_id.startswith("polymarket_us:"):
        return league_for_polymarket_market_id(market_id)
    return league_for_ticker(market_id)


# ESPN displayName → the name every other source uses, keyed by ESPN league
# slug. Needed where the SAME ESPN displayName denotes different clubs in
# different leagues: ESPN calls Liga MX's Santos Laguna plain "Santos", which
# normalizes to the canonical of Brazil's Santos FC (Pinnacle "Santos"). Seeds
# and the resolve-time feed would merge the two clubs into one Elo entity.
# Applied at every ESPN extraction site (resolver._fetch_espn_scores,
# scripts/seed_espn.py, scripts/seed_soccer_xg.py) via espn_display_name().
ESPN_DISPLAY_NAME_OVERRIDES: dict[str, dict[str, str]] = {
    "mex.1": {"Santos": "Santos Laguna"},
}


def espn_display_name(league_slug: Optional[str], display_name: str) -> str:
    """Resolve an ESPN displayName to its cross-source name for a league."""
    if not league_slug or not display_name:
        return display_name
    return ESPN_DISPLAY_NAME_OVERRIDES.get(league_slug, {}).get(display_name, display_name)


def is_known_league(league: Optional[str]) -> bool:
    return bool(league) and league in SOCCER_LEAGUES
