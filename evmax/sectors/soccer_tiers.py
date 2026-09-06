"""Soccer league tier lookup — maps Kalshi ticker → sharp_weight override.

The walk-forward backtest showed that in top-5 European leagues Pinnacle's
closing Brier is the informational ceiling. Mixing our stat models at the
default sharp_weight=0.40 drags the blend 0.003–0.007 Brier worse than pure
sharp. This module reads data/soccer_league_tiers.yaml and returns the
per-league sharp_weight the coordinator should apply.

Tickers are matched by their Kalshi series prefix (the portion before the
first `-`), which identifies the league unambiguously.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from evmax.sectors.soccer_leagues import KALSHI_SERIES_LEAGUE, league_for_ticker

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "soccer_league_tiers.yaml"


@lru_cache(maxsize=1)
def _load_tiers() -> dict:
    """Read and cache the tier config. Raises if the file is missing or
    malformed — there's no sensible silent default for a pricing decision."""
    if not _CONFIG_PATH.exists():
        return {"default_sharp_weight": 0.40, "tiers": {}}
    with _CONFIG_PATH.open("r") as f:
        data = yaml.safe_load(f) or {}
    return data


def _tier_leagues(tier: dict) -> set[str]:
    """Leagues a tier covers: explicit `leagues:` plus any `kalshi_series:`
    mapped through KALSHI_SERIES_LEAGUE (legacy config shape)."""
    leagues = {str(lg).lower() for lg in (tier.get("leagues") or [])}
    for series in tier.get("kalshi_series") or []:
        lg = KALSHI_SERIES_LEAGUE.get(str(series).upper())
        if lg:
            leagues.add(lg)
    return leagues


@lru_cache(maxsize=1)
def _league_to_weight() -> dict[str, float]:
    """Flatten tiers into league → sharp_weight for fast lookup."""
    cfg = _load_tiers()
    mapping: dict[str, float] = {}
    for tier_name, tier in (cfg.get("tiers") or {}).items():
        weight = float(tier.get("sharp_weight", cfg.get("default_sharp_weight", 0.40)))
        for lg in _tier_leagues(tier):
            mapping[lg] = weight
    return mapping


@lru_cache(maxsize=1)
def _league_to_ramp() -> dict[str, tuple[float, float, float]]:
    """Flatten tiers into league → (threshold, saturate_at, cap) for the
    ensemble's disagreement ramp. Tiers without `disagreement_ramp` are absent
    from the map, so their events fall through to the sector-level override in
    EnsembleModelAgent.DISAGREEMENT_OVERRIDES (unchanged behaviour)."""
    cfg = _load_tiers()
    mapping: dict[str, tuple[float, float, float]] = {}
    for tier_name, tier in (cfg.get("tiers") or {}).items():
        ramp = tier.get("disagreement_ramp")
        if not ramp:
            continue
        try:
            params = (
                float(ramp["threshold"]),
                float(ramp["saturate_at"]),
                float(ramp["cap"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"soccer_league_tiers.yaml tier {tier_name!r}: disagreement_ramp "
                f"needs threshold/saturate_at/cap floats ({e})"
            ) from e
        if not (0.0 <= params[0] <= params[1] and 0.0 <= params[2] <= 1.0):
            raise ValueError(
                f"soccer_league_tiers.yaml tier {tier_name!r}: disagreement_ramp "
                f"must satisfy 0 <= threshold <= saturate_at and 0 <= cap <= 1, got {params}"
            )
        for lg in _tier_leagues(tier):
            mapping[lg] = params
    return mapping


def _series_to_weight() -> dict[str, float]:
    """Series-prefix view of the league map (kept for callers/tests that
    think in Kalshi series)."""
    by_league = _league_to_weight()
    return {
        series: by_league[lg]
        for series, lg in KALSHI_SERIES_LEAGUE.items()
        if lg in by_league
    }


@lru_cache(maxsize=1)
def _shadow_leagues() -> frozenset[str]:
    """Leagues listed under `shadow_leagues:` — league-level shadow inside the
    soccer sector (mode is per SECTOR, so a newly wired league would otherwise
    be live on its first scan). See the YAML header for the semantics."""
    cfg = _load_tiers()
    return frozenset(str(lg).lower() for lg in (cfg.get("shadow_leagues") or []))


def league_is_live(league: Optional[str]) -> bool:
    """False when `league` is on the tier config's shadow list.

    A gap with no league (single-league sectors, or a soccer market whose
    league could not be derived) is NOT held back — the list only ever
    restricts leagues that are explicitly named.
    """
    if not league:
        return True
    return league.lower() not in _shadow_leagues()


def shadow_leagues() -> frozenset[str]:
    return _shadow_leagues()


def default_sharp_weight() -> float:
    """Fallback weight used when a ticker doesn't map to any tier."""
    return float(_load_tiers().get("default_sharp_weight", 0.40))


def sharp_weight_for_league(league: Optional[str]) -> float:
    """Return the tier sharp_weight for a canonical league key.

    Returns `default_sharp_weight()` when the league is None/unknown. This is
    the venue-agnostic lookup: a Polymarket US market has no Kalshi ticker, so
    before the league dimension existed PolyUS EPL/UCL games silently fell to
    the 0.40 default meant for MLS.
    """
    if not league:
        return default_sharp_weight()
    return _league_to_weight().get(league.lower(), default_sharp_weight())


def sharp_weight_for_ticker(ticker: Optional[str]) -> float:
    """Return the sharp_weight for a Kalshi ticker's series prefix.

    Tickers look like `KXEPLGAME-26APR24MANCHE-CHE`. The series prefix is
    everything before the first `-`. Returns `default_sharp_weight()` when
    the ticker is empty or the series isn't configured.
    """
    return sharp_weight_for_league(league_for_ticker(ticker))


def sharp_weight_for_market(market) -> float:
    """Tier sharp_weight for a PredictionMarket: league first (works for every
    venue), ticker series as the fallback for markets that predate `league`."""
    league = getattr(market, "league", None)
    if league:
        return sharp_weight_for_league(league)
    return sharp_weight_for_ticker(getattr(market, "ticker", None))


def disagreement_ramp_for_league(
    league: Optional[str],
) -> Optional[tuple[float, float, float]]:
    """(threshold, saturate_at, cap) for the league's tier, or None when the
    tier doesn't configure one (→ sector-level ramp applies)."""
    if not league:
        return None
    return _league_to_ramp().get(league.lower())


def disagreement_ramp_for_market(market) -> Optional[tuple[float, float, float]]:
    league = getattr(market, "league", None) or league_for_ticker(
        getattr(market, "ticker", None)
    )
    return disagreement_ramp_for_league(league)


def reset_cache() -> None:
    """Clear caches. Useful in tests when the YAML is monkey-patched."""
    _load_tiers.cache_clear()
    _league_to_weight.cache_clear()
    _league_to_ramp.cache_clear()
    _shadow_leagues.cache_clear()


def remove_shadow_league(text: str, league: str) -> tuple[str, bool]:
    """Drop `league` from the `shadow_leagues:` list in the tier YAML TEXT.

    Line-based (like the categories.yaml mode flip) so comments and layout
    survive. Returns (new_text, removed). Only the `- league` item lines that
    sit under the `shadow_leagues:` key are touched.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_block = False
    removed = False
    for line in lines:
        stripped = line.strip()
        if not line.startswith((" ", "\t", "-")) and stripped and not stripped.startswith("#"):
            in_block = stripped.startswith("shadow_leagues:")
            out.append(line)
            continue
        if in_block and stripped.startswith("-"):
            item = stripped[1:].split("#", 1)[0].strip().strip("'\"")
            if item.lower() == league.lower():
                removed = True
                continue
        out.append(line)
    return "".join(out), removed
