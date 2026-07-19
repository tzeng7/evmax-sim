"""Bullpen quality + availability — shared by the live pitcher agent and the
ESPN walk-forward backtest.

Reliever quality + availability are the two signals that move baseball lines
past the static-quality info Pinnacle already prices. Static quality (a
team's season bullpen FIP) is already in sharp lines — the edge is
*day-to-day availability*: a closer who threw 30 pitches in back-to-back
games likely isn't available today, but the line may not yet reflect that
until the lineup card / late news drops.

Extracted verbatim from evmax/backtest/sources/espn_walkforward.py
(2026-07-19, WS2a) so the live agent and the backtest run ONE
implementation — the walk-forward imports from here.

Data shapes (shared by both consumers):
  reliever stats:  {"ip": float, "er": int, "bb": int, "so": int, "hr": int, ...}
  appearances:     list of (date, ip, pitch_count) tuples, chronological
  reliever_team:   {reliever_name: team_name}
"""

from __future__ import annotations

from datetime import date
from typing import Optional

PEN_MIN_IP_FOR_PRED = 30.0  # team-aggregate IP floor — below this we don't have
                            # enough sample for the bullpen to contribute.
PEN_BLEND_STARTER_SHARE = 0.60  # starter handles ~5.5 of 9 IP on average
PEN_BLEND_RELIEVER_SHARE = 0.40
TOP_N_AVAILABLE_PEN = 3  # rank available relievers by FIP, take the top N
                         # as the "high-leverage" available pool. Mirrors
                         # closer + setup + 1 high-leverage reliever.
FATIGUE_PC_THRESHOLD = 25  # threw 25+ pitches yesterday → unavailable
RELIEVER_MIN_IP = 5.0  # per-reliever IP floor to enter the quality pool


def is_reliever_available(
    appearances: list[tuple],  # list of (date, ip, pitch_count)
    today: date,
) -> bool:
    """Fatigue heuristic mirroring how MLB managers actually deploy relievers:

    - Threw yesterday with 25+ pitches → unavailable today (full rest needed)
    - Threw 2 of last 3 days (any usage) → unavailable today
    - Otherwise → available

    Note this is the rule-of-thumb version. Real availability also reflects
    handedness matchups, leverage, and manager preference — out of scope for
    public-data modeling. The 25-pitch threshold matches MLB's typical
    "15+ pitches → no back-to-back" line plus a buffer.
    """
    if not appearances:
        return True

    last_three: list[tuple] = []
    for app in reversed(appearances):
        days_ago = (today - app[0]).days
        if days_ago > 3:
            break
        if days_ago > 0:
            last_three.append(app)

    yesterday = [a for a in last_three if (today - a[0]).days == 1]
    if yesterday and yesterday[0][2] >= FATIGUE_PC_THRESHOLD:
        return False

    used_in_last_three = sum(1 for a in last_three if (today - a[0]).days <= 3)
    if used_in_last_three >= 2:
        return False

    return True


def team_pen_quality(
    team: str,
    today: date,
    reliever_running: dict[str, dict],
    reliever_appearances: dict[str, list],
    reliever_team: dict[str, str],
    league_pen_cfip_value: float,
) -> Optional[float]:
    """Today's available bullpen quality for a team: average FIP of the top-N
    available relievers. Returns None if fewer than N relievers have
    IP >= RELIEVER_MIN_IP (too thin to be meaningful)."""
    candidates: list[float] = []
    for name, team_aff in reliever_team.items():
        if team_aff != team:
            continue
        running = reliever_running.get(name)
        if running is None or running.get("ip", 0.0) < RELIEVER_MIN_IP:
            continue
        appearances = reliever_appearances.get(name, [])
        if not is_reliever_available(appearances, today):
            continue
        # BB-only FIP from running totals + the league's reliever cFIP.
        # (Public box scores don't expose HBP; the constant absorbs the
        # systematic offset — consistent with the starter calculation.)
        ip = running["ip"]
        fip = (
            13 * running.get("hr", 0) + 3 * running.get("bb", 0)
            - 2 * running.get("so", 0)
        ) / ip + league_pen_cfip_value
        candidates.append(fip)

    if len(candidates) < TOP_N_AVAILABLE_PEN:
        return None
    candidates.sort()
    return sum(candidates[:TOP_N_AVAILABLE_PEN]) / TOP_N_AVAILABLE_PEN


def league_pen_cfip(totals: dict) -> float:
    """League-wide reliever cFIP = league_ERA − league_FIP_raw. Falls back
    to 3.10 (textbook) if we don't have enough relief innings yet."""
    ip = totals.get("ip", 0.0)
    if ip < 100:
        return 3.10
    league_era = (totals["er"] * 9.0) / ip
    league_fip_raw = (13 * totals["hr"] + 3 * totals["bb"] - 2 * totals["so"]) / ip
    return league_era - league_fip_raw


def team_rate_with_pen(
    starter_rate: float,
    pen_quality_fip: Optional[float],
) -> float:
    """Blend a team's expected runs-allowed rate: starter (60%) + pen (40%).

    Falls back to starter-only if pen quality is unavailable (early-season,
    not enough relievers with sample, or all relievers fatigued — rare with
    N=3 and a 7-9 man pen). Takes the already-computed starter rate so each
    consumer can bring its own effective-ERA definition (the live agent's
    includes xERA; the backtest's is FIP/ERA-only).
    """
    if pen_quality_fip is None:
        return starter_rate
    return (
        PEN_BLEND_STARTER_SHARE * starter_rate
        + PEN_BLEND_RELIEVER_SHARE * pen_quality_fip
    )
