"""Canonical outcome-label formatting — the single source for how a bet's
"Outcome" column renders across the CLI tables, the web dashboard, the
notifier, and EVGap.display_label.

Lives at the package root (imports nothing from evmax) so every layer —
models, agents, web, cli — can use it without an import cycle. Before this
module there were four near-but-not-identical copies; the divergences caused
real bugs (a NO-side +9.5 cover rendering as "Warriors 9.5", indistinguishable
from a -9.5 favorite). The format here is the richest of the four: positive
spreads get a leading "+", totals render "Over"/"Under" (not "O/U"), and props
render "Mathurin 24+ AST".
"""

from __future__ import annotations

from typing import Any, Optional

# Short stat labels for player props (points -> PTS, etc.).
STAT_LABELS = {
    "points": "PTS", "assists": "AST", "rebounds": "REB",
    "threes": "3PM", "pra": "PRA", "pts_reb": "P+R", "pts_ast": "P+A",
    "reb_ast": "R+A", "steals": "STL", "blocks": "BLK",
    "strikeouts": "K", "hits": "H", "total_bases": "TB",
}


def format_outcome_label(
    *,
    yes_team: Optional[str],
    market_type: Optional[str],
    line: Optional[float] = None,
    prop_player_name: Optional[str] = None,
    prop_stat_type: Optional[str] = None,
    prop_threshold: Optional[float] = None,
) -> str:
    """Render the human-readable outcome label for one bet.

    Game markets: "Lakers ML", "Warriors +9.5", "Pistons -7.5", "Over 220.5".
    Player props: "Mathurin 24+ AST". Unknown market types fall back to the
    capitalized team name.
    """
    mt = (market_type or "").lower()
    team_raw = (yes_team or "?").strip()

    if mt == "player_prop":
        player = (prop_player_name or team_raw or "?").replace("_", " ")
        player = " ".join(p.capitalize() for p in player.split())
        stat_raw = (prop_stat_type or "").lower()
        stat = STAT_LABELS.get(stat_raw, stat_raw.replace("_", " ").upper() or "PROP")
        thr = prop_threshold if prop_threshold is not None else line
        if thr is not None:
            try:
                thr_str = f"{float(thr):g}+"
            except (TypeError, ValueError):
                thr_str = str(thr)
            return f"{player} {thr_str} {stat}"
        return f"{player} {stat}"

    team = team_raw.capitalize() if team_raw else "?"
    if mt == "moneyline":
        return f"{team} ML"
    if mt == "advance":
        return f"{team} to advance"
    if mt == "spread" and line is not None:
        try:
            ln = float(line)
        except (TypeError, ValueError):
            return f"{team} {line}"
        # Positive line = NO-side / "+spread" cover; render with leading "+".
        sign = "+" if ln > 0 else ""
        line_str = f"{sign}{ln:.1f}".rstrip("0").rstrip(".")
        return f"{team} {line_str}"
    if mt in ("over_under", "total") and line is not None:
        side = "Under" if team_raw.lower().startswith("u") else "Over"
        try:
            return f"{side} {float(line):.1f}"
        except (TypeError, ValueError):
            return f"{side} {line}"
    return team


def format_outcome_label_for_row(row: dict[str, Any]) -> str:
    """Adapter: render the label from a predictions.db row / scan-gap dict."""
    return format_outcome_label(
        yes_team=row.get("yes_team"),
        market_type=row.get("market_type"),
        line=row.get("line"),
        prop_player_name=row.get("prop_player_name"),
        prop_stat_type=row.get("prop_stat_type"),
        prop_threshold=row.get("prop_threshold"),
    )
