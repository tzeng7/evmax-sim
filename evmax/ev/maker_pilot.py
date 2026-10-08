"""Maker-only pilot lane for model-priced shallow NFL spread rungs (2026-10-07).

Since 2026-10-03 an NFL spread rung with no Pinnacle price at its own line is
priced by the key-number margin PMF (`spread_pmf`) and kept backend-only:
``has_full_blend`` requires the `sharp_ladder` token, so the row is hidden and
logged shadow. Pinnacle never posts a rung where the underdog lays points, so
the "underdog wins by N.5+" rungs can only ever be PMF-priced.

The pilot re-opens the SHALLOW part of that lane as maker-only plays:

- Eligible: sector in ``MAKER_PILOT_SECTORS``, spread market, priced by
  `spread_pmf` (no `sharp_ladder`), ``|line| <= MAKER_PILOT_MAX_ABS_LINE``,
  venue in ``MAKER_PILOT_VENUES``, the YES team is the Pinnacle UNDERDOG
  (``MAKER_PILOT_UNDERDOG_ONLY``), and the gap has an actionable maker rest
  price with a positive maker Kelly.
- Execution: maker only. The taker stake is zeroed and ``maker_only`` is set,
  so ``log_gaps`` persists the row as shadow and ``agents pick`` never crosses
  the ask. The row becomes a live position only through ``evmax agents fill``
  (or the dashboard Fill button) once the resting order fills.
- Size: ``MAKER_PILOT_KELLY_MULT`` (¼) of the maker Kelly, and the sum over one
  game is capped at ``MAKER_PILOT_MAX_GAME_KELLY`` (¼ of the 8% per-game guard).
  The rungs of one game all bet on the same underdog margin, so they are one
  correlated position.
- Tracking: the `maker_pilot` token in model_sources. Filled rows are
  ``mode='live'`` (set by ``record_maker_fill``), so
  ``scripts/maker_pilot_readout.py`` reads the checkpoint from them.

Why only underdogs: every shallow `spread_pmf` row in the evidence (Kalshi,
2026-09-22 to 10-07: 174 underdog lays + 18 underdog takes) bet the underdog;
there were none on favorites. Favorite rungs show up where Pinnacle's ladder
stops short of the main line plus a few points (e.g. Texans -10.5 off a -7.5
main), and nothing has measured them. The underdog is Pinnacle's ``outcome_b``
(``_parse_spread`` puts the negative handicap on ``outcome_a``), the same flag
the PMF already receives as ``yes_is_underdog``.

Why only shallow rungs: the PMF runs about +3.1pp rich against Pinnacle's own
ladder beyond 10.5 points, and the deep-rung CLV was weak (+0.44pp, 51%
positive). Why maker only: the shallow-rung CLV was +1.21pp per game before
fees but 0.00pp after the Kalshi taker fee (2026-10-07 gate check).

Checkpoint rules (judged on REAL fills, net of the maker fee):
- 15 filled games: negative mean CLV -> stop (empty ``MAKER_PILOT_SECTORS``);
  positive with >=55% of rows positive -> raise the multiplier to 0.5.
- 30 filled games: the normal CLV gate decides full maker size.

Turn the pilot off by setting ``MAKER_PILOT_SECTORS = set()``.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

MAKER_PILOT_TOKEN = "maker_pilot"
MAKER_PILOT_SECTORS: set[str] = {"nfl"}
MAKER_PILOT_VENUES: set[str] = {"kalshi"}
MAKER_PILOT_MAX_ABS_LINE = 10.5
MAKER_PILOT_UNDERDOG_ONLY = True
MAKER_PILOT_KELLY_MULT = 0.25
MAKER_PILOT_MAX_GAME_KELLY = 0.02

_PMF_TOKEN = "spread_pmf"
_LADDER_TOKEN = "sharp_ladder"


def _tokens(model_sources: Optional[str]) -> set[str]:
    return {tok.strip() for tok in (model_sources or "").split("+")}


def is_maker_pilot(model_sources: Optional[str]) -> bool:
    """True when the row was sized by the pilot (carries the pilot token)."""
    return MAKER_PILOT_TOKEN in _tokens(model_sources)


def maker_pilot_eligible(
    sector: Optional[str],
    market_type: Optional[str],
    model_sources: Optional[str],
    line: Optional[float],
    venue: Optional[str],
    yes_is_underdog: Optional[bool] = None,
) -> bool:
    """True when a gap belongs to the pilot lane (before the maker-plan check).

    ``yes_is_underdog``: whether the YES team is Pinnacle's underdog. Unknown
    (None) fails closed while ``MAKER_PILOT_UNDERDOG_ONLY`` is on.
    """
    if MAKER_PILOT_UNDERDOG_ONLY and yes_is_underdog is not True:
        return False
    if (sector or "").lower() not in MAKER_PILOT_SECTORS:
        return False
    if market_type != "spread" or line is None:
        return False
    if (venue or "kalshi") not in MAKER_PILOT_VENUES:
        return False
    tokens = _tokens(model_sources)
    if _PMF_TOKEN not in tokens or _LADDER_TOKEN in tokens:
        return False
    return abs(float(line)) <= MAKER_PILOT_MAX_ABS_LINE


def apply_maker_pilot(gap: Any, yes_is_underdog: Optional[bool] = None) -> Any:
    """Return ``gap`` converted to a pilot maker play, or unchanged.

    ``yes_is_underdog`` is whether the gap's YES team (the opponent, for a
    NO-side gap) is Pinnacle's underdog. A gap without an actionable maker rest
    price (no bid ladder, or the bid already past the maker ceiling) or a
    quarantined gap stays as it was: hidden and backend-only.
    """
    if is_maker_pilot(gap.model_sources):
        return gap  # already converted — never quarter the stake twice
    if not maker_pilot_eligible(
        gap.sector, gap.market_type, gap.model_sources, gap.line,
        getattr(gap, "venue", "kalshi"), yes_is_underdog,
    ):
        return gap
    if getattr(gap, "quarantined", False):
        return gap
    maker_kelly = getattr(gap, "maker_bid_kelly_fraction", None)
    if getattr(gap, "maker_bid_price", None) is None or not maker_kelly or maker_kelly <= 0:
        return gap
    return dataclasses.replace(
        gap,
        maker_only=True,
        kelly_fraction=0.0,
        maker_bid_kelly_fraction=round(maker_kelly * MAKER_PILOT_KELLY_MULT, 4),
        model_sources=f"{gap.model_sources}+{MAKER_PILOT_TOKEN}",
        full_blend=True,
    )


def _game_key(event_id: Optional[str]) -> str:
    # Same grouping as the coordinator's per-game exposure guard: drop the
    # ::spread[::line] suffix so every rung of one matchup shares a budget.
    return "::".join((event_id or "").split("::")[:3])


def cap_maker_pilot_game_exposure(
    gaps: list[Any],
    max_game_kelly: float = MAKER_PILOT_MAX_GAME_KELLY,
) -> list[Any]:
    """Scale pilot maker stakes down so one game's total stays under the cap.

    Only pilot gaps are touched, and only their ``maker_bid_kelly_fraction``.
    Within an over-cap game every pilot rung shrinks by the same factor, so the
    relative sizing between rungs is kept. Order of ``gaps`` is preserved.
    """
    totals: dict[str, float] = {}
    for g in gaps:
        if is_maker_pilot(g.model_sources):
            key = _game_key(g.event_id)
            totals[key] = totals.get(key, 0.0) + (g.maker_bid_kelly_fraction or 0.0)
    scale = {
        key: max_game_kelly / total
        for key, total in totals.items()
        if total > max_game_kelly
    }
    if not scale:
        return gaps
    out = []
    for g in gaps:
        factor = scale.get(_game_key(g.event_id)) if is_maker_pilot(g.model_sources) else None
        if factor is None:
            out.append(g)
        else:
            # Floor (not round) to 4 dp so the game's total never ends above the cap.
            capped = math.floor((g.maker_bid_kelly_fraction or 0.0) * factor * 1e4) / 1e4
            out.append(dataclasses.replace(g, maker_bid_kelly_fraction=capped))
    return out


# ---------------------------------------------------------------------------
# Checkpoint readout — judged on REAL fills, net of the maker fee
# ---------------------------------------------------------------------------

# Mirrors the CLV promotion gate in evmax/cli/commands/shadow.py
# (MIN_CLV_RESOLVED / CLV_MIN_MEAN_PP / CLV_MIN_FRAC_POSITIVE), kept local so the
# ev layer does not import the CLI.
PILOT_CHECKPOINT_GAMES = 15
PILOT_GATE_GAMES = 30
PILOT_MIN_FRAC_POSITIVE = 0.55


def checkpoint_verdict(rows: list[dict]) -> dict:
    """Score filled pilot rows and return the checkpoint decision.

    ``rows``: filled pilot rows with ``event_id``, ``kalshi_clv_pct`` (pp,
    measured from the fill price), ``placed_price`` (fill, 0-1) and ``venue``.

    Net CLV = CLV − the venue's maker fee at the fill price. The mean is taken
    per GAME (rungs of one game are one correlated position); % positive is
    per row, as in the promotion gate.
    """
    from evmax.fees import venue_fee_prob

    by_game: dict[str, list[float]] = {}
    nets: list[float] = []
    for r in rows:
        price = r.get("placed_price")
        clv = r.get("kalshi_clv_pct")
        if price is None or clv is None or not (0.0 < price < 1.0):
            continue
        net = clv - venue_fee_prob(r.get("venue") or "kalshi", price, maker=True) * 100
        nets.append(net)
        by_game.setdefault(_game_key(r.get("event_id")), []).append(net)

    games = len(by_game)
    if games == 0:
        return {"games": 0, "rows": 0, "mean_net_pp": None, "z": None,
                "frac_positive": None, "verdict": "COLLECTING",
                "action": f"No filled pilot games yet (checkpoint at {PILOT_CHECKPOINT_GAMES})."}
    game_means = [sum(v) / len(v) for v in by_game.values()]
    mean = sum(game_means) / games
    if games > 1:
        sd = math.sqrt(sum((m - mean) ** 2 for m in game_means) / (games - 1))
        z = mean / (sd / math.sqrt(games)) if sd > 0 else None
    else:
        z = None
    frac_pos = sum(1 for n in nets if n > 0) / len(nets)

    if games < PILOT_CHECKPOINT_GAMES:
        verdict, action = "COLLECTING", f"{games}/{PILOT_CHECKPOINT_GAMES} filled games to the checkpoint."
    elif mean < 0:
        verdict, action = "STOP", "Set MAKER_PILOT_SECTORS = set() in evmax/ev/maker_pilot.py."
    elif games >= PILOT_GATE_GAMES and frac_pos >= PILOT_MIN_FRAC_POSITIVE:
        verdict, action = "FULL-SIZE", "Gate cleared: set MAKER_PILOT_KELLY_MULT = 1.0."
    elif frac_pos >= PILOT_MIN_FRAC_POSITIVE:
        verdict, action = "STEP-UP", "Set MAKER_PILOT_KELLY_MULT = 0.5 (if still 0.25)."
    else:
        verdict, action = "HOLD", "Positive but below 55% of rows positive: keep ¼ and collect."
    return {"games": games, "rows": len(nets), "mean_net_pp": round(mean, 3),
            "z": round(z, 2) if z is not None else None,
            "frac_positive": round(frac_pos, 3), "verdict": verdict, "action": action}
