"""Model-anchored pricing for sectors with no sharp line.

The scanner prices a market against a sharp book. A sector the sharp book does
not post — women's college basketball outside March (Pinnacle league 583 is
valid and empty; no free feed of another sharp book exists) — therefore matches
nothing and is never priced. This module fills that gap with the sector's own
model blend:

    for every game the venue lists, the ensemble's models run with NO sharp
    input, and their blend becomes the game's anchor record.

The record then flows through the normal pipeline (matching, EV, logging), so a
market whose venue price disagrees with the model is logged like any other gap.
Three things keep it from ever being mistaken for a sharp-priced play:

  * the record's book is ``SharpBook.evmax_model`` (archived under that name,
    so ``WHERE book = 'pinnacle'`` never returns it);
  * the row's ``model_sources`` carries ``model_anchor`` in place of ``sharp``;
  * ``has_full_blend`` rejects that token, so the row is logged in shadow,
    never displayed as a play and never sized.

It exists to MEASURE: does a market with no sharp reference misprice against
our model? Judge it on outcomes and entry-to-close movement of the rows tagged
``model_anchor`` (``cleanup shadow clv <sector> --sources-token model_anchor``).

It only runs for a sector in ``MODEL_ANCHOR_SECTORS`` and only in a scan where
the sharp fetch returned NOTHING for that sector. The moment the sharp book
posts a line the sector prices off it as usual.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from evmax.agents.base import AgentRequest
from evmax.clients.time_util import kalshi_game_day
from evmax.models.market import MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds

MODEL_ANCHOR_SECTORS: frozenset[str] = frozenset({"ncaaw"})
MODEL_ANCHOR_TOKEN = "model_anchor"
# Models that MUST have contributed for a game to be anchored. An anchor built
# from whatever happened to fire is not the model that was validated: on the
# 2026-11-16 slate, games where only generic Elo fired (the efficiency model
# waits for 4 games per team) sat a median ~70pp from the venue price. The
# women's efficiency model is the one with a held-out record (Brier 0.1602 on
# 2025-26, n=4731).
MODEL_ANCHOR_REQUIRED: dict[str, frozenset[str]] = {
    "ncaaw": frozenset({"ncaaw_efficiency"}),
}
# A model blend this close to certainty is a broken input, not a price.
_MIN_PROB, _MAX_PROB = 0.02, 0.98


def model_anchor_enabled(sector: Optional[str]) -> bool:
    return (sector or "").lower() in MODEL_ANCHOR_SECTORS


def game_stubs(sector: str, markets: list[PredictionMarket]) -> dict[str, tuple[SharpOdds, PredictionMarket]]:
    """One placeholder record per listed game: {event_id: (stub, a market of that game)}.

    Built from the venue's own moneyline markets: home and away come from the
    market, the date from its game day. The stub carries no price (50/50); it
    only tells the models which two teams meet and when.
    """
    out: dict[str, tuple[SharpOdds, PredictionMarket]] = {}
    for m in markets:
        if m.market_type != MarketType.moneyline or not m.team_home or not m.team_away:
            continue
        if m.team_home == m.team_away or m.event_date is None:
            continue
        home = m.team_home.replace(" ", "_")
        away = m.team_away.replace(" ", "_")
        event_id = f"{sector}::{kalshi_game_day(m.event_date, sector)}::{home}_vs_{away}"
        if event_id in out:
            continue
        out[event_id] = (SharpOdds(
            event_id=event_id, book=SharpBook.evmax_model, sector=sector,
            outcome_a_label=m.team_home, outcome_b_label=m.team_away,
            outcome_a_decimal=2.0, outcome_b_decimal=2.0,
            true_prob_a=0.5, true_prob_b=0.5, event_date=m.event_date,
        ), m)
    return out


_ANCHOR_TOKENS = (MODEL_ANCHOR_TOKEN, "soft_anchor")


def retag_sources(model_sources: Optional[str], token: str = MODEL_ANCHOR_TOKEN) -> str:
    """Replace the ``sharp`` token with ``token`` (the anchor was not a sharp book)."""
    tokens = [t.strip() for t in (model_sources or "").split("+") if t.strip()]
    tokens = [t for t in tokens if t not in ("sharp", "sharp(capped)", *_ANCHOR_TOKENS)]
    return "+".join(tokens + [token])


def base_event_id(event_id: Optional[str]) -> str:
    """``sector::date::home_vs_away`` of any record or gap id (drops ::spread / ::total…)."""
    return "::".join((event_id or "").split("::")[:3])


def demote_anchored_gaps(gaps: list, anchored: dict[str, str]) -> int:
    """Tag and neutralize every gap of a non-sharp-anchored game.

    ``anchored`` maps a base event id to its token. The moneyline blend is
    retagged before pricing, but a spread / total gap builds its own source
    string inside the EV agent, so the tag is enforced here for every market
    type: sources retagged, ``full_blend`` False, Kelly zero.
    """
    n = 0
    for g in gaps:
        token = anchored.get(base_event_id(getattr(g, "event_id", None)))
        if token is None:
            continue
        g.model_sources = retag_sources(g.model_sources, token)
        g.full_blend = False
        g.kelly_fraction = 0.0
        g.kelly_full = 0.0
        n += 1
    return n


async def model_anchor_odds(ensemble: Any, sector: str,
                            markets: list[PredictionMarket]) -> list[SharpOdds]:
    """Anchor records for ``sector`` from the ensemble's models alone.

    A game is returned only when at least one model produced a prediction that
    cleared the ensemble's confidence gate. Any failure returns [] — the sector
    then simply stays unpriced, exactly as before.
    """
    stubs = game_stubs(sector, markets)
    if not stubs:
        return []
    pairs = [{"market": m, "sharp": stub} for stub, m in stubs.values()]
    try:
        responses = await asyncio.gather(
            *(model(AgentRequest(sector=sector, params={"pairs": pairs}))
              for model in ensemble._models),
            return_exceptions=True,
        )
    except Exception:  # noqa: BLE001
        return []
    per_event: dict[str, dict] = {}
    for model, resp in zip(ensemble._models, responses):
        if isinstance(resp, Exception) or getattr(resp, "status", "") != "ok" or not resp.data:
            continue
        for event_id, pred in resp.data.items():
            per_event.setdefault(event_id, {})[model.name] = pred

    names = ensemble._sector_model_names(sector)
    out: list[SharpOdds] = []
    for event_id, preds in per_event.items():
        stub = stubs.get(event_id)
        if stub is None:
            continue
        blend = ensemble._blend(event_id, preds, None, 0.0, sector, sector_model_names=names)
        # No models past the gate = no anchor (with sharp=None the blend is None).
        if blend is None or not (_MIN_PROB < blend.true_prob_a < _MAX_PROB):
            continue
        fired = {t.strip() for t in (blend.model_sources or "").split("+")}
        if not MODEL_ANCHOR_REQUIRED.get(sector.lower(), frozenset()) <= fired:
            continue
        p_a = float(blend.true_prob_a)
        out.append(stub[0].model_copy(update={
            "true_prob_a": p_a, "true_prob_b": 1.0 - p_a,
            "outcome_a_decimal": round(1.0 / p_a, 4),
            "outcome_b_decimal": round(1.0 / (1.0 - p_a), 4),
        }))
    return out
