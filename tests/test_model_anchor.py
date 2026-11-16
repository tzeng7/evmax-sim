"""Tests for the model-anchored shadow lane (agents/models/model_anchor.py)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evmax.agents.models import model_anchor as ma
from evmax.agents.odds.ev_gap_agent import _NON_MODEL_TOKENS, has_full_blend
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook

GAME_DAY = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def market(home, away, yes, mt=MarketType.moneyline, price=0.5):
    return PredictionMarket(
        id=f"kalshi:KXNCAAWBGAME-X-{yes}", source=MarketSource.kalshi, sector="ncaaw",
        market_type=mt, ticker=f"KXNCAAWBGAME-X-{yes}", yes_price=price, no_price=1 - price + 0.02,
        team_home=home, team_away=away, yes_team=yes, event_date=GAME_DAY)


class _Model:
    def __init__(self, name, probs):
        self.name, self._probs = name, probs

    async def __call__(self, request):
        data = {}
        for pair in request.params["pairs"]:
            eid = pair["sharp"].event_id
            if eid in self._probs:
                data[eid] = SimpleNamespace(true_prob_a=self._probs[eid])
        return SimpleNamespace(status="ok", data=data)


class _Ensemble:
    """Stand-in for EnsembleModelAgent: averages whatever fired, sharp ignored."""

    def __init__(self, models):
        self._models = models
        self.blend_calls = []

    def _sector_model_names(self, sector):
        return {m.name for m in self._models}

    def _blend(self, event_id, preds, sharp, sharp_weight, sector, sector_model_names=None):
        self.blend_calls.append((event_id, sharp, sharp_weight))
        if not preds:
            return None
        p = sum(x.true_prob_a for x in preds.values()) / len(preds)
        return SimpleNamespace(true_prob_a=p, true_prob_b=1 - p,
                               model_sources="+".join(sorted(preds)))


EID = "ncaaw::2026-11-20::portland_vs_long_beach_state"


def _run(models, markets):
    ens = _Ensemble(models)
    return asyncio.run(ma.model_anchor_odds(ens, "ncaaw", markets)), ens


def test_only_listed_sectors_are_anchored():
    assert ma.model_anchor_enabled("ncaaw") and ma.model_anchor_enabled("NCAAW")
    assert not ma.model_anchor_enabled("nba") and not ma.model_anchor_enabled(None)


def test_game_stubs_one_per_game_from_moneyline_markets_only():
    mk = [market("portland", "long beach state", "portland"),
          market("portland", "long beach state", "long beach state"),
          market("portland", "long beach state", "over", mt=MarketType.total),
          market("", "x", "x"), market("y", "y", "y")]
    stubs = ma.game_stubs("ncaaw", mk)
    assert list(stubs) == [EID]
    stub, m = stubs[EID]
    assert stub.book == SharpBook.evmax_model
    assert (stub.outcome_a_label, stub.outcome_b_label) == ("portland", "long beach state")
    assert stub.true_prob_a == 0.5                      # carries no price
    assert m.market_type == MarketType.moneyline


def test_anchor_is_the_model_blend_with_no_sharp_input():
    mk = [market("portland", "long beach state", "portland")]
    (anchor,), ens = _run([_Model("ncaaw_efficiency", {EID: 0.70}), _Model("elo", {EID: 0.60})], mk)
    assert anchor.true_prob_a == pytest.approx(0.65)
    assert anchor.true_prob_b == pytest.approx(0.35)
    assert anchor.outcome_a_decimal == pytest.approx(1 / 0.65, abs=1e-3)
    assert anchor.book == SharpBook.evmax_model and anchor.event_id == EID
    # the models were blended with sharp=None and zero sharp weight
    assert ens.blend_calls == [(EID, None, 0.0)]


def test_game_is_skipped_unless_the_validated_model_fired():
    mk = [market("portland", "long beach state", "portland")]
    anchors, _ = _run([_Model("elo", {EID: 0.10})], mk)          # Elo alone: no anchor
    assert anchors == []
    anchors, _ = _run([_Model("ncaaw_efficiency", {EID: 0.55})], mk)
    assert len(anchors) == 1


def test_no_prediction_or_degenerate_blend_gives_no_anchor():
    mk = [market("portland", "long beach state", "portland")]
    assert _run([_Model("ncaaw_efficiency", {})], mk)[0] == []
    assert _run([_Model("ncaaw_efficiency", {EID: 0.999})], mk)[0] == []
    assert _run([_Model("ncaaw_efficiency", {EID: 0.7})], [])[0] == []


def test_a_failing_model_does_not_break_the_others():
    class Boom:
        name = "form"

        async def __call__(self, request):
            raise RuntimeError("state missing")

    mk = [market("portland", "long beach state", "portland")]
    anchors, _ = _run([Boom(), _Model("ncaaw_efficiency", {EID: 0.6})], mk)
    assert anchors[0].true_prob_a == pytest.approx(0.6)


def test_retag_replaces_sharp_and_is_idempotent():
    assert ma.retag_sources("elo+ncaaw_efficiency+sharp+injury") == (
        "elo+ncaaw_efficiency+injury+model_anchor")
    assert ma.retag_sources("sharp(capped)") == "model_anchor"
    once = ma.retag_sources("elo+sharp")
    assert ma.retag_sources(once) == once
    assert ma.retag_sources(None) == "model_anchor"


def test_model_anchored_rows_are_never_live_eligible():
    src = "elo+ncaaw_efficiency+ncaaw_possession_sim+model_anchor"
    for mt in ("moneyline", "spread", "total", None):
        assert has_full_blend("ncaaw", src, mt) is False
    # whatever the sector: the token alone disqualifies the row
    assert has_full_blend("baseball", "pitcher_v2+elo+model_anchor", "moneyline") is False
    assert "model_anchor" in _NON_MODEL_TOKENS


def test_token_is_in_every_non_model_token_mirror():
    from evmax.agents.cleanup import contamination, integrity

    assert "model_anchor" in contamination._NON_MODEL_TOKENS
    assert "total_pmf" in contamination._NON_MODEL_TOKENS
    src = open(integrity.__file__).read()
    assert '"model_anchor"' in src and '"total_pmf"' in src


def test_ncaaw_ships_in_shadow():
    from datetime import date

    from evmax.modes import get_mode

    d = date(2026, 11, 20)
    for mt in ("moneyline", "spread", "total"):
        assert get_mode("ncaaw", mt, today=d) == "shadow"
