"""UFCRatingAgent — Glicko-2 fighter ratings + results-derived feature layer.

UFC differs from team sports the same way tennis does (individual
competitors, name-matching, neutral venue) plus two of its own quirks:
sparse schedules (2-3 bouts/year → Glicko-2's rating deviation beats a fixed
Elo K) and hard age/inactivity cliffs (decline past ~33, ring rust after
long layoffs, KO-durability loss).

Model structure:
  1. Glicko-2 rating per fighter (evmax/models_ml/glicko2.py), each bout its
     own rating period with inactivity RD inflation.
  2. A logistic feature layer on point-in-time differentials — age, layoff,
     recent-KO-loss flag, experience, shrunk win rate, last-3 streak, reach,
     height, finish rate, KO-absorbed rate — blended IN LOGIT SPACE with the
     Glicko probability. Coefficients are fitted by
     scripts/backtest_ufc_model.py on fights through 2024 (walk-forward,
     holdout 2025+) and stored in the state file under "feature_model".
     Without a fitted feature_model the agent falls back to rating-only.

  NOTE: per-minute volume stats (SLpM/SApM/TD) are NOT in the feature set —
  ufcstats.com (the only free per-fight stat source) is behind a JS anti-bot
  interstitial, and ESPN's public API carries results/methods/bios only.
  See evmax/clients/ufc_espn.py.

State file: data/models/ufc_rating_state.json
  {
    "fighters": {
      "conor mcgregor": {"rating": 1712.4, "rd": 92.1, "sigma": 0.06,
                          "n_fights": 16, "wins": 12, "losses": 4, "draws": 0,
                          "last3": ["l", "w", "l"], "ko_losses": 2,
                          "last_ko_loss": "2021-07-10", "finish_wins": 11,
                          "last_fight": "2021-07-10", "dob": "1988-07-14",
                          "height_in": 69.0, "reach_in": 74.0, "espn_id": "3022677"},
      ...
    },
    "feature_model": {"intercept": ..., "coefs": {...}, "fitted_through": "..."},
    "last_updated": "YYYY-MM-DD"
  }

Seeded walk-forward from ESPN history via scripts/seed_ufc_ratings.py.
Name resolution reuses the tennis resolver (surname cascade, same-surname
ambiguity → None — wrong-fighter data is worse than no data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.tennis_common import resolve_player
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.models_ml.glicko2 import (
    DEFAULT_RATING,
    DEFAULT_RD,
    DEFAULT_SIGMA,
    GlickoRating,
    inflate_rd,
    update as glicko_update,
    win_probability,
)

# Skip predictions when the sector state is older than this relative to the
# game being predicted (same convention as EloModelAgent / FormModelAgent).
STALE_DAYS = 60

# Debut fighters have no "days since last fight" — use a typical mid-career
# gap so the layoff differential is neutral-ish rather than extreme.
DEFAULT_LAYOFF_DAYS = 270.0

# Feature order is part of the fitted-model contract — coefficients are
# stored by name, but keep this canonical list for the backtest harness.
FEATURE_NAMES: list[str] = [
    "age", "layoff", "recent_ko", "exp", "winrate",
    "streak", "reach", "height", "finish_rate", "ko_absorbed",
]


@dataclass
class FighterState:
    """Point-in-time record for one fighter (all fields pre-fight)."""

    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    sigma: float = DEFAULT_SIGMA
    n_fights: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last3: list = field(default_factory=list)   # most recent LAST, "w"/"l"/"d"
    ko_losses: int = 0
    last_ko_loss: Optional[str] = None
    finish_wins: int = 0
    last_fight: Optional[str] = None
    dob: Optional[str] = None
    height_in: Optional[float] = None
    reach_in: Optional[float] = None
    espn_id: Optional[str] = None

    def glicko(self) -> GlickoRating:
        return GlickoRating(self.rating, self.rd, self.sigma)

    def to_dict(self) -> dict:
        return {
            "rating": round(self.rating, 2), "rd": round(self.rd, 2),
            "sigma": round(self.sigma, 5),
            "n_fights": self.n_fights, "wins": self.wins,
            "losses": self.losses, "draws": self.draws,
            "last3": list(self.last3), "ko_losses": self.ko_losses,
            "last_ko_loss": self.last_ko_loss, "finish_wins": self.finish_wins,
            "last_fight": self.last_fight, "dob": self.dob,
            "height_in": self.height_in, "reach_in": self.reach_in,
            "espn_id": self.espn_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FighterState":
        return cls(**{k: d.get(k, getattr(cls(), k)) for k in cls().__dict__})


# ---------------------------------------------------------------------------
# Point-in-time helpers (pure — shared by live agent, seed, and backtest)
# ---------------------------------------------------------------------------


def _days_between(earlier: Optional[str], later: str) -> Optional[float]:
    if not earlier or not later:
        return None
    try:
        d0 = date.fromisoformat(earlier[:10])
        d1 = date.fromisoformat(later[:10])
    except ValueError:
        return None
    return float((d1 - d0).days)


def _age_years(dob: Optional[str], on: str) -> Optional[float]:
    days = _days_between(dob, on)
    return days / 365.25 if days is not None else None


def compute_features(a: FighterState, b: FighterState, fight_date: str) -> dict[str, float]:
    """Differential feature vector (a − b), all point-in-time / pre-fight.

    Missing bio attributes contribute 0 (no signal) rather than a guess.
    """
    age_a = _age_years(a.dob, fight_date)
    age_b = _age_years(b.dob, fight_date)
    age = (age_a - age_b) if (age_a is not None and age_b is not None) else 0.0

    lay_a = _days_between(a.last_fight, fight_date)
    lay_b = _days_between(b.last_fight, fight_date)
    lay_a = lay_a if lay_a is not None and lay_a >= 0 else DEFAULT_LAYOFF_DAYS
    lay_b = lay_b if lay_b is not None and lay_b >= 0 else DEFAULT_LAYOFF_DAYS
    layoff = math.log1p(lay_a) - math.log1p(lay_b)

    def _recent_ko(f: FighterState) -> float:
        days = _days_between(f.last_ko_loss, fight_date)
        return 1.0 if days is not None and 0 <= days <= 365 else 0.0

    recent_ko = _recent_ko(a) - _recent_ko(b)
    exp = math.log1p(a.n_fights) - math.log1p(b.n_fights)

    def _winrate(f: FighterState) -> float:
        return (f.wins + 2.5) / (f.n_fights + 5.0)

    winrate = _winrate(a) - _winrate(b)

    def _streak(f: FighterState) -> float:
        recent = f.last3[-3:]
        w = sum(1 for r in recent if r == "w")
        return (w + 1.5) / (len(recent) + 3.0)

    streak = _streak(a) - _streak(b)

    reach = (
        a.reach_in - b.reach_in
        if a.reach_in is not None and b.reach_in is not None else 0.0
    )
    height = (
        a.height_in - b.height_in
        if a.height_in is not None and b.height_in is not None else 0.0
    )

    def _finish_rate(f: FighterState) -> float:
        return (f.finish_wins + 1.0) / (f.wins + 2.0)

    finish_rate = _finish_rate(a) - _finish_rate(b)

    def _ko_absorbed(f: FighterState) -> float:
        return (f.ko_losses + 0.5) / (f.n_fights + 3.0)

    ko_absorbed = _ko_absorbed(a) - _ko_absorbed(b)

    return {
        "age": age, "layoff": layoff, "recent_ko": recent_ko, "exp": exp,
        "winrate": winrate, "streak": streak, "reach": reach, "height": height,
        "finish_rate": finish_rate, "ko_absorbed": ko_absorbed,
    }


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def blend_probability(
    p_rating: float,
    features: dict[str, float],
    feature_model: Optional[dict],
) -> float:
    """Combine the Glicko probability with the fitted feature layer.

    ``feature_model`` = {"intercept": float, "rating_coef": float,
    "coefs": {feature_name: coef}}. When absent, returns ``p_rating``
    unchanged (rating-only fallback).
    """
    if not feature_model:
        return p_rating
    z = float(feature_model.get("intercept", 0.0))
    z += float(feature_model.get("rating_coef", 1.0)) * _logit(p_rating)
    coefs = feature_model.get("coefs", {})
    for name, x in features.items():
        z += float(coefs.get(name, 0.0)) * x
    return _sigmoid(z)


def apply_fight_result(
    fighters: dict[str, FighterState],
    name_a: str,
    name_b: str,
    score_a: float,
    fight_date: str,
    method: Optional[str] = None,
) -> None:
    """Update both fighters' Glicko ratings + bookkeeping for one bout.

    ``score_a``: 1.0 = a won, 0.0 = b won, 0.5 = draw. Ratings update against
    the opponent's PRE-fight (inactivity-inflated) rating — standard pairwise
    Glicko-2 with one bout per rating period.
    """
    a = fighters.setdefault(name_a, FighterState())
    b = fighters.setdefault(name_b, FighterState())

    # Inflate RD for time since each fighter's previous bout.
    for f in (a, b):
        gap = _days_between(f.last_fight, fight_date)
        if gap and gap > 0:
            g = inflate_rd(f.glicko(), gap)
            f.rd = g.rd

    pre_a, pre_b = a.glicko(), b.glicko()
    new_a = glicko_update(pre_a, [(pre_b, score_a)])
    new_b = glicko_update(pre_b, [(pre_a, 1.0 - score_a)])
    a.rating, a.rd, a.sigma = new_a.rating, new_a.rd, new_a.sigma
    b.rating, b.rd, b.sigma = new_b.rating, new_b.rd, new_b.sigma

    for f, score in ((a, score_a), (b, 1.0 - score_a)):
        f.n_fights += 1
        f.last_fight = fight_date
        if score == 0.5:
            f.draws += 1
            f.last3 = (f.last3 + ["d"])[-3:]
        elif score == 1.0:
            f.wins += 1
            f.last3 = (f.last3 + ["w"])[-3:]
            if method in ("ko_tko", "submission"):
                f.finish_wins += 1
        else:
            f.losses += 1
            f.last3 = (f.last3 + ["l"])[-3:]
            if method == "ko_tko":
                f.ko_losses += 1
                f.last_ko_loss = fight_date


def replay_history(
    fights: list[dict],
    bios: dict[str, dict],
    collect: bool = False,
    feature_model: Optional[dict] = None,
) -> tuple[dict[str, FighterState], list[dict]]:
    """Walk-forward replay of a chronological fight list.

    Args:
        fights: rows with keys fighter_a / fighter_b / fighter_a_id /
            fighter_b_id / date / winner ("a"|"b"|"none") / method — the
            schema of data/backtest/ufc/fights.csv. MUST be chronological.
        bios: espn athlete_id → {dob, height_in, reach_in} static attributes.
        collect: also emit one prediction row per bout (pre-fight probs +
            features + outcome) for the backtest harness.
        feature_model: optional fitted feature layer, applied to the
            collected ``p_blend`` column (p_rating stays rating-only).

    Returns (fighters_state, collected_rows). No-contests and unresolvable
    rows are skipped; draws update ratings at 0.5.

    Name-collision guard: two DIFFERENT athletes (distinct ESPN ids) sharing
    a display name get keyed apart as "{name} #{id}" so their histories never
    merge; prediction-time surname resolution then sees the ambiguity and
    returns None (the tennis rule — never guess between two real people).
    """
    fighters: dict[str, FighterState] = {}
    key_by_id: dict[str, str] = {}
    rows: list[dict] = []

    def _key(name: str, espn_id: str) -> str:
        base = name.lower().strip()
        if espn_id and espn_id in key_by_id:
            return key_by_id[espn_id]
        key = base
        existing = fighters.get(base)
        if existing is not None and espn_id and existing.espn_id and existing.espn_id != espn_id:
            key = f"{base} #{espn_id}"
        if espn_id:
            key_by_id[espn_id] = key
        return key

    for fight in fights:
        winner = (fight.get("winner") or "none").strip()
        method = (fight.get("method") or "").strip() or None
        fight_date = (fight.get("date") or "")[:10]
        name_a = (fight.get("fighter_a") or "").strip()
        name_b = (fight.get("fighter_b") or "").strip()
        if not (name_a and name_b and fight_date):
            continue
        if winner == "a":
            score_a = 1.0
        elif winner == "b":
            score_a = 0.0
        elif method == "draw":
            score_a = 0.5
        else:
            continue  # NC / unknown non-result — no rating information

        key_a = _key(name_a, str(fight.get("fighter_a_id") or ""))
        key_b = _key(name_b, str(fight.get("fighter_b_id") or ""))
        if key_a == key_b:
            continue

        a = fighters.setdefault(key_a, FighterState())
        b = fighters.setdefault(key_b, FighterState())
        for f, name_key, fid in ((a, key_a, fight.get("fighter_a_id")), (b, key_b, fight.get("fighter_b_id"))):
            if f.espn_id is None and fid:
                f.espn_id = str(fid)
                bio = bios.get(str(fid), {})
                f.dob = bio.get("dob") or None
                f.height_in = _float_or_none(bio.get("height_in"))
                f.reach_in = _float_or_none(bio.get("reach_in"))

        if collect:
            # Pre-fight snapshot — probabilities + features BEFORE update.
            pa = a.glicko()
            pb = b.glicko()
            gap_a = _days_between(a.last_fight, fight_date)
            gap_b = _days_between(b.last_fight, fight_date)
            if gap_a and gap_a > 0:
                pa = inflate_rd(pa, gap_a)
            if gap_b and gap_b > 0:
                pb = inflate_rd(pb, gap_b)
            p_rating = win_probability(pa, pb)
            feats = compute_features(a, b, fight_date)
            rows.append({
                "date": fight_date,
                "fighter_a": key_a,
                "fighter_b": key_b,
                "p_rating": p_rating,
                "p_blend": blend_probability(p_rating, feats, feature_model),
                "outcome_a": score_a,
                "n_a": a.n_fights,
                "n_b": b.n_fights,
                **{f"x_{k}": v for k, v in feats.items()},
            })

        apply_fight_result(fighters, key_a, key_b, score_a, fight_date, method)

    return fighters, rows


def _float_or_none(v: Any) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The live model agent
# ---------------------------------------------------------------------------


class UFCRatingAgent(ModelAgent):
    """Glicko-2 + feature-layer win-probability model for UFC bouts.

    Only activates for sector == "ufc". Returns None for other sectors,
    unknown fighters, ambiguous same-surname resolutions, and stale state.
    """

    name = "ufc_rating"
    weight = 1.0   # sole non-sharp UFC model; SECTOR_WEIGHT_OVERRIDES pins it

    def _fighters(self) -> dict[str, dict]:
        return self._state.setdefault("fighters", {})

    def _resolve(self, label: str) -> Optional[str]:
        """Resolve a Kalshi/Pinnacle fighter label to a state key.

        Same cascade as tennis (exact → dehyphenated → surname →
        normalized-surname → prefix); same-surname ambiguity between distinct
        fighters returns None. Same-person duplicates break ties by fight
        count.
        """
        store = self._fighters()
        return resolve_player(
            label.lower().strip(),
            store,
            weight_fn=lambda k: store[k].get("n_fights", 0),
        )

    def _last_updated(self) -> Optional[str]:
        return self._state.get("last_updated")

    @staticmethod
    def _is_stale(last_updated: Optional[str], reference: Optional[date] = None) -> bool:
        if not last_updated:
            return False   # legacy/foreign state without stamp — never stale
        ref = reference or datetime.now(timezone.utc).date()
        try:
            upd = date.fromisoformat(last_updated[:10])
        except ValueError:
            return True
        return (ref - upd).days > STALE_DAYS

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "ufc":
            return None

        label_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        label_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()
        if not label_a or not label_b:
            return None

        reference = market.event_date.date() if market.event_date else None
        if self._is_stale(self._last_updated(), reference):
            return None

        key_a = self._resolve(label_a)
        key_b = self._resolve(label_b)
        if key_a is None or key_b is None or key_a == key_b:
            return None

        a = FighterState.from_dict(self._fighters()[key_a])
        b = FighterState.from_dict(self._fighters()[key_b])

        fight_date = (
            market.event_date.date().isoformat()
            if market.event_date
            else datetime.now(timezone.utc).date().isoformat()
        )

        ga, gb = a.glicko(), b.glicko()
        gap_a = _days_between(a.last_fight, fight_date)
        gap_b = _days_between(b.last_fight, fight_date)
        if gap_a and gap_a > 0:
            ga = inflate_rd(ga, gap_a)
        if gap_b and gap_b > 0:
            gb = inflate_rd(gb, gap_b)

        p_rating = win_probability(ga, gb)
        feats = compute_features(a, b, fight_date)
        prob_a = blend_probability(p_rating, feats, self._state.get("feature_model"))

        min_n = min(a.n_fights, b.n_fights)
        if min_n == 0:
            confidence = 0.35     # debutant — below the 0.45 ensemble gate
        elif min_n < 3:
            confidence = 0.48
        elif min_n < 6:
            confidence = 0.55
        elif min_n < 15:
            confidence = 0.65
        else:
            confidence = 0.72

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=1.0 - prob_a,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_n,
            notes=(
                f"glicko_a={a.rating:.0f}±{ga.rd:.0f} glicko_b={b.rating:.0f}±{gb.rd:.0f} "
                f"p_rating={p_rating:.3f} n={min_n} "
                f"feat={'on' if self._state.get('feature_model') else 'off'}"
            ),
        )

    # ------------------------------------------------------------------
    # Update from result (resolve-time; method unknown at this call site)
    # ------------------------------------------------------------------

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        """Feed a completed bout. score_a > score_b means fighter a won.

        Method of victory is unavailable through this generic interface, so
        finish/KO counters don't move here — the weekly reseed
        (scripts/seed_ufc_ratings.py) rebuilds them from ESPN with methods.
        """
        if sector != "ufc":
            return
        key_a = self._resolve(team_a.lower().strip()) or team_a.lower().strip()
        key_b = self._resolve(team_b.lower().strip()) or team_b.lower().strip()
        if key_a == key_b:
            return
        fight_date = (event_date or datetime.now(timezone.utc).date().isoformat())[:10]

        fighters = {
            k: FighterState.from_dict(v) for k, v in self._fighters().items()
        }
        if score_a > score_b:
            s = 1.0
        elif score_b > score_a:
            s = 0.0
        else:
            s = 0.5
        apply_fight_result(fighters, key_a, key_b, s, fight_date, method=None)
        self._state["fighters"] = {k: f.to_dict() for k, f in fighters.items()}

        prev = self._state.get("last_updated")
        if prev is None or fight_date > prev:
            self._state["last_updated"] = fight_date

    # ------------------------------------------------------------------
    # Seeding (full replace — idempotent)
    # ------------------------------------------------------------------

    def seed_from_history(
        self,
        fights: list[dict],
        bios: dict[str, dict],
        feature_model: Optional[dict] = None,
    ) -> int:
        """Full-replace rebuild of the fighter store from a chronological
        fight list (see :func:`replay_history` for the schema). Returns the
        number of fighters seeded. Refuses to write on an empty fight list
        (the tennis-seed precedent: never clobber good state with nothing).
        """
        if not fights:
            self.log.warning("ufc_seed_empty_history_abort")
            return 0
        fighters, _ = replay_history(fights, bios)
        self._state["fighters"] = {k: f.to_dict() for k, f in fighters.items()}
        if feature_model is not None:
            self._state["feature_model"] = feature_model
        last = max((f.get("date") or "")[:10] for f in fights)
        self._state["last_updated"] = last
        self.save_state()
        self.log.info("ufc_ratings_seeded", fighters=len(fighters), through=last)
        return len(fighters)
