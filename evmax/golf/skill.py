"""Skill model — projected strokes-gained per round + player variance.

This is Layer 2/3 of the design doc, adapted to the free-data ceiling
(per-EVENT / season SG, never per-round shot-level). Two entry points, both
producing ``{canonical_name: PlayerSkill}`` that the simulator consumes:

  * ``skill_from_pga`` — from real PGA Tour season SG snapshots. Applies
    *category-specific predictive shrinkage* (the design doc's per-category
    shrinkage: OTT/APP are stable and retain most of their edge; ARG/PUTT
    regress hard, so they're shrunk toward the field) plus sample-size
    shrinkage toward replacement level. This is the live, real-SG path.

  * ``skill_from_records`` — from per-event ``SGRecord``s (ESPN score-proxy or
    the historical ASA set) with exponential recency decay. This is the
    offline-testable path and the fallback when no SG feed is available.

Both then feed ``to_sim_inputs`` which aligns the field to mean/SD arrays,
assigning missing golfers a low-confidence replacement level (PGA-only SG can't
see LIV / DP-World-only players — a documented limitation, not field-average).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from evmax.golf.data.pga import PgaSgSnapshot
from evmax.golf.models import PlayerSkill, SGRecord
from evmax.golf.names import canonical

_log = logging.getLogger(__name__)


@dataclass
class SkillConfig:
    # Category predictive-retention weights (how much of a player's category
    # edge survives regression to the mean). OTT/APP stable, ARG/PUTT noisy.
    retention: dict[str, float] = field(
        default_factory=lambda: {"ott": 0.85, "app": 0.90, "arg": 0.65, "putt": 0.55}
    )
    # Sample-size shrinkage toward replacement: reliability = n/(n+shrink_rounds).
    # Season SG stabilises fast (~a dozen rounds), so this is light — a 50-round
    # sample keeps ~85% of its signal.
    shrink_rounds: float = 8.0
    # Exponential recency decay for the per-event record path (half-life in
    # EVENTS, since per-event is the grain).
    decay_halflife_events: float = 12.0
    # Per-round score standard deviation (field baseline; ~2.7-2.9 on tour).
    base_round_sd: float = 2.8
    # Better players are slightly more consistent: sd = base - sd_skill_beta*skill,
    # clamped to [sd_floor, sd_ce=base+0.4]. Small, real effect.
    sd_skill_beta: float = 0.15
    sd_floor: float = 2.3
    # Replacement level (SG/round vs field) for golfers with no PGA SG. Set
    # meaningfully below average: most no-data entries in a major field are
    # club pros / cross-tour qualifiers who are genuinely 1.5-3 strokes worse.
    # (Known blind spot: this WRONGLY buckets elite LIV players — Rahm,
    # DeChambeau — as replacement level too; the free PGA feed can't see them.)
    replacement_sg: float = -1.20
    replacement_confidence: float = 0.20
    min_confidence: float = 0.30
    max_confidence: float = 0.85


def _round_sd(skill_sg: float, cfg: SkillConfig) -> float:
    sd = cfg.base_round_sd - cfg.sd_skill_beta * skill_sg
    return max(cfg.sd_floor, min(cfg.base_round_sd + 0.4, sd))


def _confidence(n_rounds: Optional[float], cfg: SkillConfig) -> float:
    if not n_rounds or n_rounds <= 0:
        return cfg.min_confidence
    c = n_rounds / (n_rounds + cfg.shrink_rounds)
    return max(cfg.min_confidence, min(cfg.max_confidence, c))


def skill_from_pga(
    snapshots: dict[str, PgaSgSnapshot],
    cfg: Optional[SkillConfig] = None,
) -> dict[str, PlayerSkill]:
    """Build per-player skill from PGA season SG snapshots.

    predictive_sg = Σ retention_c · category_sg  (categories present), OR the
    shrunk SG-total when categories are absent. Then sample-size-shrunk toward
    replacement by measured rounds.
    """
    cfg = cfg or SkillConfig()
    out: dict[str, PlayerSkill] = {}
    for name, snap in snapshots.items():
        rounds = snap.measured_rounds
        reliability = (
            rounds / (rounds + cfg.shrink_rounds) if rounds else 0.0
        )
        components: dict[str, float] = {}
        if snap.has_categories:
            predictive = 0.0
            for cat, attr in (("ott", "sg_ott"), ("app", "sg_app"),
                              ("arg", "sg_arg"), ("putt", "sg_putt")):
                raw = getattr(snap, attr) or 0.0
                w = cfg.retention.get(cat, 0.75)
                components[cat] = raw
                predictive += w * raw
        elif snap.sg_total is not None:
            predictive = snap.sg_total * 0.80  # generic retention on the aggregate
        else:
            continue

        # Shrink toward replacement level by sample reliability.
        skill_sg = reliability * predictive + (1.0 - reliability) * cfg.replacement_sg
        out[name] = PlayerSkill(
            player=name,
            skill_sg=skill_sg,
            round_sd=_round_sd(skill_sg, cfg),
            n_events=int((rounds or 0) // 4),
            confidence=_confidence(rounds, cfg),
            components=components,
        )
    return out


def skill_from_records(
    records: Sequence[SGRecord],
    as_of_index: Optional[dict[str, int]] = None,
    cfg: Optional[SkillConfig] = None,
) -> dict[str, PlayerSkill]:
    """Build per-player skill from per-event SG records with recency decay.

    Records are grouped by player; each contributes ``sg_total`` weighted by
    ``0.5 ** (age_in_events / halflife)``. ``age_in_events`` is the player's
    own record ordinal (0 = most recent) unless ``as_of_index`` supplies a
    global event ordering. The decayed mean is sample-shrunk toward replacement
    by the effective (decay-weighted) round count.
    """
    cfg = cfg or SkillConfig()
    by_player: dict[str, list[SGRecord]] = {}
    for r in records:
        if r.sg_total is None:
            continue
        by_player.setdefault(r.player, []).append(r)

    out: dict[str, PlayerSkill] = {}
    for name, recs in by_player.items():
        # Most-recent-first ordering: prefer a global index, else input order
        # reversed (callers pass records oldest→newest or provide the index).
        if as_of_index is not None:
            recs = sorted(recs, key=lambda r: as_of_index.get(r.event, 0), reverse=True)
        else:
            recs = list(reversed(recs))

        wsum = 0.0
        wval = 0.0
        eff_rounds = 0.0
        for age, r in enumerate(recs):
            w = 0.5 ** (age / cfg.decay_halflife_events)
            wsum += w
            wval += w * r.sg_total
            eff_rounds += w * (r.n_rounds or 4)
        if wsum <= 0:
            continue
        decayed_mean = wval / wsum
        reliability = eff_rounds / (eff_rounds + cfg.shrink_rounds)
        skill_sg = reliability * decayed_mean + (1.0 - reliability) * cfg.replacement_sg
        out[name] = PlayerSkill(
            player=name,
            skill_sg=skill_sg,
            round_sd=_round_sd(skill_sg, cfg),
            n_events=len(recs),
            confidence=_confidence(eff_rounds, cfg),
            components={},
        )
    return out


def to_sim_inputs(
    field_names: Sequence[str],
    skills: dict[str, PlayerSkill],
    cfg: Optional[SkillConfig] = None,
) -> tuple[list[str], list[float], list[float]]:
    """Align a tournament field to (names, means, sds) for the simulator.

    Golfers absent from ``skills`` get a low-confidence replacement level. The
    field's mean skill is re-centred to 0 so ``mean_score`` sits on a true
    field-relative scale (the sim only cares about relative means).
    """
    cfg = cfg or SkillConfig()
    names = [canonical(n) for n in field_names]
    raw_skill: list[float] = []
    sds: list[float] = []
    for n in names:
        ps = skills.get(n)
        if ps is None:
            raw_skill.append(cfg.replacement_sg)
            sds.append(_round_sd(cfg.replacement_sg, cfg))
        else:
            raw_skill.append(ps.skill_sg)
            sds.append(ps.round_sd)

    # Re-centre so the FIELD averages 0 SG (skill is relative to THIS field, not
    # the season tour average — a weak Corales field and a major are different
    # baselines).
    field_mean = sum(raw_skill) / len(raw_skill) if raw_skill else 0.0
    means = [-(s - field_mean) for s in raw_skill]  # lower score = better
    return names, means, sds
