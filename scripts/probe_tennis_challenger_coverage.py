"""Reconnaissance: does the Challenger-tennis opportunity exist on our books,
and do our models cover it?

Answers the three questions that gate any Challenger-tennis work (see the
tennis coverage discussion) BEFORE we build anything:

  Q1  Does Kalshi (KXATPMATCH / KXWTAMATCH) actually list sub-tour matches?
      → groups every open tennis market by `competition` and classifies the
        tier (main tour / challenger / ITF-125 / unknown).

  Q2  Is there a Pinnacle sharp anchor for those matches?
      → the whole pipeline devigs Pinnacle; with no anchor a game is NEVER
        priced (not even shadow). Cross-matches each Kalshi market to the
        Pinnacle tennis feed by surname.

  Q3  Do our models cover the Challenger field?
      → runs the FOUR full-blend-required agents' real predict_pair against
        each market (surface / serve_return / form / advanced) and counts how
        many fire. `REQUIRED_BLEND_MODELS[tennis]` needs all four for a LIVE
        play; anything short demotes to shadow.

Read-only. No DB writes, no persistence. Run on the prod box (needs Kalshi +
Pinnacle + the seeded tennis state; the network is blocked in CI/cloud).

Usage:
    uv run python scripts/probe_tennis_challenger_coverage.py
"""
from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.clients.kalshi import KalshiClient  # noqa: E402
from evmax.clients.esports_pinnacle import PinnacleGuestClient  # noqa: E402
from evmax.models.market import MarketType  # noqa: E402
from evmax.models.odds import SharpBook, SharpOdds  # noqa: E402
from evmax.agents.models.tennis_model_agent import TennisModelAgent  # noqa: E402
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent  # noqa: E402
from evmax.agents.models.tennis_form_agent import TennisFormAgent  # noqa: E402
from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent  # noqa: E402

# Tier classification from the Kalshi `competition` string (e.g. "ATP Munich",
# "ATP Challenger Lima", "ITF W15 Cairo"). Lowercased substring match.
_CHALLENGER_KW = ("challenger", "chall", " ch ", "125")
_ITF_KW = ("itf", "w15", "w25", "w35", "w50", "w75", "w100", "m15", "m25")


def _tier(competition: str | None) -> str:
    c = (competition or "").lower().strip()
    if not c:
        return "unknown(no competition)"
    if any(k in c for k in _ITF_KW):
        return "itf/125"
    if any(k in c for k in _CHALLENGER_KW):
        return "challenger"
    if c.startswith("atp") or c.startswith("wta"):
        return "main tour"
    return f"other: {c[:24]}"


def _surname(name: str | None) -> str:
    n = (name or "").lower().strip()
    return n.split()[-1] if n else ""


def _pair_key(a: str | None, b: str | None) -> frozenset[str]:
    return frozenset({_surname(a), _surname(b)}) - {""}


async def main() -> int:
    # --- Fetch both books ---------------------------------------------------
    async with KalshiClient() as k:
        markets = await k.get_markets("tennis", status="open", limit=1000)
    markets = [m for m in markets if m.market_type == MarketType.moneyline]

    try:
        async with PinnacleGuestClient() as p:
            sharps = await p.get_odds("tennis")
    except Exception as e:  # noqa: BLE001
        print(f"  ! Pinnacle tennis fetch failed: {e}", file=sys.stderr)
        sharps = []
    pinn_pairs = {_pair_key(s.outcome_a_label, s.outcome_b_label) for s in sharps}
    pinn_pairs.discard(frozenset())

    print(f"Kalshi open tennis moneyline markets: {len(markets)}")
    print(f"Pinnacle tennis matchups returned:    {len(sharps)}\n")

    # --- Q1: tier breakdown -------------------------------------------------
    by_tier: dict[str, list] = defaultdict(list)
    comps: dict[str, int] = defaultdict(int)
    for m in markets:
        by_tier[_tier(m.competition)].append(m)
        comps[m.competition or "(none)"] += 1

    print("=== Q1  Tier breakdown (Kalshi competition) ===")
    for tier in sorted(by_tier, key=lambda t: -len(by_tier[t])):
        print(f"  {tier:<22} {len(by_tier[tier]):>4} markets")
    print("  distinct competitions:")
    for comp, n in sorted(comps.items(), key=lambda kv: -kv[1])[:25]:
        print(f"    {n:>3}  {comp}")

    sub_tour = by_tier.get("challenger", []) + by_tier.get("itf/125", [])
    if not sub_tour:
        print("\nVERDICT: Kalshi lists NO sub-tour (challenger/ITF) tennis markets "
              "under KXATPMATCH/KXWTAMATCH. The challenger-edge thesis has no "
              "surface on our current venues — stop here (or probe PolyUS / a "
              "different Kalshi series before building coverage).")
        return 0

    # --- Q2 + Q3 on the sub-tour markets -----------------------------------
    surface = TennisModelAgent()
    serve = TennisServeReturnAgent()
    form = TennisFormAgent()
    adv = TennisAdvancedStatsAgent()
    required = [("surface", surface), ("serve_return", serve),
                ("form", form), ("advanced", adv)]

    def _fake_sharp(m) -> SharpOdds:
        # Coverage only depends on player labels + market.competition; the sharp
        # probability doesn't gate whether a model FIRES. 0.5 is a safe stand-in.
        return SharpOdds(
            event_id=f"tennis::probe::{m.id}",
            book=SharpBook.pinnacle, sector="tennis",
            outcome_a_label=m.team_home, outcome_b_label=m.team_away,
            outcome_a_decimal=2.0, outcome_b_decimal=2.0,
            true_prob_a=0.5, true_prob_b=0.5,
        )

    anchored = 0
    full_cover = 0
    fire_counts: dict[str, int] = defaultdict(int)
    partial_examples: list[str] = []

    print(f"\n=== Q2/Q3  Sub-tour coverage ({len(sub_tour)} challenger+ITF markets) ===")
    for m in sub_tour:
        has_anchor = _pair_key(m.team_home, m.team_away) in pinn_pairs
        anchored += has_anchor
        sharp = _fake_sharp(m)
        fired = []
        for name, agent in required:
            try:
                pred = await agent.predict_pair(m, sharp)
            except Exception:  # noqa: BLE001
                pred = None
            if pred is not None:
                fire_counts[name] += 1
                fired.append(name)
        if len(fired) == 4:
            full_cover += 1
        elif len(partial_examples) < 12:
            miss = [n for n, _ in required if n not in fired]
            anchor_tag = "anchored" if has_anchor else "NO-ANCHOR"
            partial_examples.append(
                f"    {(m.competition or '?')[:22]:<22} "
                f"{_surname(m.team_home)}/{_surname(m.team_away):<14} "
                f"fires={len(fired)}/4 missing={','.join(miss)} [{anchor_tag}]"
            )

    n = len(sub_tour)
    print(f"  Pinnacle-anchored:        {anchored}/{n}  "
          f"({100*anchored/n:.0f}%) — unanchored can never be priced")
    print(f"  full 4-model coverage:    {full_cover}/{n}  "
          f"({100*full_cover/n:.0f}%) — would clear the full-blend gate")
    print("  per-model fire rate on sub-tour:")
    for name, _ in required:
        print(f"    {name:<14} {fire_counts[name]:>4}/{n}  ({100*fire_counts[name]/n:.0f}%)")
    if partial_examples:
        print("  sample gaps:")
        for line in partial_examples:
            print(line)

    # --- Decision -----------------------------------------------------------
    priceable = min(anchored, full_cover)  # both conditions needed to bet live
    print("\n=== DECISION ===")
    print(f"  {n} sub-tour markets · {anchored} anchored · {full_cover} fully covered.")
    if anchored == 0:
        print("  Pinnacle does NOT anchor these — no sharp reference, so they can't be")
        print("  priced at all. A challenger play would need a standalone model the")
        print("  codebase has never trusted. NOT worth the coverage build yet.")
    elif full_cover / max(1, anchored) < 0.3:
        print("  Anchored games exist but our models barely cover them → the win is a")
        print("  COVERAGE build (WTA challenger source + surface-Elo-from-matchmx), then")
        print("  measure the shadow CLV before relaxing the full-blend gate.")
    else:
        print("  Anchored AND covered games exist today → they are already logging as")
        print("  shadow. Build `cleanup shadow clv-tiers tennis` and read the CLV before")
        print("  promoting — the disciplined path, not a blanket gate relaxation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
