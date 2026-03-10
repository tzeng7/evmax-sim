"""Soccer diagnostic v2: compare Kalshi and Pinnacle keys side by side."""
import asyncio
import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

from evmax.clients.kalshi import KalshiClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.matching.normalizer import NameNormalizer
from rapidfuzz import fuzz


async def main() -> None:
    norm = NameNormalizer("soccer")

    async with KalshiClient() as kalshi:
        markets = await kalshi.get_markets("soccer")

    async with PinnacleClient() as pinnacle:
        sharp_list = await pinnacle.get_odds("soccer")

    # Build Kalshi canonical keys
    kalshi_keys: dict[str, object] = {}
    for m in markets:
        if m.event_date and m.team_home and m.team_away:
            key = norm.normalize_event_key(
                m.team_home, m.team_away, str(m.event_date)[:10], "soccer"
            )
            kalshi_keys[key] = m

    # Build Pinnacle canonical keys
    pinnacle_keys: dict[str, object] = {}
    for s in sharp_list:
        pinnacle_keys[s.event_id] = s

    print(f"Kalshi unique event keys : {len(kalshi_keys)}")
    print(f"Pinnacle unique event keys: {len(pinnacle_keys)}")

    # Exact matches
    exact = kalshi_keys.keys() & pinnacle_keys.keys()
    print(f"\nExact matches: {len(exact)}")
    for k in sorted(exact):
        print(f"  EXACT: {k}")

    # Fuzzy candidates (threshold 80+ among remaining)
    remaining_k = [k for k in kalshi_keys if k not in exact]
    remaining_p = list(pinnacle_keys.keys())

    print(f"\nFuzzy check for {len(remaining_k)} unmatched Kalshi keys...")
    for kk in sorted(remaining_k):
        best_score = 0
        best_pk = ""
        for pk in remaining_p:
            # Only compare same date
            kdate = kk.split("::")[1]
            pdate = pk.split("::")[1]
            if abs((len(kdate) and len(pdate))) > 0:  # always true, just check date diff
                from datetime import date
                try:
                    kd = date.fromisoformat(kdate)
                    pd = date.fromisoformat(pdate)
                    if abs((kd - pd).days) > 1:
                        continue
                except Exception:
                    pass
            score = fuzz.token_sort_ratio(kk, pk)
            if score > best_score:
                best_score = score
                best_pk = pk
        if best_score >= 80:
            print(f"  FUZZY {best_score:3.0f}: {kk}")
            print(f"         vs {best_pk}")

    # Show all Pinnacle UCL-era events (Feb 24-25)
    print("\n=== Pinnacle events on Feb 24-25 ===")
    for pk in sorted(pinnacle_keys.keys()):
        if "2026-02-24" in pk or "2026-02-25" in pk:
            s = pinnacle_keys[pk]
            print(f"  {pk}  [{s.outcome_a_label} vs {s.outcome_b_label}]")


if __name__ == "__main__":
    asyncio.run(main())
