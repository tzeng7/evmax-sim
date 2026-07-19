"""Seed PitcherModelAgent state from Baseball Reference via pybaseball.

Computes FIP from box-score totals (HR, BB, HBP, K, IP) using the standard
formula with a year-specific cFIP constant tuned to the season's actual
league-average ERA — so the resulting FIP values land on the same numeric
scale as ERA (a 3.50 FIP reads like a 3.50 ERA).

This replaces the hand-curated dict in scripts/seed_espn.py::seed_pitchers
with an automated pull of every qualified MLB pitcher.

v2 additions (2026-07-19, WS2a):
  - RELIEVERS: pitchers with >= 3 relief appearances (G − GS) or GS/G < 0.5
    and IP >= 5 are written to state["relievers"] as RAW counting stats
    (ip/er/bb/so/hr + team) — the bullpen model (evmax/models_ml/bullpen.py)
    recomputes FIP from counts at predict time against the league reliever
    cFIP, which is also seeded (state["league_pen_cfip"]).
  - xERA: Baseball Savant expected-stats (evmax/clients/savant.py) joined by
    ascii-folded name; each pitcher gets "xera" / "xera_bip" (current season)
    and "xera_prior" (previous season) for the thin-sample fallback. Savant
    fetch failure leaves the fields absent — the agent degrades to FIP/ERA.

Usage:
    python scripts/seed_pitcher_fip.py                # default: 2026, IP >= 30
    python scripts/seed_pitcher_fip.py --year 2025
    python scripts/seed_pitcher_fip.py --min-ip 50
    python scripts/seed_pitcher_fip.py --skip-savant  # offline: no xERA attach
    python scripts/seed_pitcher_fip.py --dry-run      # preview without writing

Resilience: pybaseball scrapes Baseball Reference. If BR's HTML changes or
returns non-200, this script exits non-zero and leaves existing state alone
— the agent silently degrades to whatever was previously seeded.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import pybaseball as pb

# Make sure we can import from the repo root when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evmax.agents.models.pitcher_agent import PitcherModelAgent  # noqa: E402

# Map (BR city, BR league) → ESPN-style team short name (matches the keys
# PitcherModelAgent stores in team_starters and matches against ESPN
# scoreboard's shortDisplayName.lower()).
TEAM_MAP: dict[tuple[str, str], str] = {
    ("Arizona", "Maj-NL"): "diamondbacks",
    ("Atlanta", "Maj-NL"): "braves",
    ("Athletics", "Maj-AL"): "athletics",
    ("Baltimore", "Maj-AL"): "orioles",
    ("Boston", "Maj-AL"): "red sox",
    ("Chicago", "Maj-AL"): "white sox",
    ("Chicago", "Maj-NL"): "cubs",
    ("Cincinnati", "Maj-NL"): "reds",
    ("Cleveland", "Maj-AL"): "guardians",
    ("Colorado", "Maj-NL"): "rockies",
    ("Detroit", "Maj-AL"): "tigers",
    ("Houston", "Maj-AL"): "astros",
    ("Kansas City", "Maj-AL"): "royals",
    ("Los Angeles", "Maj-AL"): "angels",
    ("Los Angeles", "Maj-NL"): "dodgers",
    ("Miami", "Maj-NL"): "marlins",
    ("Milwaukee", "Maj-NL"): "brewers",
    ("Minnesota", "Maj-AL"): "twins",
    ("New York", "Maj-AL"): "yankees",
    ("New York", "Maj-NL"): "mets",
    ("Philadelphia", "Maj-NL"): "phillies",
    ("Pittsburgh", "Maj-NL"): "pirates",
    ("San Diego", "Maj-NL"): "padres",
    ("San Francisco", "Maj-NL"): "giants",
    ("Seattle", "Maj-AL"): "mariners",
    ("St. Louis", "Maj-NL"): "cardinals",
    ("Tampa Bay", "Maj-AL"): "rays",
    ("Texas", "Maj-AL"): "rangers",
    ("Toronto", "Maj-AL"): "blue jays",
    ("Washington", "Maj-NL"): "nationals",
}


def _decode_br_escapes(name: str) -> str:
    """Baseball Reference returns accented names as literal escape-sequence
    text — e.g., 'Cristopher S\\xc3\\xa1nchez' with backslash-x-HH as actual
    characters in the string, not as unicode codepoints.

    This re-interprets those literals as UTF-8 bytes:
        latin-1 encode → unicode_escape decode → latin-1 encode → utf-8 decode

    Pure-ASCII names pass through unchanged.
    """
    try:
        return (
            name.encode("latin-1")
            .decode("unicode_escape")
            .encode("latin-1")
            .decode("utf-8")
        )
    except (UnicodeError, ValueError):
        return name


def _ascii_fold(name: str) -> str:
    """Decode BR escape sequences then strip accents.

    'Cristopher S\\xc3\\xa1nchez' → 'Cristopher Sánchez' → 'cristopher sanchez'

    ESPN's scoreboard returns ASCII-only names, so the seed must fold to
    ASCII or the live-starter lookup will miss accented players.
    """
    decoded = _decode_br_escapes(name)
    nfkd = unicodedata.normalize("NFKD", decoded)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _league_cfip(df: pd.DataFrame) -> tuple[float, float]:
    """Return (cFIP, league_avg_era) computed from the filtered DataFrame.

    cFIP = league_ERA - league_FIP_raw, where FIP_raw is the unconstanted
    formula. This makes our per-pitcher FIP values directly comparable to
    their ERA on the same numeric scale.
    """
    total_ip = df["IP"].sum()
    total_hr = df["HR"].sum()
    total_bb = df["BB"].sum()
    total_hbp = df["HBP"].sum()
    total_so = df["SO"].sum()
    total_er = df["ER"].sum()

    league_era = (total_er * 9.0) / total_ip
    league_fip_raw = (13 * total_hr + 3 * (total_bb + total_hbp) - 2 * total_so) / total_ip
    return league_era - league_fip_raw, league_era


def _pitcher_fip(row: pd.Series, cfip: float) -> float:
    if row["IP"] <= 0:
        return float("nan")
    return (
        13 * row["HR"] + 3 * (row["BB"] + row["HBP"]) - 2 * row["SO"]
    ) / row["IP"] + cfip


RELIEVER_MIN_IP = 5.0       # per-reliever IP floor (mirrors bullpen.RELIEVER_MIN_IP)
RELIEVER_MIN_RELIEF_G = 3   # >= 3 relief appearances (G − GS) classifies a reliever
XERA_THIN_BIP = 40          # below this current-season BIP, the agent prefers xera_prior


def classify_relievers(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rows whose usage pattern says reliever: G−GS >= 3 or GS/G < 0.5.

    Swingmen (openers, spot starters) can satisfy both roles; counting them
    in the pen is correct — they are available relief innings.
    """
    g = df["G"].astype(float)
    gs = df["GS"].astype(float)
    relief_g = g - gs
    frac_started = gs / g.where(g > 0, other=1)
    mask = (relief_g >= RELIEVER_MIN_RELIEF_G) | (frac_started < 0.5)
    return df[mask & (df["IP"] >= RELIEVER_MIN_IP)].copy()


def _attach_xera(records: dict[str, dict], year: int) -> tuple[int, int]:
    """Join Savant xERA onto pitcher records by ascii-folded name.

    Attaches "xera" + "xera_bip" from the current season and "xera_prior"
    from the previous one. Returns (n_current, n_prior) matched. Fetch
    failure returns (0, 0) and leaves records untouched — the agent
    degrades to FIP/ERA.
    """
    import asyncio

    from evmax.clients.savant import fetch_expected_stats

    async def _pull() -> tuple[dict, dict]:
        cur = await fetch_expected_stats(year)
        prior = await fetch_expected_stats(year - 1)
        return cur, prior

    cur, prior = asyncio.run(_pull())
    cur_by_name = {rec["name_key"]: rec for rec in cur.values()}
    prior_by_name = {rec["name_key"]: rec for rec in prior.values()}

    n_cur = n_prior = 0
    for name, rec in records.items():
        c = cur_by_name.get(name)
        if c:
            rec["xera"] = c["xera"]
            rec["xera_bip"] = c["bip"]
            n_cur += 1
        p = prior_by_name.get(name)
        if p:
            rec["xera_prior"] = p["xera"]
            n_prior += 1
    return n_cur, n_prior


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--min-ip",
        type=float,
        default=30.0,
        help="Minimum IP to include a STARTER (default: 30, which catches everyone "
        "with at least ~5-6 starts). Relievers use their own 5-IP floor.",
    )
    parser.add_argument(
        "--skip-savant", action="store_true",
        help="Skip the Savant xERA attach (offline runs).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Fetching pitching_stats_bref({args.year}) from Baseball Reference...")
    try:
        df = pb.pitching_stats_bref(args.year)
    except Exception as e:
        print(f"ERROR: pybaseball BR fetch failed: {e}", file=sys.stderr)
        print("State was NOT modified; existing pitcher_state.json is intact.", file=sys.stderr)
        return 1

    # MLB only (drop minor-league rows).
    df = df[df["Lev"].str.contains("Maj", na=False)].copy()
    # Drop multi-team season totals — these have comma-separated Tm values.
    df = df[~df["Tm"].astype(str).str.contains(",", na=False)].copy()
    # Drop cross-league moves (Lev like "Maj-AL,Maj-NL").
    df = df[~df["Lev"].astype(str).str.contains(",", na=False)].copy()

    df["team_canonical"] = df.apply(
        lambda r: TEAM_MAP.get((r["Tm"], r["Lev"])), axis=1
    )
    unmapped = sorted(df[df["team_canonical"].isna()]["Tm"].unique().tolist())
    if unmapped:
        print(f"WARN: dropped pitchers with unmapped (city, league) combos: {unmapped}")
    df = df[df["team_canonical"].notna()].copy()

    # ---- Relievers: raw counting stats for the bullpen model --------------
    rel_df = classify_relievers(df)
    relievers: dict[str, dict] = {}
    pen_totals = {"ip": 0.0, "er": 0, "bb": 0, "so": 0, "hr": 0}
    for _, r in rel_df.iterrows():
        name = _ascii_fold(str(r["Name"]))
        relievers[name] = {
            "team": r["team_canonical"],
            "ip": float(r["IP"]),
            "er": int(r["ER"]),
            "bb": int(r["BB"]),
            "so": int(r["SO"]),
            "hr": int(r["HR"]),
        }
        pen_totals["ip"] += float(r["IP"])
        for k in ("er", "bb", "so", "hr"):
            pen_totals[k] += int(r[k.upper()])

    from evmax.models_ml.bullpen import league_pen_cfip  # noqa: E402

    pen_cfip = league_pen_cfip(pen_totals)
    print(
        f"Relievers: {len(relievers)} (G−GS>={RELIEVER_MIN_RELIEF_G} or GS/G<0.5, "
        f"IP>={RELIEVER_MIN_IP:g})  ·  league pen cFIP: {pen_cfip:.3f}"
    )

    # ---- Starters: FIP-blended quality (the v1 path, unchanged) -----------
    df = df[df["IP"] >= args.min_ip].copy()
    if df.empty:
        print(f"ERROR: no pitchers met the IP >= {args.min_ip} threshold for {args.year}.", file=sys.stderr)
        print("Try lowering --min-ip or check that the season has data yet.", file=sys.stderr)
        return 1

    cfip, league_avg_era = _league_cfip(df)
    df["FIP"] = df.apply(lambda r: _pitcher_fip(r, cfip), axis=1)
    print(f"League cFIP: {cfip:.3f}  ·  League avg ERA: {league_avg_era:.2f}  ·  {len(df)} pitchers above {args.min_ip} IP")

    # Sort by team then GS descending so the most-starts pitcher per team
    # is first in the iteration order. PitcherModelAgent.seed_pitchers
    # uses dict insertion order: the first pitcher per team becomes the
    # team_starters entry (which we want to be the workhorse starter).
    df = df.sort_values(["team_canonical", "GS"], ascending=[True, False])

    pitchers: dict[str, dict] = {}
    for _, r in df.iterrows():
        name = _ascii_fold(str(r["Name"]))
        pitchers[name] = {
            "era": float(r["ERA"]),
            "fip": float(r["FIP"]),
            "ip": float(r["IP"]),
            "team": r["team_canonical"],
        }

    teams_seen = sorted(set(p["team"] for p in pitchers.values()))
    print(f"Mapped {len(pitchers)} pitchers across {len(teams_seen)} teams")
    if len(teams_seen) < 30:
        missing = set(TEAM_MAP.values()) - set(teams_seen)
        print(f"WARN: teams with no qualifying pitcher: {sorted(missing)}")

    # ---- xERA attach (starters get the blend input; relievers informative)
    if not args.skip_savant:
        try:
            n_cur, n_prior = _attach_xera(pitchers, args.year)
            cov = n_cur / len(pitchers) * 100 if pitchers else 0
            print(f"xERA: {n_cur} current-season ({cov:.0f}% of starters), {n_prior} prior-season matched")
            if cov < 60:
                print("WARN: xERA coverage below 60% — check the Savant join; "
                      "unmatched pitchers degrade to FIP/ERA.")
        except Exception as e:  # noqa: BLE001 — Savant failure never blocks the seed
            print(f"WARN: Savant xERA attach failed ({e}); old xera fields "
                  "are preserved for pitchers that keep their records.")

    print("\nTop 10 by FIP:")
    top = sorted(pitchers.items(), key=lambda x: x[1]["fip"])[:10]
    for n, d in top:
        xe = f"  xERA={d['xera']:>5.2f}" if "xera" in d else ""
        print(f"  {n:<28}  FIP={d['fip']:>5.2f}  ERA={d['era']:>5.2f}  IP={d['ip']:>6.1f}  ({d['team']}){xe}")

    if args.dry_run:
        print("\n(dry-run: state file NOT written)")
        return 0

    agent = PitcherModelAgent()
    # Clear stale entries so retired pitchers + outdated team assignments
    # don't linger. The agent's seed_pitchers preserves first-encountered
    # starter per team, which we want only when applied to fresh data.
    agent._state["pitchers"] = {}
    agent._state["team_starters"] = {}
    agent.seed_pitchers(pitchers, league_avg_era=round(league_avg_era, 2))
    # Bullpen state rides the same file: raw reliever counts + league pen
    # cFIP. seed_pitchers already saved; write the pen block and save again.
    from datetime import date as _date

    agent._state["relievers"] = relievers
    agent._state["league_pen_cfip"] = round(pen_cfip, 3)
    agent._state["pen_seeded_at"] = _date.today().isoformat()
    agent.save_state()
    print(f"\nWrote {len(pitchers)} starters + {len(relievers)} relievers to {agent._state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
