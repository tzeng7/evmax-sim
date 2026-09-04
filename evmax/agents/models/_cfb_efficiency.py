"""Shared math core for the NCAA-football opponent-adjusted EPA efficiency model.

This is a THIRD parallel efficiency stack, deliberately importing nothing from
the NBA/WNBA or college-basketball cores. Football efficiency is NOT basketball
efficiency:

  * The signal is Expected Points Added per play (EPA/play), which is SIGNED and
    roughly zero-mean — not a positive points-per-100 ratio. So the opponent
    adjustment is an ADDITIVE ridge least-squares model, not the multiplicative
    log-space solver the basketball stack uses.
  * College football's defining problem is schedule-strength variance (FCS buy
    games, conference mismatches) plus tiny in-season samples (12–13 games). The
    literature (SP+, FPI) answers both with heavy PRESEASON-PRIOR shrinkage that
    ramps off as in-season data accrues. That ramp is baked into the model here
    (prior_ramp_weight) and consumed at predict time, so early-season games lean
    on the prior and week-9 games lean on opponent-adjusted EPA.

Pipeline: ESPN play-by-play → expected-points table → per-play EPA → per
game-side offensive/defensive EPA/play (garbage-time filtered) → additive ridge
opponent adjustment → league-relative off/def ratings → (+ regressed prior).

Consumers:
  - scripts/seed_ncaaf_efficiency.py       (builds data/models/ncaaf_efficiency_state.json)
  - scripts/backtest_ncaaf_efficiency.py   (walk-forward validation)
  - evmax/agents/models/ncaaf_efficiency_agent.py
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np

from evmax.models_ml._math import normal_cdf as _normal_cdf

# ---------------------------------------------------------------------------
# Season calendar
# ---------------------------------------------------------------------------

# Months a CFB season is live: late Aug → mid Jan (CFP national championship).
# Wraps the year boundary, so an explicit month set (mirrors the NFL guard).
CFB_SEASON_MONTHS = {8, 9, 10, 11, 12, 1}

# Canonical id for the pooled FCS / non-FBS opponent (buy games). Every team not
# in the FBS universe collapses to this single pseudo-team in the solver so a
# 56-3 win over an FCS side is credited against a weak pooled defense, not
# treated as evidence the FBS offense is elite.
FCS_POOL_ID = "__fcs__"


def cfb_season_start_year(d: date) -> int:
    """Season label = the fall calendar year the season starts in.

    Aug–Dec belong to that year's season; a Jan CFP game belongs to the season
    that started the prior calendar year. Aug+ counts toward the upcoming
    season so a fresh preseason seed labels itself correctly.
    """
    return d.year if d.month >= 7 else d.year - 1


def state_is_stale_for_today(state: dict, today: Optional[date] = None) -> bool:
    """True when the state's source season is behind the active CFB season
    during the season window.

    Same guard the WNBA/NFL/college-basketball stacks use (the +24pp chalk-bias
    lesson): during Aug–Jan, a frozen prior-season seed must not fire. College
    football roster turnover (transfer portal + NFL draft + graduation) makes
    this even more important than the NFL. Outside the window the guard relaxes.
    Missing/unparseable source season disables the guard (back-compat).
    """
    today = today or date.today()
    if today.month not in CFB_SEASON_MONTHS:
        return False
    raw = (state.get("ncaaf") or {}).get("source_season")
    if raw is None:
        return False
    try:
        return int(raw) < cfb_season_start_year(today)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Expected Points model
# ---------------------------------------------------------------------------
#
# EP(state) = expected value of the NEXT score in the half, from the perspective
# of the team with the ball (offense positive, opponent's next score negative).
# State is binned by (down, yards-to-goal bucket). Distance-to-first is folded
# into a coarse long/short flag only on later downs, where it matters most and
# the sample supports it. This is the classic Carter/Romer/Burke construction.

# 10-yard field-position buckets, 0 (own goal-line, 99 yds to go) → 100 (scoring).
YTG_BUCKET_YARDS = 10
N_YTG_BUCKETS = 10  # 0-9 (99-90 to go) … up to the opponent goal line


def _ytg_bucket(yards_to_goal: float) -> int:
    """Bucket yards-to-opponent-goal into [0, N_YTG_BUCKETS-1].

    yards_to_goal is ESPN's ``yardsToEndzone`` (distance to the OFFENSE's
    scoring end zone). Small = close to scoring.
    """
    b = int(yards_to_goal // YTG_BUCKET_YARDS)
    return max(0, min(N_YTG_BUCKETS - 1, b))


def _ep_state_key(down: int, yards_to_goal: float, distance: float) -> Optional[str]:
    """A hashable EP state bin, or None if the state is not a normal scrimmage
    down (kickoffs, PATs, malformed states carry down<=0 or down>4)."""
    if down is None or down < 1 or down > 4:
        return None
    bucket = _ytg_bucket(yards_to_goal)
    # Long/short flag only on 3rd/4th down (where distance drives EP most and the
    # bin still has enough samples). 1st/2nd collapse the distance dimension.
    if down >= 3:
        ls = "L" if (distance or 0) >= 7 else "S"
        return f"{down}:{bucket}:{ls}"
    return f"{down}:{bucket}"


def build_ep_table(plays: list[dict]) -> dict:
    """Fit EP(state) = mean signed next-score value, from historical plays.

    Each play row needs: game_id, half (1/2), off_team, down, distance,
    yards_to_goal, and — for scoring plays — the eventual next-score fields are
    derived here by scanning forward within (game_id, half).

    `plays` MUST be in chronological order within each game. Returns
    {"ep": {state_key: ep_value}, "n": {state_key: count}, "default_by_bucket":
    {bucket: ep}} — the last is a fallback for unseen (down, bucket) states.

    Next-score value convention: +7 offense TD, +3 offense FG, −7/−3 for the
    opponent's next score, 0 if the half ends with no further score. (PAT/2pt and
    safeties are folded into the TD/other buckets; the ±7/±3 approximation is the
    standard EP simplification and washes out in EPA differences.)
    """
    # 1) Attach next-score value to each play by scanning forward per half.
    from collections import defaultdict

    by_half: dict[tuple, list[dict]] = defaultdict(list)
    for p in plays:
        by_half[(p["game_id"], p["half"])].append(p)

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    bucket_sums: dict[int, float] = defaultdict(float)
    bucket_counts: dict[int, int] = defaultdict(int)

    for (_gid, _half), half_plays in by_half.items():
        # Find scoring events in order: (index, scoring_team, points).
        scores = [
            (i, p["off_team"] if p.get("score_off") else p.get("score_team"), p["score_points"])
            for i, p in enumerate(half_plays)
            if p.get("score_points")
        ]
        for i, p in enumerate(half_plays):
            key = _ep_state_key(p.get("down"), p.get("yards_to_goal"), p.get("distance"))
            if key is None:
                continue
            # next score strictly AFTER this play (>= i so a scoring play counts
            # its own score as the next score)
            nxt = next((s for s in scores if s[0] >= i), None)
            if nxt is None:
                val = 0.0
            else:
                _, team, pts = nxt
                val = pts if team == p["off_team"] else -pts
            sums[key] += val
            counts[key] += 1
            bucket_sums[_ytg_bucket(p["yards_to_goal"])] += val
            bucket_counts[_ytg_bucket(p["yards_to_goal"])] += 1

    ep = {k: sums[k] / counts[k] for k in sums if counts[k] > 0}
    default_by_bucket = {
        b: bucket_sums[b] / bucket_counts[b] for b in bucket_sums if bucket_counts[b] > 0
    }
    return {"ep": ep, "n": dict(counts), "default_by_bucket": default_by_bucket}


def ep_value(table: dict, down: int, yards_to_goal: float, distance: float) -> float:
    """EP for a state, with graceful fallback to the (down-agnostic) bucket mean."""
    key = _ep_state_key(down, yards_to_goal, distance)
    ep = table["ep"]
    if key is not None and key in ep:
        return ep[key]
    # Fallback 1: same down/bucket without the long/short flag.
    if down and 1 <= down <= 4:
        base = f"{down}:{_ytg_bucket(yards_to_goal)}"
        if base in ep:
            return ep[base]
    # Fallback 2: field-position bucket mean across downs.
    return table.get("default_by_bucket", {}).get(_ytg_bucket(yards_to_goal), 0.0)


def play_epa(table: dict, play: dict) -> Optional[float]:
    """EPA for one scrimmage play, from the offense's perspective.

    play needs: down, distance, yards_to_goal (start state), off_team; and
    end_down, end_distance, end_yards_to_goal, end_team (post-play state); plus
    scoring fields score_points / score_off when the play itself scored.

    Returns None for non-scrimmage plays (kickoffs, PATs, states with no valid
    start down) so the caller can skip them.
    """
    start_key = _ep_state_key(play.get("down"), play.get("yards_to_goal"), play.get("distance"))
    if start_key is None:
        return None
    ep_start = ep_value(table, play["down"], play["yards_to_goal"], play["distance"])

    # End value from the offense's perspective.
    if play.get("score_points"):
        ep_end = play["score_points"] if play.get("score_off") else -play["score_points"]
    else:
        end_down = play.get("end_down")
        end_ytg = play.get("end_yards_to_goal")
        if end_down is None or end_ytg is None or end_down < 1:
            # Possession ended without a score and without a clean next-down state
            # (punt, end of half) — treat as a change of possession worth roughly
            # the negative of a 1st-and-10 at the flipped field position if known,
            # else neutral. Missing end state → skip.
            return None
        ep_end_off = ep_value(table, end_down, end_ytg, play.get("end_distance") or 10)
        same_team = play.get("end_team") == play.get("off_team")
        ep_end = ep_end_off if same_team else -ep_end_off
    return ep_end - ep_start


def is_success(down: int, distance: float, yards_gained: float) -> bool:
    """Standard football success-rate definition: gain ≥ 50% of yards-to-go on
    1st down, ≥ 70% on 2nd, ≥ 100% on 3rd/4th."""
    if not distance or distance <= 0:
        return yards_gained > 0
    need = {1: 0.5, 2: 0.7}.get(down, 1.0) * distance
    return yards_gained >= need


# Play-type substrings that are NOT offensive scrimmage plays. Standard
# "EPA/play" is measured over scrimmage snaps (rush/pass), excluding special
# teams and administrative plays — this both removes special-teams variance and
# down-weights the highest-variance turnover-return luck (a defensive/return TD
# is not the offense's scrimmage efficiency). Turnovers ON scrimmage plays keep
# their (negative) EPA, which is correct — those are real, somewhat repeatable.
_NON_SCRIMMAGE = (
    "kickoff", "punt", "field goal", "extra point", "two-point", "timeout",
    "end period", "end of", "coin toss", "no play",
)


def is_scrimmage_play(type_text: str) -> bool:
    t = (type_text or "").lower()
    return not any(s in t for s in _NON_SCRIMMAGE)


def aggregate_game_sides(
    plays: list[dict],
    fbs_ids: set[str],
    games_by_id: dict[str, dict],
    ep_table: dict,
    garbage_margin: float = 28.0,
    min_plays: int = 10,
) -> tuple[list[dict], dict]:
    """Reduce per-play rows to per-(game, offense) EPA means + per-team tallies.

    Applies the garbage-time filter (drop 4th-quarter plays when the pre-play
    margin exceeds ``garbage_margin``) and the scrimmage-play filter, pools every
    non-FBS team into FCS_POOL_ID, and tags each game-side with the home/away
    side flag from the game metadata.

    Returns (game_sides, team_tally) where team_tally[id] = {"gp", "off_plays",
    "off_success", "def_plays", "def_success"} for FBS ids only (for success-rate
    secondary signals and game counts). game_sides feed opponent_adjust_epa.
    """
    from collections import defaultdict

    side_epa: dict[tuple, list[float]] = defaultdict(list)
    side_succ: dict[tuple, int] = defaultdict(int)
    side_meta: dict[tuple, tuple] = {}
    tally = defaultdict(lambda: {
        "gp_games": set(), "off_plays": 0, "off_success": 0,
        "def_plays": 0, "def_success": 0,
    })

    for p in plays:
        if not is_scrimmage_play(p.get("type", "")):
            continue
        if p.get("down") is None or p["down"] < 1:
            continue
        # garbage time: 4th quarter (period 4) with a lopsided pre-play margin
        if p.get("period") and p["period"] >= 4 and abs(p.get("off_margin_pre", 0)) > garbage_margin:
            continue
        epa = play_epa(ep_table, p)
        if epa is None:
            continue
        off, dfn = p["off_team"], p["def_team"]
        key = (p["game_id"], off)
        side_epa[key].append(epa)
        side_meta[key] = (off, dfn)
        succ = is_success(p["down"], p.get("distance") or 0, p.get("yards_gained") or 0)
        side_succ[key] += int(succ)
        if off in fbs_ids:
            tally[off]["off_plays"] += 1
            tally[off]["off_success"] += int(succ)
            tally[off]["gp_games"].add(p["game_id"])
        if dfn in fbs_ids:
            tally[dfn]["def_plays"] += 1
            tally[dfn]["def_success"] += int(succ)

    game_sides: list[dict] = []
    for (gid, off), epas in side_epa.items():
        if len(epas) < min_plays:
            continue
        _, dfn = side_meta[(gid, off)]
        g = games_by_id.get(gid)
        if not g:
            continue
        off_pool = off if off in fbs_ids else FCS_POOL_ID
        def_pool = dfn if dfn in fbs_ids else FCS_POOL_ID
        if g.get("neutral"):
            side = 0.0
        else:
            side = 1.0 if off == g["home"]["id"] else -1.0
        game_sides.append({
            "off_id": off_pool,
            "def_id": def_pool,
            "epa": sum(epas) / len(epas),
            # success rate over the SAME filtered scrimmage plays — the second
            # rating dimension of ncaaf_efficiency_v2 (opponent-adjusted by the
            # same solver via value_key="sr").
            "sr": side_succ[(gid, off)] / len(epas),
            "plays": len(epas),
            "side": side,
        })

    team_tally = {}
    for tid, t in tally.items():
        team_tally[tid] = {
            "gp": len(t["gp_games"]),
            "off_success_rate": round(t["off_success"] / t["off_plays"], 4) if t["off_plays"] else 0.0,
            "def_success_rate": round(t["def_success"] / t["def_plays"], 4) if t["def_plays"] else 0.0,
        }
    return game_sides, team_tally


def build_team_ratings(
    plays: list[dict],
    fbs_ids: set[str],
    games_by_id: dict[str, dict],
    ep_table: dict,
    ridge: float = 1.0,
    garbage_margin: float = 28.0,
    min_plays: int = 10,
) -> dict:
    """End-to-end: plays → opponent-adjusted off/def EPA ratings per FBS team.

    Returns {"teams": {id: {"off_epa_adj", "def_epa_adj", "gp",
    "off_success_rate", "def_success_rate"}}, "league_mean_epa", "hfa_epa"}.

    Sign convention matches the NFL agent: off_epa_adj > 0 = better offense;
    def_epa_adj = EPA allowed vs league (LOWER = better defense). So the solver's
    def_allowed rating is stored directly as def_epa_adj and net = off − def.

    v2 (2026-09-03) adds the same solve on per-game-side SUCCESS RATE:
    off_sr_adj / def_sr_adj (league-relative, def = success rate allowed, lower
    = better). Success rate is the one play-level signal that added held-out
    value on top of EPA (2023/2024/2025 walk-forward, scripts/backtest_ncaaf_v2.py);
    explosiveness, pass/rush splits, turnover splits and special teams did not.
    The raw (un-adjusted) tallies are kept for back-compat / diagnostics.
    """
    game_sides, tally = aggregate_game_sides(
        plays, fbs_ids, games_by_id, ep_table, garbage_margin, min_plays
    )
    adj = opponent_adjust_epa(game_sides, ridge=ridge)
    adj_sr = opponent_adjust_epa(game_sides, ridge=ridge, value_key="sr")
    teams: dict[str, dict] = {}
    for tid in fbs_ids:
        t = tally.get(tid, {"gp": 0, "off_success_rate": 0.0, "def_success_rate": 0.0})
        teams[tid] = {
            "off_epa_adj": adj["off"].get(tid, 0.0),
            "def_epa_adj": adj["def_allowed"].get(tid, 0.0),
            "off_sr_adj": adj_sr["off"].get(tid, 0.0),
            "def_sr_adj": adj_sr["def_allowed"].get(tid, 0.0),
            "gp": t["gp"],
            "off_success_rate": t["off_success_rate"],
            "def_success_rate": t["def_success_rate"],
        }
    return {
        "teams": teams,
        "league_mean_epa": adj["league_mean_epa"],
        "hfa_epa": adj["hfa"],
        "league_mean_sr": adj_sr["league_mean_epa"],
        "hfa_sr": adj_sr["hfa"],
    }


# ---------------------------------------------------------------------------
# Additive ridge opponent adjustment on EPA/play
# ---------------------------------------------------------------------------


def opponent_adjust_epa(
    game_sides: list[dict],
    max_iters: int = 2000,
    tol: float = 1e-5,
    ridge: float = 1.0,
    estimate_hfa: bool = True,
    value_key: str = "epa",
) -> dict:
    """Solve league-relative offensive / defensive EPA-per-play ratings.

    ``value_key`` selects the per-game-side response column ("epa" by default;
    "sr" for success rate) — the additive model and solver are identical for
    any zero-mean-ish per-play average, so v2's success-rate ratings reuse it.

    Additive model, one observation per (game, offense side):

        off_epa_obs  ≈  μ  +  off[t]  +  def_allowed[u]  +  hfa · side/2

    where μ is the league mean offensive EPA/play, off[t] > 0 means team t's
    offense beats league average, def_allowed[u] > 0 means opponents move the
    ball better than average against u (a WORSE defense), and side = +1 when the
    offense is at home, −1 away, 0 neutral. Observations are weighted by the
    number of plays behind each game-side mean.

    Solved by ridge-regularized block coordinate descent (off given def,hfa is a
    per-team ridge weighted mean; then def; then hfa), which converges
    monotonically to the unique ridge-least-squares optimum. Ridge pulls each
    rating toward 0 (league average) with `ridge` virtual average plays — this
    both makes the optimum unique on disconnected schedules (the FCS-pool games
    connect the graph) and stabilizes 1–2 game early-season ratings.

    game_sides rows: off_id, def_id, epa (mean off EPA/play that game-side),
    plays (int weight), side (+1/−1/0). Returns:
        {"off": {id: rating}, "def_allowed": {id: rating}, "league_mean_epa": μ,
         "hfa": float (EPA/play home edge, full split), "iterations", "converged"}
    """
    if not game_sides:
        return {"off": {}, "def_allowed": {}, "league_mean_epa": 0.0,
                "hfa": 0.0, "iterations": 0, "converged": True}

    ids: list[str] = []
    index: dict[str, int] = {}

    def _idx(tid: str) -> int:
        if tid not in index:
            index[tid] = len(ids)
            ids.append(tid)
        return index[tid]

    off_i, def_i, epa, w, side = [], [], [], [], []
    for gs in game_sides:
        pl = float(gs.get("plays", 0))
        if pl <= 0:
            continue
        off_i.append(_idx(gs["off_id"]))
        def_i.append(_idx(gs["def_id"]))
        epa.append(float(gs[value_key]))
        w.append(pl)
        side.append(float(gs.get("side", 0.0)))

    n = len(ids)
    if n == 0 or not off_i:
        return {"off": {}, "def_allowed": {}, "league_mean_epa": 0.0,
                "hfa": 0.0, "iterations": 0, "converged": True}

    off_i = np.asarray(off_i, np.int64)
    def_i = np.asarray(def_i, np.int64)
    epa = np.asarray(epa, np.float64)
    w = np.asarray(w, np.float64)
    side = np.asarray(side, np.float64)

    mu = float(np.average(epa, weights=w))
    y = epa - mu  # response relative to league mean

    a = np.zeros(n)  # off ratings
    d = np.zeros(n)  # def_allowed ratings
    h = 0.0
    home_mask = side > 0
    away_mask = side < 0
    can_hfa = estimate_hfa and bool(home_mask.any()) and bool(away_mask.any())

    # Per-team ridge-weighted denominators.
    wsum_off = np.zeros(n)
    np.add.at(wsum_off, off_i, w)
    wsum_def = np.zeros(n)
    np.add.at(wsum_def, def_i, w)
    denom_off = wsum_off + ridge
    denom_def = wsum_def + ridge

    converged = False
    iterations = 0
    for iterations in range(1, max_iters + 1):
        # off block: minimize Σ w (y − a[t] − d[u] − h·side/2)² + ridge·a²
        resid = y - d[def_i] - h * side / 2.0
        num = np.zeros(n)
        np.add.at(num, off_i, w * resid)
        a_new = num / denom_off
        # def block
        resid = y - a_new[off_i] - h * side / 2.0
        num = np.zeros(n)
        np.add.at(num, def_i, w * resid)
        d_new = num / denom_def
        # hfa block (weighted): mean home residual minus mean away residual
        h_new = h
        if can_hfa:
            r = y - a_new[off_i] - d_new[def_i]
            hw = w[home_mask]
            aw = w[away_mask]
            h_new = float(
                np.average(r[home_mask], weights=hw) - np.average(r[away_mask], weights=aw)
            )
        delta = max(
            float(np.max(np.abs(a_new - a))),
            float(np.max(np.abs(d_new - d))),
            abs(h_new - h) / 2.0,
        )
        a, d, h = a_new, d_new, h_new
        if delta < tol:
            converged = True
            break

    off = {ids[i]: round(float(a[i]), 5) for i in range(n)}
    def_allowed = {ids[i]: round(float(d[i]), 5) for i in range(n)}
    return {
        "off": off,
        "def_allowed": def_allowed,
        "league_mean_epa": round(mu, 5),
        "hfa": round(float(h), 5),
        "iterations": iterations,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Preseason prior ramp
# ---------------------------------------------------------------------------


def prior_ramp_weight(gp: int, k: float) -> float:
    """In-season weight w(gp) = gp / (gp + k); prior gets (1 − w).

    k is the "half-life" game count: at gp == k the split is 50/50. The CFB
    literature leans ~60–70% prior through week 3 and phases it out by week 9;
    k ≈ 3–4 reproduces that (week 3 gp=3 → ~0.5 with k=3). Swept in the backtest.
    """
    gp = max(0, int(gp))
    return gp / (gp + k) if (gp + k) > 0 else 0.0


def blended_net(stats: dict, k: float) -> float:
    """Net EPA/play rating = off_epa_adj − def_epa_adj, blending in-season with
    the regressed preseason prior by the game-count ramp.

    stats carries off_epa_adj/def_epa_adj (in-season, league-relative) and
    off_epa_prior/def_epa_prior (regressed prior-season net, 0 for newcomers).
    def_epa_adj / def_epa_prior are "EPA allowed vs league" (lower = better),
    matching the NFL agent, so net = off − def.
    """
    gp = stats.get("gp", 0)
    w = prior_ramp_weight(gp, k)
    net_in = stats.get("off_epa_adj", 0.0) - stats.get("def_epa_adj", 0.0)
    net_prior = stats.get("off_epa_prior", 0.0) - stats.get("def_epa_prior", 0.0)
    return w * net_in + (1.0 - w) * net_prior


# ---------------------------------------------------------------------------
# v2: FPI-mixed preseason prior + success-rate second dimension
# ---------------------------------------------------------------------------

STATE_SCHEMA_V2 = 2


def state_has_v2_schema(sector_state: dict) -> bool:
    """True when the seed wrote the v2 fields (off/def_sr_adj + priors, fpi_prior)."""
    try:
        return int(sector_state.get("schema_version", 1)) >= STATE_SCHEMA_V2
    except (TypeError, ValueError):
        return False


def epa_prior_net(stats: dict, fpi_share: float, plays_per_team: float) -> float:
    """Effective preseason net-EPA/play prior for one team.

    The regressed prior-season EPA prior (off_epa_prior − def_epa_prior) is
    mixed with the team's PRESEASON FPI (``fpi_prior``: points vs the FBS
    mean, converted to EPA/play by dividing by plays per game) when the state
    carries one. Teams without an FPI row (or fpi_share == 0) fall back to the
    EPA-only prior — never imputed. 50/50 was the shipped share: pure FPI was
    equal-or-slightly-better on the three holdouts, the mix degrades
    gracefully when the FPI fetch fails.
    """
    net_prior = stats.get("off_epa_prior", 0.0) - stats.get("def_epa_prior", 0.0)
    fpi = stats.get("fpi_prior")
    if fpi is None or fpi_share <= 0.0 or plays_per_team <= 0:
        return net_prior
    return (1.0 - fpi_share) * net_prior + fpi_share * (float(fpi) / plays_per_team)


def blended_component(
    stats: dict,
    k: float,
    comp: str = "epa",
    fpi_share: float = 0.0,
    plays_per_team: float = 70.0,
) -> float:
    """Ramp-blended league-relative net rating for one component.

    comp="epa" → off_epa_adj − def_epa_adj, prior via :func:`epa_prior_net`
    (FPI-mixed). comp="sr" → off_sr_adj − def_sr_adj with the regressed
    off/def_sr_prior. Same ramp w(gp) = gp/(gp+k) as :func:`blended_net`.
    """
    gp = stats.get("gp", 0)
    w = prior_ramp_weight(gp, k)
    net_in = stats.get(f"off_{comp}_adj", 0.0) - stats.get(f"def_{comp}_adj", 0.0)
    if comp == "epa":
        net_prior = epa_prior_net(stats, fpi_share, plays_per_team)
    else:
        net_prior = stats.get(f"off_{comp}_prior", 0.0) - stats.get(f"def_{comp}_prior", 0.0)
    return w * net_in + (1.0 - w) * net_prior


def project_win_prob_v2(
    d_epa: float,
    d_sr: float,
    epa_pts: float,
    sr_pts: float,
    home_edge_pts: float,
    score_stdev: float,
    neutral: bool = False,
) -> tuple[float, float]:
    """v2 margin model: margin = epa_pts·Δnet_epa + sr_pts·Δnet_sr + home edge.

    d_epa / d_sr are home-minus-away component deltas (EPA/play and success
    rate). epa_pts / sr_pts are points of margin per unit delta — frozen
    constants fit by no-intercept OLS on 2024 and held out on 2023 + 2025
    (scripts/backtest_ncaaf_v2.py --fit). Returns (prob_home, margin), prob
    clipped to [0.02, 0.98] like v1.
    """
    edge = 0.0 if neutral else home_edge_pts
    margin = epa_pts * d_epa + sr_pts * d_sr + edge
    p = _normal_cdf(margin / score_stdev)
    return max(0.02, min(0.98, p)), margin


# ---------------------------------------------------------------------------
# Analytic win probability
# ---------------------------------------------------------------------------


def project_win_prob(
    net_home: float,
    net_away: float,
    plays_per_team: float,
    home_edge_pts: float,
    score_stdev: float,
    neutral: bool = False,
) -> tuple[float, float]:
    """Projected home margin → P(home win).

    margin = (net_home − net_away) · plays_per_team + home_edge (0 if neutral).
    net_* are EPA/play, so multiplying by plays/team converts the per-play edge
    into a point margin. Returns (prob_home, margin), prob clipped to [0.02, 0.98].
    """
    edge = 0.0 if neutral else home_edge_pts
    margin = (net_home - net_away) * plays_per_team + edge
    p = _normal_cdf(margin / score_stdev)
    return max(0.02, min(0.98, p)), margin
