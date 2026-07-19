"""PitcherModelAgent (v2) — pitcher-matchup win probability for MLB games.

Uses the Pythagorean expectation formula (Pythagenpat, exponent=1.83):
    W% = RS^e / (RS^e + RA^e)

Each team scores at the rate the *opposing* pitching allows (scaled by its
own lineup quality) and allows runs at its own pitching's rate. The pitching
rate is a per-starter quality estimate blended with the team's available
bullpen (evmax/models_ml/bullpen.py — the starter covers ~60% of innings, a
signal v1 ignored entirely), and the starter's quality estimate itself
blends xERA / FIP / ERA when Savant expected-stats are seeded (see
_effective_era). Park effects are removed from the results-based components
per pitcher (see _PARK_RUN_FACTOR); xERA is park-neutral by construction.

Live probable starters come from the MLB Stats API (ESPN scoreboard
fallback); reliever fatigue from the same API's boxscores
(fetch_reliever_appearances); reliever season stats + xERA are seeded by
scripts/seed_pitcher_fip.py. Every v2 component degrades independently to
the v1 behavior when its data is absent — v2 never reduces coverage.

Only activates for sector == "baseball". Returns None for all other sectors.
"""

from __future__ import annotations

import math
import time
from datetime import date as _date
from typing import Optional

import httpx
import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds
from evmax.models_ml.bullpen import team_pen_quality, team_rate_with_pen

logger = structlog.get_logger(__name__)

PYTHAG_EXP = 1.83
HOME_BONUS = 0.04  # Default ~54% baseline home win rate in MLB. Used when
                   # state["home_wp_running"] is absent. See _home_advantage().
HOME_BONUS_MIN = 0.01  # Floor — never go fully neutral or anti-home; even in
                       # low-HFA seasons, home teams sleep at home.
HOME_BONUS_MAX = 0.07  # Ceiling — caps over-aggressive correction in years
                       # with unusually high home WP, which often regress.
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

# When both FIP and ERA are seeded, blend them. FIP is more predictive of
# future run-prevention because it strips defensive luck and sequencing,
# but ERA still carries real signal (a pitcher who consistently outperforms
# FIP via deception, weak contact, or pitching-from-the-stretch may have a
# repeatable skill ERA captures and FIP doesn't). 60/40 in FIP's favor is
# the textbook compromise — heavier than 50/50 because FIP wins on
# year-over-year correlation, but not pure FIP because we don't want to
# discard ERA's real-world result entirely.
FIP_BLEND_WEIGHT = 0.60
ERA_BLEND_WEIGHT = 0.40

# v2: when Savant xERA is also seeded, it leads the blend — quality-of-contact
# expected ERA out-predicts both FIP (which only strips defense) and ERA on
# forward run prevention. Weights swept on the 2025 season by
# scripts/backtest_baseball_ab.py (2026 held out); these are the shipped
# values. Renormalized over whichever components are present.
XERA_BLEND_WEIGHT = 0.45
FIP_BLEND_WEIGHT_V2 = 0.35
ERA_BLEND_WEIGHT_V2 = 0.20
# Below this current-season balls-in-play sample, current xERA is noise —
# substitute the prior season's xERA at half weight instead.
XERA_THIN_BIP = 40

# Per-pitcher park normalization (v2). A symmetric venue multiplier applied
# to both teams' RS/RA cancels EXACTLY in Pythagenpat — (k·RS)^e /
# ((k·RS)^e + (k·RA)^e) is k-invariant — so "adjust for today's park" is a
# moneyline no-op. What does NOT cancel is the park baked into each
# pitcher's SAMPLE: a Rockies starter's results-based rates (FIP/ERA) are
# inflated by ~half his innings coming at Coors. Dividing by
# sqrt(park_run_factor) removes that. Applied to FIP/ERA only — xERA is
# park-neutral by construction. Values are implementation-time
# approximations of Savant's 3-yr rolling RUN park factors (1.00 = neutral),
# keyed by MLB team id; unlisted parks are ~neutral.
_PARK_RUN_FACTOR: dict[int, float] = {
    115: 1.24,  # Rockies — Coors
    113: 1.10,  # Reds — Great American
    111: 1.06,  # Red Sox — Fenway
    109: 1.05,  # D-backs — Chase
    118: 1.04,  # Royals — Kauffman
    141: 1.03,  # Blue Jays — Rogers Centre
    121: 0.96,  # Mets — Citi Field
    139: 0.96,  # Rays — Tropicana
    146: 0.95,  # Marlins — loanDepot
    135: 0.94,  # Padres — Petco
    137: 0.93,  # Giants — Oracle
    136: 0.92,  # Mariners — T-Mobile
}

# Team-offense scaling (v2): each side scores at the opposing pitching's
# allowed rate × its own lineup quality (season runs/game over league
# average). Clamped — season-to-date team offense is noisy and the clamp
# keeps a hot April lineup from swinging the matchup more than pitching does.
OFFENSE_CLAMP_LO = 0.85
OFFENSE_CLAMP_HI = 1.15


def _park_factor_for(pitcher: dict) -> float:
    """Run park factor for a pitcher's HOME park (via seeded team nickname)."""
    team = (pitcher.get("team") or "").lower().strip()
    if not team:
        return 1.0
    from evmax.clients.mlb_statsapi import MLB_TEAM_NICKNAME

    for tid, nickname in MLB_TEAM_NICKNAME.items():
        if nickname == team:
            return _PARK_RUN_FACTOR.get(tid, 1.0)
    return 1.0


def _effective_era(
    pitcher: dict, league_avg: float, park_factor: float = 1.0
) -> float:
    """Compute the run-allowed rate to feed into Pythag.

    Preference order:
      1. xERA + FIP + ERA present → weighted blend (XERA 0.45 / FIP 0.35 /
         ERA 0.20, renormalized over present components). Thin current-season
         xERA sample (bip < XERA_THIN_BIP) substitutes the prior season's
         xERA at HALF weight.
      2. FIP + ERA → 60/40 v1 blend
      3. FIP only → FIP;  ERA only → ERA
      4. Neither → league average

    ``park_factor`` (>1 = hitter's park) removes the pitcher's home-park
    effect from the RESULTS-based components: FIP/ERA are divided by
    sqrt(factor) (~half his innings are at home). xERA is already
    park-neutral and is never adjusted.

    A non-positive era/fip is treated as MISSING, not as a real rate. The MLB
    Stats API seeds ``era: 0.0`` as a placeholder for "look this pitcher up in
    the seeded DB"; when the starter isn't in our seed, that 0.0 leaks through.
    Feeding 0.0 into the Pythag formula makes the denominator
    ``0**e + 0**e == 0`` for a both-unseeded matchup → ZeroDivisionError, which
    (with no per-pair guard upstream) used to kill the entire pitcher batch and,
    via the pitcher-required guard, drop every baseball moneyline. Falling back
    to league average keeps the math finite. ``predict_pair`` separately
    abstains (returns None) when a starter has no real rate, so an info-less
    matchup never masquerades as a pitcher contributor.
    """
    def _pos(v) -> Optional[float]:
        return float(v) if v is not None and float(v) > 0 else None

    park_div = math.sqrt(park_factor) if park_factor and park_factor > 0 else 1.0
    fip = _pos(pitcher.get("fip"))
    era = _pos(pitcher.get("era"))
    if fip is not None:
        fip /= park_div
    if era is not None:
        era /= park_div

    xera = _pos(pitcher.get("xera"))
    xera_w = XERA_BLEND_WEIGHT
    bip = pitcher.get("xera_bip") or 0
    if xera is not None and bip < XERA_THIN_BIP:
        prior = _pos(pitcher.get("xera_prior"))
        xera, xera_w = (prior, XERA_BLEND_WEIGHT * 0.5) if prior is not None else (None, 0.0)
    elif xera is None:
        prior = _pos(pitcher.get("xera_prior"))
        if prior is not None and bip < XERA_THIN_BIP:
            xera, xera_w = prior, XERA_BLEND_WEIGHT * 0.5

    if xera is not None and (fip is not None or era is not None):
        parts = [(xera, xera_w)]
        if fip is not None:
            parts.append((fip, FIP_BLEND_WEIGHT_V2))
        if era is not None:
            parts.append((era, ERA_BLEND_WEIGHT_V2))
        total_w = sum(w for _, w in parts)
        return sum(v * w for v, w in parts) / total_w

    if fip is not None and era is not None:
        return FIP_BLEND_WEIGHT * fip + ERA_BLEND_WEIGHT * era
    if fip is not None:
        return fip
    if era is not None:
        return era
    if xera is not None:
        return xera
    return league_avg


def _has_real_rate(pitcher: dict) -> bool:
    """True if the pitcher carries a usable ERA or FIP (positive value).

    A starter resolved from the live MLB feed but absent from the seeded DB
    arrives with era=0.0 — that's a placeholder, not a real rate, so it returns
    False and the matchup is treated as having no pitcher signal.
    """
    for key in ("fip", "era"):
        val = pitcher.get(key)
        if val is not None and float(val) > 0:
            return True
    return False


# Cache ESPN probable starters for 30 minutes
_probable_cache: dict[str, dict] = {}
_probable_cache_ts: float = 0.0
_CACHE_TTL = 1800


async def _fetch_probable_starters() -> dict[str, dict]:
    """Today's probable starters, MLB Stats API primary, ESPN fallback.

    The official MLB Stats API keys teams by stable integer ID, so it never
    suffers the nickname mismatch that dropped Red Sox / White Sox / Blue Jays
    games from the pitcher blend. ESPN remains a fallback for the rare case the
    Stats API is unreachable or hasn't populated probables yet.
    """
    from evmax.clients.mlb_statsapi import fetch_probable_starters as _mlb_fetch

    mlb = await _mlb_fetch()
    if mlb:
        return mlb
    logger.info("probable_starters_fallback_espn")
    return await _fetch_probable_starters_espn()


async def _fetch_probable_starters_espn() -> dict[str, dict]:
    """Fetch today's probable starters from ESPN (fallback source).

    Returns: {team_short_name: {"name": str, "era": float, "ip": float}}
    """
    global _probable_cache, _probable_cache_ts
    if _probable_cache and (time.time() - _probable_cache_ts) < _CACHE_TTL:
        return _probable_cache

    result: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(ESPN_SCOREBOARD, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()

        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for team in comp.get("competitors", []):
                    team_name = team.get("team", {}).get("shortDisplayName", "").lower().strip()
                    if not team_name:
                        team_name = team.get("team", {}).get("displayName", "").lower().strip()
                        # Extract last word: "New York Yankees" → "yankees"
                        if team_name:
                            team_name = team_name.split()[-1]

                    probables = team.get("probables", [])
                    for prob in probables:
                        athlete = prob.get("athlete", {})
                        pitcher_name = athlete.get("fullName", "").lower().strip()
                        # Parse ERA from statistics or record
                        era = 4.08  # default
                        for stat in prob.get("statistics", []):
                            if stat.get("abbreviation") == "ERA":
                                try:
                                    era = float(stat["displayValue"])
                                except (ValueError, KeyError):
                                    pass
                        if team_name and pitcher_name:
                            result[team_name] = {
                                "name": pitcher_name,
                                "era": era,
                                "ip": 0,  # not available from scoreboard
                            }

        _probable_cache = result
        _probable_cache_ts = time.time()
        logger.info("probable_starters_fetched", teams=len(result))
    except Exception as e:
        logger.warning("probable_starters_fetch_failed", error=str(e))

    return result


class PitcherModelAgent(ModelAgent):
    # "pitcher_v2" (2026-07-19): the rename is the shadow-reset mechanism —
    # the contamination filter dates rows by model_sources signature, so
    # every v1 row drops out of the clean promotion sample the moment v2
    # ships. The ev_gap_agent pitcher-required guard checks the substring
    # "pitcher", which matches both names.
    name = "pitcher_v2"
    weight = 0.50

    def _league_avg_era(self) -> float:
        return self._state.get("league_avg_era", 4.08)

    def _pitchers(self) -> dict[str, dict]:
        return self._state.get("pitchers", {})

    def _relievers(self) -> dict[str, dict]:
        return self._state.get("relievers", {})

    @staticmethod
    def _team_nickname(team_label: str) -> Optional[str]:
        """Resolve a full team label to the seeded nickname ('boston red sox'
        → 'red sox'). Longest-suffix match over the canonical nickname set —
        the same rule _match_live_starter uses, so multi-word nicknames
        (Red Sox / White Sox / Blue Jays) resolve unambiguously."""
        from evmax.clients.mlb_statsapi import MLB_TEAM_NICKNAME

        team = (team_label or "").lower().strip()
        best = None
        for nickname in MLB_TEAM_NICKNAME.values():
            if team == nickname or team.endswith(nickname):
                if best is None or len(nickname) > len(best):
                    best = nickname
        return best

    def _pen_quality(
        self,
        team_label: str,
        live_appearances: dict[str, list],
        today: _date,
    ) -> Optional[float]:
        """Available-bullpen quality (FIP) for a team, or None (degrade to
        starter-only). Reliever season stats come from the seed; the fatigue
        log comes from the live MLB Stats API feed (empty log = every
        reliever counts as rested — static pen quality, still better than
        ignoring 40% of innings)."""
        relievers = self._relievers()
        if not relievers:
            return None
        nickname = self._team_nickname(team_label)
        if nickname is None:
            return None
        reliever_team = {name: rec.get("team", "") for name, rec in relievers.items()}
        appearances: dict[str, list] = {}
        for name in relievers:
            log = live_appearances.get(name)
            if log:
                appearances[name] = [
                    (_date.fromisoformat(d), ip, pc) for d, ip, pc in log
                ]
        return team_pen_quality(
            nickname,
            today,
            relievers,
            appearances,
            reliever_team,
            self._state.get("league_pen_cfip", 3.10),
        )

    @staticmethod
    def _offense_factor(team_label: str) -> float:
        """Lineup-quality multiplier: season runs/game over league average,
        clamped. 1.0 (v1 behavior) when the props cache has no offense block
        or the team can't be resolved."""
        try:
            from evmax.clients.baseball_props_cache import (
                get_team_offense,
                league_runs_per_game,
            )
            from evmax.clients.mlb_statsapi import MLB_TEAM_NICKNAME

            nickname = PitcherModelAgent._team_nickname(team_label)
            if nickname is None:
                return 1.0
            team_id = next(
                (tid for tid, nick in MLB_TEAM_NICKNAME.items() if nick == nickname),
                None,
            )
            if team_id is None:
                return 1.0
            offense = get_team_offense(team_id)
            league = league_runs_per_game()
            if not offense or not league or league <= 0:
                return 1.0
            rpg = offense.get("runs_per_game") or 0.0
            if rpg <= 0:
                return 1.0
            return max(OFFENSE_CLAMP_LO, min(OFFENSE_CLAMP_HI, rpg / league))
        except Exception:  # cache trouble is never a reason to drop the model
            return 1.0

    def _home_advantage(self) -> float:
        """Resolve the home-side probability bonus, preferring an adaptive
        running estimate when one is recorded in state.

        State shape (optional):
            {"home_wp_running": {"games": int, "home_wins": int}, ...}

        If at least 200 games of running data are available we compute
        HFA = home_wp - 0.50, clamped to [HOME_BONUS_MIN, HOME_BONUS_MAX].
        Below 200 games (very early-season), we fall back to the static
        HOME_BONUS to avoid noisy small-sample HFA. This number can be
        refreshed by `scripts/update_home_advantage.py` from resolved
        outcomes in predictions.db.
        """
        running = self._state.get("home_wp_running") or {}
        games = running.get("games", 0)
        home_wins = running.get("home_wins", 0)
        if games < 200:
            return HOME_BONUS
        wp = home_wins / games
        return max(HOME_BONUS_MIN, min(HOME_BONUS_MAX, wp - 0.50))

    @staticmethod
    def _match_live_starter(
        team: str, team_short: str, live_starters: dict[str, dict]
    ) -> Optional[dict]:
        """Resolve an ESPN probable-starter entry for a team label.

        Pinnacle passes full names ("boston red sox") while ESPN keys by
        nickname ("red sox"). The last-word fallback ("sox") fails for the
        multi-word nicknames — Red Sox, White Sox, Blue Jays — so any game
        involving them silently lost the pitcher model (predict_pair returns
        None if either starter is unresolved). Match the ESPN nickname key as
        a suffix of the full label to recover them. Suffix (not substring)
        avoids the "sox" ambiguity between Red Sox and White Sox.
        """
        # Exact / last-word first (cheapest, unambiguous).
        for key in (team, team_short):
            if key in live_starters:
                return live_starters[key]
        # Nickname-suffix: "boston red sox".endswith("red sox"). Prefer the
        # longest matching key so "red sox" wins over a hypothetical "sox".
        best_key = None
        for espn_key in live_starters:
            if team.endswith(espn_key) and (best_key is None or len(espn_key) > len(best_key)):
                best_key = espn_key
        return live_starters[best_key] if best_key else None

    def _find_starter(self, team: str, live_starters: dict[str, dict] | None = None) -> tuple[Optional[dict], bool]:
        """Find the probable starter for a team.

        Returns (pitcher_data, is_live) — is_live=True if from ESPN probables.
        """
        team = team.lower().strip()
        # Last word fallback: "new york yankees" → "yankees"
        team_short = team.split()[-1] if team else team

        # 1) Live ESPN probable starters (today's actual starter)
        if live_starters:
            live = self._match_live_starter(team, team_short, live_starters)
            if live is not None:
                # Check if we have this pitcher in our ERA database
                pitcher_name = live["name"]
                stored = self._pitchers().get(pitcher_name)
                if stored:
                    return stored, True
                # Use ESPN's ERA directly
                return live, True

        # 2) Static team_starters (fallback)
        team_starters = self._state.get("team_starters", {})
        for key in (team, team_short):
            pitcher_name = team_starters.get(key)
            if pitcher_name:
                data = self._pitchers().get(pitcher_name)
                if data:
                    return data, False

        # Substring match
        for stored_team, pitcher_name in team_starters.items():
            if stored_team in team or team in stored_team:
                data = self._pitchers().get(pitcher_name)
                if data:
                    return data, False

        return None, False

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "baseball":
            return None

        home = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        away = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        # Fetch live probable starters from ESPN
        live_starters = await _fetch_probable_starters()

        home_pitcher, home_live = self._find_starter(home, live_starters)
        away_pitcher, away_live = self._find_starter(away, live_starters)

        if not home_pitcher or not away_pitcher:
            return None

        # Abstain when either starter lacks a real rate. Live probable starters
        # from the MLB Stats API arrive with era=0.0 until matched to our seeded
        # ERA/FIP DB; an unseeded starter therefore carries no signal. Returning
        # None here (rather than predicting off a league-average placeholder)
        # keeps "pitcher" out of model_sources, so the pitcher-required guard
        # correctly drops the moneyline instead of betting blind.
        if not _has_real_rate(home_pitcher) or not _has_real_rate(away_pitcher):
            return None

        league_avg = self._league_avg_era()
        home_rate = _effective_era(
            home_pitcher, league_avg, park_factor=_park_factor_for(home_pitcher)
        )
        away_rate = _effective_era(
            away_pitcher, league_avg, park_factor=_park_factor_for(away_pitcher)
        )

        # v2: blend each starter's rate with the team's AVAILABLE bullpen
        # (60/40 — the starter covers ~5.5 of 9 innings). Reliever fatigue
        # from the live MLB feed; degrades to starter-only when the pen is
        # unseeded/thin. The fatigue fetch only fires when relievers are
        # seeded — an unseeded state must not touch the network (and unit
        # tests exercising Pythag math stay hermetic).
        from evmax.clients.mlb_statsapi import fetch_reliever_appearances

        live_appearances = (
            await fetch_reliever_appearances() if self._relievers() else {}
        )
        today = _date.today()
        pen_home = self._pen_quality(home, live_appearances, today)
        pen_away = self._pen_quality(away, live_appearances, today)
        home_rate_pen = team_rate_with_pen(home_rate, pen_home)
        away_rate_pen = team_rate_with_pen(away_rate, pen_away)

        # v2: lineup quality — each side scores at the opposing pitching's
        # allowed rate scaled by its own offense (clamped; 1.0 when no data).
        off_home = self._offense_factor(home)
        off_away = self._offense_factor(away)

        # Pythagorean matchup: each team scores at the rate the opposing
        # pitching (starter+pen) gives up runs × its own lineup quality, and
        # allows at its own pitching's rate × the opposing lineup quality —
        # so home_ra == away_rs and the probabilities stay complementary.
        e = PYTHAG_EXP
        home_rs = away_rate_pen * off_home
        home_ra = home_rate_pen * off_away
        away_rs = home_rate_pen * off_away
        away_ra = away_rate_pen * off_home

        # Defensive guard: with _effective_era's non-positive fallback the
        # denominators are always positive, but never let a degenerate rate
        # pair raise — skip the matchup instead of crashing the batch.
        home_denom = home_rs ** e + home_ra ** e
        away_denom = away_rs ** e + away_ra ** e
        if home_denom <= 0 or away_denom <= 0:
            return None
        home_wp = (home_rs ** e) / home_denom
        away_wp = (away_rs ** e) / away_denom

        # home_wp and away_wp are already complements by construction
        # (home_wp = away_era^e / (away_era^e + home_era^e) = 1 - away_wp),
        # but normalize defensively in case ERAs are degenerate.
        total = home_wp + away_wp
        prob_a = home_wp / total if total > 0 else 0.5

        # Apply home field advantage — adaptive when state carries a running
        # estimate, else falls back to the static HOME_BONUS default.
        prob_a = min(0.90, max(0.10, prob_a + self._home_advantage()))
        prob_b = 1.0 - prob_a

        # Confidence tiers — designed so the model fires (>= 0.45 gate) on
        # any pitcher with FIP data, even early-season when IP totals are low.
        # FIP itself is the data-quality signal; IP is a sample-size proxy
        # that mattered more when seeds were prior-season totals (~150+ IP).
        # With current-season FIP seeded, 30 IP of recent data is more
        # informative than 200 IP of last-year data, so the FIP path gets
        # a higher floor.
        home_ip = home_pitcher.get("ip", 0)
        away_ip = away_pitcher.get("ip", 0)
        min_ip = min(home_ip, away_ip)
        both_live = home_live and away_live
        both_fip = (
            home_pitcher.get("fip") is not None
            and away_pitcher.get("fip") is not None
        )

        if both_live and min_ip >= 150:
            confidence = 0.80  # Live + deep history
        elif both_live and min_ip >= 100:
            confidence = 0.70
        elif both_fip and min_ip >= 100:
            confidence = 0.75  # FIP-armed + good sample
        elif both_fip and min_ip >= 30:
            confidence = 0.60  # FIP-armed early-season
        elif both_fip and min_ip >= 15:
            confidence = 0.50  # FIP-armed thin sample — barely above 0.45 gate.
                               # Justified empirically: backtest shows pitcher
                               # is the best single MLB model where it fires,
                               # and a pitcher with FIP from 2-3 starts is more
                               # informative than no signal at all.
        elif both_live:
            confidence = 0.55  # Live but ESPN-only (no IP, no FIP)
        elif min_ip >= 150:
            confidence = 0.65  # ERA-only deep history
        elif min_ip >= 100:
            confidence = 0.55  # ERA-only moderate
        else:
            confidence = 0.35  # ERA-only thin — below gate, won't contribute

        pitcher_notes = []
        if home_live:
            pitcher_notes.append(f"home={home_pitcher.get('name', '?')}(live)")
        if away_live:
            pitcher_notes.append(f"away={away_pitcher.get('name', '?')}(live)")

        def _rate_label(p: dict) -> str:
            if p.get("fip") is not None and p.get("era") is not None:
                return f"fip={p['fip']:.2f}/era={p['era']:.2f}"
            if p.get("fip") is not None:
                return f"fip={p['fip']:.2f}"
            return f"era={p.get('era', league_avg):.2f}"

        v2_notes = []
        if pen_home is not None or pen_away is not None:
            v2_notes.append(
                f"pen={pen_home if pen_home is None else round(pen_home, 2)}"
                f"/{pen_away if pen_away is None else round(pen_away, 2)}"
            )
        if off_home != 1.0 or off_away != 1.0:
            v2_notes.append(f"off={off_home:.2f}/{off_away:.2f}")
        if "xera" in home_pitcher or "xera" in away_pitcher:
            v2_notes.append("xera")

        notes = (
            f"home[{_rate_label(home_pitcher)}] "
            f"away[{_rate_label(away_pitcher)}] "
            f"effective_home={home_rate_pen:.2f} effective_away={away_rate_pen:.2f} "
            f"{' '.join(v2_notes)} "
            f"{' '.join(pitcher_notes)}"
        ).strip()

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=round(prob_a, 4),
            true_prob_b=round(prob_b, 4),
            confidence=confidence,
            weight=self.weight,
            sample_size=int(min_ip),
            notes=notes,
        )

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        # Pitcher data is seeded externally, not updated from game results
        pass

    def seed_pitchers(self, pitchers: dict[str, dict], league_avg_era: float = 4.08) -> None:
        """Bulk-seed pitcher data.

        Args:
            pitchers: {name: {"era": float, "fip": float (optional), "ip": float, "team": str}}
                ERA is required as the baseline rate. FIP is optional but
                strongly preferred — when present, the agent blends 60% FIP
                + 40% ERA, which is more predictive of future run prevention.
            league_avg_era: league average ERA for normalizing fallback rates
        """
        self._state["league_avg_era"] = league_avg_era
        store = self._state.setdefault("pitchers", {})
        team_starters: dict[str, str] = self._state.setdefault("team_starters", {})

        for name, data in pitchers.items():
            key = name.lower().strip()
            store[key] = data
            team = data.get("team", "").lower()
            if team:
                # First pitcher encountered per team becomes the "starter"
                # (seed script should list #1 starter first)
                if team not in team_starters:
                    team_starters[team] = key

        self.save_state()
        self.log.info("pitchers_seeded", count=len(pitchers), league_avg=league_avg_era)
