"""Touchdown projections: team TD volume x player expected-TD (xTD) share.

Touchdowns are the noisiest stat a player has (yards-to-TD conversion barely
persists year to year), so the player side is built from WHERE his touches
happen, not from his past TDs:

1. Every carry and target is bucketed by field position (``yardline_100``).
   A bucket's league TD rate (point-in-time, from games before the cutoff)
   turns a player's touches into expected TDs (xTD).
2. A player's share of his team's rushing / receiving xTD is recency-weighted
   and shrunk toward his position's mean share (``player_model`` does the same
   for targets and carries).
3. Team rushing and receiving TDs per game are projected like any other team
   volume (``player_model.VOLUME_STATS``): opponent-adjusted ratings plus the
   game model's projected script.

Expected TDs = team TDs x xTD share; counts are Poisson, so
P(anytime TD) = 1 - exp(-(lambda_rush + lambda_rec)) and P(2+) follows. The
starting QB's passing TDs are the team's receiving TDs times his share of team
pass attempts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from evmax.nfl_projections import data

# Field-position buckets on yardline_100 (yards from the opponent's goal line).
RZ_BUCKETS: tuple[tuple[int, int], ...] = ((1, 2), (3, 5), (6, 10), (11, 20), (21, 99))
RZ_SCHEMA_VERSION = 1

_PBP_COLS = ["game_id", "play_type", "rusher_player_id", "receiver_player_id", "yardline_100",
             "rush_touchdown", "pass_touchdown", "two_point_attempt"]


def _bucket_cols(kind: str) -> list[str]:
    return [f"{kind}_b{i}" for i in range(len(RZ_BUCKETS))]


RUSH_OPP = _bucket_cols("rush")
RUSH_TD = _bucket_cols("rush_td")
TGT_OPP = _bucket_cols("tgt")
REC_TD = _bucket_cols("rec_td")
RZ_COLS = RUSH_OPP + RUSH_TD + TGT_OPP + REC_TD


def build_rz_usage(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, player_id): carries/targets and their TDs by field-position bucket.

    Two-point attempts are excluded (they are not touchdowns). QB scrambles
    count as carries (they are rushing attempts with rushing-TD credit).
    """
    p = pbp[pbp["play_type"].isin(["run", "pass"]) & (pbp["two_point_attempt"].fillna(0) == 0)
            & pbp["yardline_100"].notna()]
    b = np.full(len(p), -1)
    y = p["yardline_100"].to_numpy()
    for i, (lo, hi) in enumerate(RZ_BUCKETS):
        b[(y >= lo) & (y <= hi)] = i
    p = p.assign(bucket=b)
    p = p[p["bucket"] >= 0]
    frames = []
    for idcol, opp, td, tdcol in (("rusher_player_id", "rush", "rush_td", "rush_touchdown"),
                                  ("receiver_player_id", "tgt", "rec_td", "pass_touchdown")):
        q = p[p[idcol].notna()]
        g = q.groupby(["game_id", idcol, "bucket"]).agg(n=("bucket", "size"), td=(tdcol, "sum")).reset_index()
        g = g.rename(columns={idcol: "player_id"})
        n = g.pivot_table(index=["game_id", "player_id"], columns="bucket", values="n", fill_value=0)
        t = g.pivot_table(index=["game_id", "player_id"], columns="bucket", values="td", fill_value=0)
        n.columns = [f"{opp}_b{int(c)}" for c in n.columns]
        t.columns = [f"{td}_b{int(c)}" for c in t.columns]
        frames.append(n.join(t, how="outer"))
    out = frames[0].join(frames[1], how="outer").fillna(0.0)
    for c in RZ_COLS:
        if c not in out:
            out[c] = 0.0
    return out[RZ_COLS].astype(float).reset_index()


def rz_usage_file(d: Optional[Path] = None) -> Path:
    return data.data_dir(d) / f"rz_usage_v{RZ_SCHEMA_VERSION}.parquet"


def load_rz_usage(seasons: Iterable[int], d: Optional[Path] = None, rebuild: bool = False) -> pd.DataFrame:
    """Cached red-zone usage for ``seasons`` (rebuilt when a play-by-play file is newer)."""
    seasons = sorted(set(seasons))
    out = rz_usage_file(d)
    sources = [data.pbp_file(s, d) for s in seasons]
    newest = max((p.stat().st_mtime for p in sources if p.exists()), default=0)
    if not rebuild and out.exists() and out.stat().st_mtime >= newest:
        rz = pd.read_parquet(out)
        have = set(rz["game_id"].str[:4].astype(int).unique())
        if set(s for s in seasons if data.pbp_file(s, d).exists()) <= have:
            return rz[rz["game_id"].str[:4].astype(int).isin(seasons)].reset_index(drop=True)
    rz = build_rz_usage(data.load_pbp(seasons, d, columns=_PBP_COLS))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.part")
    rz.to_parquet(tmp, index=False)
    tmp.replace(out)  # atomic: a concurrent reader never sees a half-written file
    return rz


def bucket_td_rates(rz: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """League TD rate per carry and per target in each bucket (pooled over ``rz``)."""
    r_rush = rz[RUSH_TD].sum().to_numpy() / np.maximum(rz[RUSH_OPP].sum().to_numpy(), 1.0)
    r_tgt = rz[REC_TD].sum().to_numpy() / np.maximum(rz[TGT_OPP].sum().to_numpy(), 1.0)
    return r_rush, r_tgt


def expected_tds(rz: pd.DataFrame, r_rush: np.ndarray, r_tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rushing xTD, receiving xTD) per row: touches by bucket x the bucket's league TD rate."""
    return rz[RUSH_OPP].to_numpy() @ r_rush, rz[TGT_OPP].to_numpy() @ r_tgt


def poisson_at_least(lam: np.ndarray, k: int) -> np.ndarray:
    """P(N >= k) for N ~ Poisson(lam)."""
    lam = np.asarray(lam, dtype=float)
    term = np.exp(-lam)
    below = np.zeros_like(lam)
    for i in range(k):
        below += term
        term = term * lam / (i + 1)
    return 1.0 - below
