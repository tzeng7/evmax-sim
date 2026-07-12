"""Walk-forward backtest of the UFC Glicko-2 + feature-layer model.

Every prediction is made from PRE-FIGHT-ONLY state (chronological replay via
evmax.agents.models.ufc_rating_agent.replay_history — the exact code the live
agent and seed script use). The feature layer is fitted on fights through
TRAIN_END and evaluated on the untouched holdout after it.

Compares:
  (a) rating-only baseline — Glicko-2 win probability
  (b) rating + feature-layer blend (logit-space logistic)

Metrics: Brier, accuracy, calibration buckets (train and holdout separately).

GATE (see the mission notes in data/categories.yaml `ufc`): the blend must
beat rating-only on HOLDOUT Brier with sane calibration (no bucket off by
>10pp at n>=50). If the blend fails, ship rating-only IF it is itself
calibrated; if both fail, the sector goes sharp-only.

Outputs:
  - console report
  - data/backtest/ufc/feature_weights.json  (fitted coefficients, for seed)
  - docs/ufc-model-eval.md                  (verdict document)

Usage:
    python scripts/backtest_ufc_model.py [--train-end 2024-12-31]
                                         [--eval-start 2015-01-01] [--min-n 1]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models.ufc_rating_agent import FEATURE_NAMES, replay_history

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "backtest" / "ufc"
DOCS_PATH = Path(__file__).resolve().parents[1] / "docs" / "ufc-model-eval.md"
WEIGHTS_PATH = DATA_DIR / "feature_weights.json"

L2 = 1e-3  # ridge penalty on feature coefs (not on intercept / rating coef)


def load_dataset() -> tuple[list[dict], dict[str, dict]]:
    fights_path = DATA_DIR / "fights.csv"
    fighters_path = DATA_DIR / "fighters.csv"
    if not fights_path.exists() or not fighters_path.exists():
        sys.exit(f"missing {fights_path} — run scripts/fetch_ufc_history.py first")
    with fights_path.open() as fh:
        fights = list(csv.DictReader(fh))
    with fighters_path.open() as fh:
        bios = {r["athlete_id"]: r for r in csv.DictReader(fh)}
    fights.sort(key=lambda f: (f["date"], f["event_id"], f["comp_id"]))
    return fights, bios


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit [intercept, rating_coef, feature coefs...] by penalized log-loss."""

    def loss(w: np.ndarray) -> float:
        z = X @ w
        # numerically-stable log-loss
        ll = np.mean(np.logaddexp(0.0, z) - y * z)
        return ll + L2 * np.sum(w[2:] ** 2)

    w0 = np.zeros(X.shape[1])
    w0[1] = 1.0  # start at "trust the rating fully"
    res = minimize(loss, w0, method="L-BFGS-B")
    return res.x


def design_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.column_stack(
        [np.ones(len(rows)), logit(np.array([r["p_rating"] for r in rows]))]
        + [np.array([r[f"x_{name}"] for r in rows]) for name in FEATURE_NAMES]
    )
    y = np.array([r["outcome_a"] for r in rows])
    return X, y


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y == 1.0)))


def calibration(p: np.ndarray, y: np.ndarray, n_buckets: int = 10) -> list[dict]:
    out = []
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "pred": float(p[mask].mean()),
            "actual": float(y[mask].mean()),
        })
    return out


def calibration_ok(buckets: list[dict], max_gap: float = 0.10, min_n: int = 50) -> bool:
    return all(
        abs(b["pred"] - b["actual"]) <= max_gap for b in buckets if b["n"] >= min_n
    )


def fmt_cal(buckets: list[dict]) -> str:
    lines = ["  bucket     n     pred   actual   gap"]
    for b in buckets:
        gap = b["pred"] - b["actual"]
        flag = " ⚑" if b["n"] >= 50 and abs(gap) > 0.10 else ""
        lines.append(
            f"  {b['bucket']}  {b['n']:>5}  {b['pred']:.3f}  {b['actual']:.3f}  {gap:+.3f}{flag}"
        )
    return "\n".join(lines)


def main(train_end: str, eval_start: str, min_n: int) -> None:
    fights, bios = load_dataset()
    print(f"loaded {len(fights)} fights, {len(bios)} fighter bios")

    # Pass 1: rating-only replay collecting rows + features (point-in-time).
    _, rows = replay_history(fights, bios, collect=True)
    rows = [r for r in rows if r["outcome_a"] in (0.0, 1.0)]  # drop draws from scoring
    eligible = [
        r for r in rows
        if r["date"] >= eval_start and min(r["n_a"], r["n_b"]) >= min_n
    ]
    train = [r for r in eligible if r["date"] <= train_end]
    hold = [r for r in eligible if r["date"] > train_end]
    print(f"eligible (min_n≥{min_n}, ≥{eval_start}): {len(eligible)} "
          f"| train ≤{train_end}: {len(train)} | holdout: {len(hold)}")
    if len(train) < 500 or len(hold) < 100:
        sys.exit("not enough data for a meaningful fit/holdout")

    # Fit the feature layer on train only — SYMMETRIZED: ESPN's side-a
    # (order=1, red corner) wins ~56% of bouts, an orientation prior that
    # would NOT transfer to live prediction (side a there is Pinnacle's
    # home label, a different convention). Every training row is mirrored
    # (features negated, rating logit flipped, outcome inverted) so the
    # intercept fits ≈0 and no coefficient can learn the corner shortcut.
    def _mirror(r: dict) -> dict:
        m = dict(r)
        m["p_rating"] = 1.0 - r["p_rating"]
        m["outcome_a"] = 1.0 - r["outcome_a"]
        for name in FEATURE_NAMES:
            m[f"x_{name}"] = -r[f"x_{name}"]
        return m

    train_sym = train + [_mirror(r) for r in train]
    Xtr, ytr = design_matrix(train_sym)
    w = fit_logistic(Xtr, ytr)
    names = ["intercept", "rating_coef"] + FEATURE_NAMES
    print("\nFitted feature layer (train ≤ %s):" % train_end)
    for n, c in zip(names, w):
        print(f"  {n:>12}: {c:+.4f}")

    def blend_probs(rows_: list[dict]) -> np.ndarray:
        X, _ = design_matrix(rows_)
        z = X @ w
        return 1.0 / (1.0 + np.exp(-z))

    report: dict = {"train_end": train_end, "eval_start": eval_start, "min_n": min_n}
    for split_name, split in (("train", train), ("holdout", hold)):
        _, y = design_matrix(split)
        p_rating = np.array([r["p_rating"] for r in split])
        p_blend = blend_probs(split)
        report[split_name] = {
            "n": len(split),
            "rating_brier": brier(p_rating, y),
            "blend_brier": brier(p_blend, y),
            "rating_acc": accuracy(p_rating, y),
            "blend_acc": accuracy(p_blend, y),
            "rating_cal": calibration(p_rating, y),
            "blend_cal": calibration(p_blend, y),
        }
        r = report[split_name]
        print(f"\n== {split_name.upper()} (n={r['n']}) ==")
        print(f"  rating-only: Brier {r['rating_brier']:.4f}  acc {r['rating_acc']:.3f}")
        print(f"  blend:       Brier {r['blend_brier']:.4f}  acc {r['blend_acc']:.3f}")
        print("  blend calibration:")
        print(fmt_cal(r["blend_cal"]))

    # ---- GATE ----
    h = report["holdout"]
    blend_wins = h["blend_brier"] < h["rating_brier"]
    blend_calibrated = calibration_ok(h["blend_cal"])
    rating_calibrated = calibration_ok(h["rating_cal"])
    if blend_wins and blend_calibrated:
        verdict = "VALIDATED_BLEND"
    elif rating_calibrated:
        verdict = "VALIDATED_RATING_ONLY"
    else:
        verdict = "REJECTED_SHARP_ONLY"
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    print(f"  holdout ΔBrier (blend − rating): "
          f"{h['blend_brier'] - h['rating_brier']:+.5f}")

    # ---- outputs ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    feature_model = {
        "intercept": float(w[0]),
        "rating_coef": float(w[1]),
        "coefs": {n: float(c) for n, c in zip(FEATURE_NAMES, w[2:])},
        "fitted_through": train_end,
        "verdict": verdict,
    }
    WEIGHTS_PATH.write_text(json.dumps(feature_model, indent=2))
    print(f"wrote {WEIGHTS_PATH}")

    _write_eval_doc(report, feature_model)
    print(f"wrote {DOCS_PATH}")


def _write_eval_doc(report: dict, fm: dict) -> None:
    h, t = report["holdout"], report["train"]
    lines = [
        "# UFC model evaluation — Glicko-2 + feature layer",
        "",
        f"Generated by `scripts/backtest_ufc_model.py` on {date.today().isoformat()}.",
        "",
        "## Setup",
        "",
        "- Data: ESPN MMA public API (results, methods, bios), 2010 → present"
        " — see `evmax/clients/ufc_espn.py` for why not ufcstats.com"
        " (JS anti-bot interstitial; we don't bypass bot detection).",
        f"- Walk-forward replay, ratings warm up from 2010; evaluation from"
        f" {report['eval_start']} with both fighters having ≥{report['min_n']}"
        " prior UFC bout(s).",
        f"- Feature layer fitted on fights ≤ {report['train_end']}"
        f" (n={t['n']}); holdout is everything after (n={h['n']}).",
        "- Features (point-in-time differentials): age, layoff (log-days),"
        " recent-KO-loss flag, experience, shrunk win rate, last-3 streak,"
        " reach, height, finish rate, KO-absorbed rate."
        " Per-minute volume stats (SLpM/SApM/TD) are unavailable without"
        " ufcstats.com; documented limitation.",
        "",
        "## Results",
        "",
        "| split | n | rating Brier | blend Brier | rating acc | blend acc |",
        "|---|---|---|---|---|---|",
        f"| train | {t['n']} | {t['rating_brier']:.4f} | {t['blend_brier']:.4f}"
        f" | {t['rating_acc']:.3f} | {t['blend_acc']:.3f} |",
        f"| holdout | {h['n']} | {h['rating_brier']:.4f} | {h['blend_brier']:.4f}"
        f" | {h['rating_acc']:.3f} | {h['blend_acc']:.3f} |",
        "",
        "## Holdout calibration (blend)",
        "",
        "| bucket | n | pred | actual | gap |",
        "|---|---|---|---|---|",
    ]
    for b in h["blend_cal"]:
        gap = b["pred"] - b["actual"]
        flag = " ⚑" if b["n"] >= 50 and abs(gap) > 0.10 else ""
        lines.append(
            f"| {b['bucket']} | {b['n']} | {b['pred']:.3f} | {b['actual']:.3f} | {gap:+.3f}{flag} |"
        )
    lines += [
        "",
        "## Fitted feature layer",
        "",
        "| term | coef |",
        "|---|---|",
        f"| intercept | {fm['intercept']:+.4f} |",
        f"| logit(p_rating) | {fm['rating_coef']:+.4f} |",
    ]
    for name, c in fm["coefs"].items():
        lines.append(f"| {name} | {c:+.4f} |")
    lines += [
        "",
        f"## Verdict: **{report['verdict']}**",
        "",
        "- `VALIDATED_BLEND` — blend beats rating-only on holdout Brier with"
        " sane calibration → ship blend (feature_model in state file).",
        "- `VALIDATED_RATING_ONLY` — blend failed but the rating alone is"
        " calibrated → ship rating-only (no feature_model).",
        "- `REJECTED_SHARP_ONLY` — neither calibrated → sector ships"
        " sharp-only (`models: [sharp]`, the lol/cs2 precedent).",
        "",
        "Live predictions blend with a heavy Pinnacle sharp anchor"
        " (sharp_weight ≈ 0.85+), so the model's job is marginal CLV, not"
        " beating the closing line outright. Promotion gate: n ≥ 30 resolved"
        " shadow bets with CLV ≥ 0 (`evmax cleanup shadow promote ufc`).",
    ]
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--eval-start", default="2015-01-01")
    ap.add_argument("--min-n", type=int, default=1)
    args = ap.parse_args()
    main(args.train_end, args.eval_start, args.min_n)
