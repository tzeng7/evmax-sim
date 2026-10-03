#!/usr/bin/env python3
"""Watch Polymarket US's published fee schedule for drift.

Polymarket US publishes its fee schedule as a docs page
(``https://docs.polymarket.us/fees``; the ``.md`` twin is clean markdown, no PDF
library needed). Our trading math hard-codes the taker/maker theta in
:mod:`evmax.fees`, so a schedule change silently stales those constants — the
taker theta moved 0.06 → 0.0695 on 2026-10-01 and nothing flagged it. This is the
Polymarket US sibling of ``check_kalshi_fees.py``: it extracts the thetas, the
effective date, the rounding rule and the published "fee by price" table, then
compares them with a committed snapshot
(``data/polymarket_us_fee_schedule.snapshot.json``) AND the live ``fees.py``
constants AND ``polymarket_us_order_fee`` itself (every 100-lot row of the table).

Same verdicts and exit codes as the Kalshi watcher (reuses its pure helpers):
  match         thetas == snapshot == fees.py, table reproduced, body hash unchanged.
  hard_drift    a number we TRADE ON disagrees with fees.py / the snapshot, the
                fee-by-price table disagrees with our order-fee function, or the
                rounding rule is no longer banker's. Reconcile evmax/fees.py, then
                re-run with --update.
  soft_drift    numbers agree, document body / effective date changed. Human review.
  inconclusive  fetch or parse failed — never a false alarm.

Per-market canary (online only): every Polymarket US market object also carries
its own ``feeCoefficient`` (public gateway, same ``/v2/leagues/{slug}/events``
payload the scanner reads). The watcher sweeps every league we scan and flags any
market whose coefficient differs from ``POLYMARKET_US_TAKER_THETA`` as hard drift
— catching a per-league / per-market-type override the docs page would not show
(the Polymarket US analogue of ``check_kalshi_series_fees.py``). A market with a
null coefficient is counted but not flagged; a league that fails to fetch is
reported but never changes the verdict.

CLI:
  check_polymarket_us_fees.py                # fetch, classify, write $GITHUB_OUTPUT
  check_polymarket_us_fees.py --offline F    # classify a markdown file instead
  check_polymarket_us_fees.py --update       # (re)write the snapshot after reconciling fees.py
  check_polymarket_us_fees.py --json

Exit codes: 0 match / inconclusive   3 soft_drift   4 hard_drift
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from check_kalshi_fees import (  # noqa: E402 — shared pure helpers + verdict plumbing
    _VERDICT_EXIT,
    _approx,
    _emit_github_output,
    content_hash,
    normalize_ws,
)

SNAPSHOT_PATH = _REPO_ROOT / "data" / "polymarket_us_fee_schedule.snapshot.json"
RESULT_PATH = _REPO_ROOT / "polymarket-us-fee-result.json"  # written each online run (CI artifact)
DEFAULT_URL = "https://docs.polymarket.us/fees.md"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# "Taker pays / Maker receives" are rounded to the cent, so allow a cent of slack.
_TABLE_TOL = 0.0101


class FetchError(RuntimeError):
    """Raised when the page cannot be fetched."""


# ---------------------------------------------------------------------------
# PURE helpers
# ---------------------------------------------------------------------------

def parse_schedule(text: str) -> dict:
    """Extract the numbers we rely on. Any field is ``None``/empty when not found."""
    t = normalize_ws(text.replace("\\$", "$"))

    taker = re.search(r"\*\*Taker Fee\*\*\s*\|\s*(-?[0-9]*\.?[0-9]+)", t)
    maker = re.search(r"\*\*Maker Rebate\*\*\s*\|\s*(-?[0-9]*\.?[0-9]+)", t)
    eff = re.search(r"Effective exchange-wide from (.+?)\.\s*(?:</Info>|\S)", t)
    # Fee-by-price rows: | $0.50 | $50 | $1.74 | $0.31 |
    table: dict[str, list[float]] = {}
    for p, taker_pays, maker_gets in re.findall(
        r"\|\s*\$(0\.\d\d)\s*\|\s*\$[0-9.,]+\s*\|\s*\$([0-9.]+)\s*\|\s*\$([0-9.]+)\s*\|", t
    ):
        table.setdefault(p, [float(taker_pays), float(maker_gets)])
    return {
        "taker_theta": float(taker.group(1)) if taker else None,
        "maker_theta": float(maker.group(1)) if maker else None,
        "effective": eff.group(1).strip() if eff else None,
        "bankers_rounding": "banker's rounding" in t.lower(),
        "fee_table": table,
        "formula": "Fee = theta x C x p x (1 - p)",
    }


def build_current(text: str, *, source_url: str = DEFAULT_URL) -> dict:
    return {
        "source_url": source_url,
        "extracted": parse_schedule(text),
        "content_sha256": content_hash(text),
    }


def get_constants() -> dict:
    """What our trading math uses (single source: evmax.fees) + a table-check fn."""
    from evmax.fees import (
        POLYMARKET_US_MAKER_THETA,
        POLYMARKET_US_TAKER_THETA,
        polymarket_us_order_fee,
    )

    return {
        "taker_theta": POLYMARKET_US_TAKER_THETA,
        "maker_theta": POLYMARKET_US_MAKER_THETA,
        "order_fee": polymarket_us_order_fee,
    }


def check_fee_table(table: dict[str, list[float]], order_fee) -> list[str]:
    """Compare every published 100-lot row with ``polymarket_us_order_fee``."""
    bad: list[str] = []
    for p_str, (taker_pays, maker_gets) in sorted(table.items()):
        p = float(p_str)
        ours_t = order_fee(p, 100)
        ours_m = -order_fee(p, 100, maker=True)
        if abs(ours_t - taker_pays) > _TABLE_TOL:
            bad.append(f"taker @ ${p_str}: docs ${taker_pays:.2f} vs fees.py ${ours_t:.2f}")
        if abs(ours_m - maker_gets) > _TABLE_TOL:
            bad.append(f"maker @ ${p_str}: docs ${maker_gets:.2f} vs fees.py ${ours_m:.2f}")
    return bad


def classify_market_coefficients(
    by_league: dict[str, Optional[dict[str, int]]], theta: float
) -> tuple[list[str], dict]:
    """PURE. ``{league: {coefficient_str|'null': n_markets} | None}`` → (hard reasons, stats).

    ``None`` for a league means its fetch failed (reported, never a mismatch).
    """
    reasons: list[str] = []
    checked = null = 0
    unavailable: list[str] = []
    for league, counts in sorted(by_league.items()):
        if counts is None:
            unavailable.append(league)
            continue
        for coef, n in sorted(counts.items()):
            if coef == "null":
                null += n
                continue
            checked += n
            if not _approx(float(coef), theta):
                reasons.append(
                    f"{league}: {n} market(s) feeCoefficient {coef} != POLYMARKET_US_TAKER_THETA {theta}"
                )
    return reasons, {"checked": checked, "null": null, "unavailable": unavailable}


def classify(current: dict, snapshot: Optional[dict], constants: dict) -> tuple[str, list[str]]:
    """HARD (a traded number is wrong) beats SOFT (body changed); no theta ⇒ inconclusive."""
    cur = current["extracted"]
    if cur["taker_theta"] is None or cur["maker_theta"] is None:
        return "inconclusive", ["taker/maker theta not found in document (page format changed?)"]

    hard: list[str] = []
    if not _approx(cur["taker_theta"], constants["taker_theta"]):
        hard.append(f"taker_theta {cur['taker_theta']} != evmax.fees.POLYMARKET_US_TAKER_THETA {constants['taker_theta']}")
    if not _approx(cur["maker_theta"], constants["maker_theta"]):
        hard.append(f"maker_theta {cur['maker_theta']} != evmax.fees.POLYMARKET_US_MAKER_THETA {constants['maker_theta']}")
    if not cur["bankers_rounding"]:
        hard.append("rounding rule no longer mentions banker's rounding (fees.py assumes half-to-even)")
    if cur["fee_table"] and "order_fee" in constants:
        hard.extend(check_fee_table(cur["fee_table"], constants["order_fee"]))
    if snapshot is not None:
        snap = snapshot.get("extracted", {})
        for key in ("taker_theta", "maker_theta"):
            if not _approx(cur[key], snap.get(key)):
                hard.append(f"{key} {snap.get(key)} -> {cur[key]} vs snapshot")
    if hard:
        return "hard_drift", hard

    if snapshot is None:
        return "match", []
    soft: list[str] = []
    if current["content_sha256"] != snapshot.get("content_sha256"):
        soft.append("normalized content hash changed (wording / tables / combo schedule / rebates)")
    if cur["effective"] != snapshot.get("extracted", {}).get("effective"):
        soft.append(f"effective {snapshot.get('extracted', {}).get('effective')!r} -> {cur['effective']!r}")
    return ("soft_drift", soft) if soft else ("match", [])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_snapshot(path: Optional[Path] = None) -> Optional[dict]:
    path = path if path is not None else SNAPSHOT_PATH
    return json.loads(path.read_text()) if path.exists() else None


def write_snapshot(current: dict, path: Optional[Path] = None, *, today: str) -> None:
    path = path if path is not None else SNAPSHOT_PATH
    payload = {
        "source_url": current["source_url"],
        "fetched_at": today,
        "extracted": current["extracted"],
        "content_sha256": current["content_sha256"],
        "note": "Regenerate after Polymarket US changes the schedule: reconcile evmax/fees.py, "
                "then `uv run scripts/check_polymarket_us_fees.py --update`.",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def fetch_page_text(url: str = DEFAULT_URL, *, timeout: float = 20.0, retries: int = 3) -> str:
    """Fetch the markdown page; transient failures raise FetchError (→ inconclusive)."""
    import time

    import httpx

    last = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(url, headers={"User-Agent": _UA}, timeout=timeout, follow_redirects=True)
            if resp.status_code == 200 and "Fee" in resp.text:
                return resp.text
            last = f"HTTP {resp.status_code} content-type={resp.headers.get('content-type')!r}"
        except Exception as exc:  # noqa: BLE001 — surfaced as inconclusive
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2 * attempt)
    raise FetchError(last)


def fetch_market_coefficients(leagues: list[str], *, timeout: float = 20.0) -> dict[str, Optional[dict[str, int]]]:
    """``{league: {feeCoefficient: n_markets}}`` from the public gateway; ``None`` on failure."""
    import httpx

    from evmax.settings import get_settings

    base = get_settings().polymarket_us_base_url.rstrip("/")
    out: dict[str, Optional[dict[str, int]]] = {}
    with httpx.Client(timeout=timeout, headers={"User-Agent": _UA}) as client:
        for league in leagues:
            try:
                r = client.get(f"{base}/v2/leagues/{league}/events", params={"limit": 100})
                r.raise_for_status()
                counts: dict[str, int] = {}
                for ev in r.json().get("events", []) or []:
                    for m in ev.get("markets", []) or []:
                        k = "null" if m.get("feeCoefficient") is None else str(m["feeCoefficient"])
                        counts[k] = counts.get(k, 0) + 1
                out[league] = counts
            except Exception:  # noqa: BLE001 — reported as unavailable, never fatal
                out[league] = None
    return out


def scanned_leagues() -> list[str]:
    from evmax.arb import ARB_LEAGUE_MAP

    seen: list[str] = []
    for leagues in ARB_LEAGUE_MAP.values():
        for lg in leagues:
            if lg not in seen:
                seen.append(lg)
    return seen


def _summary(verdict: str, reasons: list[str], current: dict) -> str:
    ex = current["extracted"]
    head = {
        "match": "Polymarket US fee schedule unchanged.",
        "soft_drift": "Polymarket US fee schedule body changed (rates still match).",
        "hard_drift": "Polymarket US fee RATE changed — reconcile evmax/fees.py.",
        "inconclusive": "Polymarket US fee watch inconclusive (fetch/parse failed).",
    }[verdict]
    tail = f" taker={ex['taker_theta']} maker={ex['maker_theta']} eff={ex['effective']!r}."
    return head + tail + ((" " + "; ".join(reasons[:6])) if reasons else "")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Watch Polymarket US's fee schedule for drift.")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--offline", metavar="TEXT_FILE", help="Parse this file instead of fetching.")
    ap.add_argument("--update", action="store_true", help="Fetch and (re)write the committed snapshot.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--today", default=None, help="Override fetched_at (YYYY-MM-DD) for --update.")
    args = ap.parse_args(argv)

    try:
        text = Path(args.offline).read_text() if args.offline else fetch_page_text(args.url)
    except FetchError as exc:
        summary = f"Polymarket US fee watch inconclusive (fetch/parse failed). {exc}"
        _emit_github_output("inconclusive", summary)
        RESULT_PATH.write_text(json.dumps(
            {"verdict": "inconclusive", "reasons": [str(exc)], "summary": summary, "source_url": args.url}, indent=2
        ) + "\n")
        print(summary, file=sys.stderr)
        return _VERDICT_EXIT["inconclusive"]

    current = build_current(text, source_url=args.url)
    if args.update:
        today = args.today or __import__("datetime").date.today().isoformat()
        write_snapshot(current, today=today)
        print(f"snapshot written: {SNAPSHOT_PATH}")
        return 0

    constants = get_constants()
    verdict, reasons = classify(current, load_snapshot(), constants)
    if not args.offline and verdict != "inconclusive":
        coef_reasons, stats = classify_market_coefficients(
            fetch_market_coefficients(scanned_leagues()), constants["taker_theta"]
        )
        if coef_reasons:
            verdict, reasons = "hard_drift", reasons + coef_reasons
        if stats["unavailable"]:
            reasons = reasons + [f"note: feeCoefficient sweep skipped leagues {','.join(stats['unavailable'])}"]
        print(f"feeCoefficient sweep: {stats['checked']} markets checked, {stats['null']} null", file=sys.stderr)
    summary = _summary(verdict, reasons, current)
    RESULT_PATH.write_text(json.dumps(
        {"verdict": verdict, "reasons": reasons, "summary": summary, "source_url": args.url, "current": current},
        indent=2,
    ) + "\n")
    _emit_github_output(verdict, summary)
    print(json.dumps({"verdict": verdict, "reasons": reasons, "summary": summary}, indent=2) if args.json else summary)
    return _VERDICT_EXIT[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
