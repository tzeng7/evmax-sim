#!/usr/bin/env python3
"""Watch Kalshi's published fee schedule for drift.

Kalshi has no numeric fee API — the authoritative rates live in a dated PDF
(``https://kalshi.com/docs/kalshi-fee-schedule.pdf``). Our trading math hard-
codes the taker coefficient and maker multiplier in :mod:`evmax.fees`, so if
Kalshi changes the schedule those constants silently go stale. This script is
the canary: it fetches the PDF, extracts the coefficients + effective date +
a normalized-text hash, and compares them against a committed snapshot
(``data/kalshi_fee_schedule.snapshot.json``) AND the live constants in
``evmax/fees.py``.

Design (so the logic is testable offline, with no network / no pypdf):
  - ``parse_schedule(text)``           PURE — pull rates + effective date from text.
  - ``normalize_ws`` / ``content_hash`` PURE — stable hash of the document body.
  - ``classify(current, snapshot, constants)`` PURE — the drift verdict.
  - ``fetch_pdf_text(url)``            NETWORK + pypdf — only the scheduled job hits this.

Verdicts:
  match         extracted rates == snapshot == fees.py constants, body hash unchanged.
  hard_drift    a rate we TRADE ON disagrees with fees.py or the snapshot. Reconcile
                evmax/fees.py first, then re-run with --update to bump the snapshot.
  soft_drift    rates still agree, but the document body changed (per-series
                multiplier tables / wording / effective date). Human review.
  inconclusive  fetch or parse failed (429 / format change) — never a false alarm.

CLI:
  check_kalshi_fees.py                 # online: fetch, classify, write $GITHUB_OUTPUT, exit per verdict
  check_kalshi_fees.py --offline F     # classify a text file instead of fetching (tests / offline)
  check_kalshi_fees.py --update        # fetch and (re)write the snapshot (after reconciling fees.py)
  check_kalshi_fees.py --notify-file R # POST result R's summary via evmax.notifications.Notifier
  check_kalshi_fees.py --json          # machine-readable result to stdout

Exit codes (also mirrored as the `verdict` GITHUB_OUTPUT so the workflow can gate):
  0 match / inconclusive   3 soft_drift   4 hard_drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Repo-root anchored paths so the script works from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = _REPO_ROOT / "data" / "kalshi_fee_schedule.snapshot.json"
RESULT_PATH = _REPO_ROOT / "kalshi-fee-result.json"  # written each online run (CI artifact)
DEFAULT_URL = "https://kalshi.com/docs/kalshi-fee-schedule.pdf"

# A browser-like UA — the fee PDF 429s naive/no-UA clients (verified 2026-08-08).
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_VERDICT_EXIT = {"match": 0, "inconclusive": 0, "soft_drift": 3, "hard_drift": 4}
_TOL = 1e-9


class FetchError(RuntimeError):
    """Raised when the PDF cannot be fetched or its text extracted."""


# ---------------------------------------------------------------------------
# PURE helpers (no network, no pypdf) — the unit-tested core
# ---------------------------------------------------------------------------

def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to a single space.

    Makes the content hash robust to the spacing/line-break noise that PDF text
    extraction introduces, so a cosmetic re-export doesn't read as a fee change.
    """
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    """SHA-256 of the whitespace-normalized document body."""
    return hashlib.sha256(normalize_ws(text).encode("utf-8")).hexdigest()


def parse_schedule(text: str) -> dict:
    """Extract the coefficients we rely on from the fee-schedule text.

    The schedule states the fee as ``round up(M x <rate> x C x P x (1-P))``
    twice — once for the taker base rate (0.07) and once for the maker base
    rate (0.0175). ``M`` is the per-series multiplier carried in Tables 1/2.
    Returns ``taker_rate`` / ``maker_rate`` / ``maker_mult`` / ``effective_date``;
    any field is ``None`` when the pattern is not found (format changed).
    """
    t = normalize_ws(text)

    coeffs = re.findall(
        r"round up\(\s*M x ([0-9]*\.?[0-9]+) x C x P x \(1\s*-\s*P\)\)", t
    )
    m_maker = re.search(
        r"Maker Fees fees = round up\(\s*M x ([0-9]*\.?[0-9]+)", t
    )
    maker_rate = float(m_maker.group(1)) if m_maker else None
    # Taker = the round-up coefficient that is NOT the maker one.
    taker_rate = next(
        (float(c) for c in coeffs if maker_rate is None or abs(float(c) - maker_rate) > _TOL),
        None,
    )
    eff = re.search(
        r"Last updated and effective:\s*([A-Za-z]+ \d{1,2}, \d{4})", t
    )
    maker_mult = (
        round(maker_rate / taker_rate, 6)
        if (maker_rate is not None and taker_rate not in (None, 0))
        else None
    )
    return {
        "taker_rate": taker_rate,
        "maker_rate": maker_rate,
        "maker_mult": maker_mult,
        "effective_date": eff.group(1) if eff else None,
        "formula": "round up(M x <rate> x C x P x (1-P))",
    }


def build_current(text: str, *, source_url: str = DEFAULT_URL) -> dict:
    """Assemble the current-state record from raw schedule text."""
    return {
        "source_url": source_url,
        "extracted": parse_schedule(text),
        "content_sha256": content_hash(text),
    }


def _approx(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(_TOL, 1e-6 * max(abs(a), abs(b)))


def classify(current: dict, snapshot: Optional[dict], constants: dict) -> tuple[str, list[str]]:
    """Compare current state to the snapshot AND the live fees.py constants.

    HARD drift beats SOFT: a rate we trade on disagreeing with our code is the
    signal that matters. A missing taker rate means the document format changed
    under us — inconclusive, not a false rate alarm.
    """
    cur = current["extracted"]

    if cur["taker_rate"] is None:
        return "inconclusive", ["taker rate not found in document (PDF format changed?)"]

    hard: list[str] = []
    if not _approx(cur["taker_rate"], constants["taker_rate"]):
        hard.append(
            f"taker_rate {cur['taker_rate']} != evmax.fees.KALSHI_TAKER_RATE {constants['taker_rate']}"
        )
    if cur["maker_mult"] is not None and not _approx(cur["maker_mult"], constants["maker_mult"]):
        hard.append(
            f"maker_mult {cur['maker_mult']} != evmax.fees.KALSHI_MAKER_RATE_MULT {constants['maker_mult']}"
        )
    if snapshot is not None:
        snap = snapshot.get("extracted", {})
        if not _approx(cur["taker_rate"], snap.get("taker_rate")):
            hard.append(f"taker_rate {snap.get('taker_rate')} -> {cur['taker_rate']} vs snapshot")
        if cur["maker_rate"] is not None and not _approx(cur["maker_rate"], snap.get("maker_rate")):
            hard.append(f"maker_rate {snap.get('maker_rate')} -> {cur['maker_rate']} vs snapshot")
    if hard:
        return "hard_drift", hard

    if snapshot is None:
        return "match", []  # nothing to diff the body against yet

    soft: list[str] = []
    snap = snapshot.get("extracted", {})
    if current["content_sha256"] != snapshot.get("content_sha256"):
        soft.append("normalized content hash changed (per-series tables / wording / date)")
    if cur["effective_date"] != snap.get("effective_date"):
        soft.append(f"effective_date {snap.get('effective_date')!r} -> {cur['effective_date']!r}")
    if soft:
        return "soft_drift", soft
    return "match", []


def get_constants() -> dict:
    """The rates our trading math actually uses (single source: evmax.fees)."""
    from evmax.fees import KALSHI_MAKER_RATE_MULT, KALSHI_TAKER_RATE

    return {"taker_rate": KALSHI_TAKER_RATE, "maker_mult": KALSHI_MAKER_RATE_MULT}


# ---------------------------------------------------------------------------
# I/O side effects
# ---------------------------------------------------------------------------

def load_snapshot(path: Optional[Path] = None) -> Optional[dict]:
    # Resolve at call time (not as a bound default) so tests can monkeypatch
    # the module-level SNAPSHOT_PATH.
    path = path if path is not None else SNAPSHOT_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_snapshot(current: dict, path: Optional[Path] = None, *, today: str) -> None:
    path = path if path is not None else SNAPSHOT_PATH
    payload = {
        "source_url": current["source_url"],
        "fetched_at": today,
        "extracted": current["extracted"],
        "content_sha256": current["content_sha256"],
        "note": (
            "Regenerate after Kalshi changes the schedule: reconcile evmax/fees.py, "
            "then `uv run --with pypdf==6.15.0 scripts/check_kalshi_fees.py --update`."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def fetch_pdf_text(url: str = DEFAULT_URL, *, timeout: float = 20.0, retries: int = 3) -> str:
    """Fetch the fee PDF and return its extracted text.

    Retries with linear backoff on 429/5xx/transport errors. Raises
    :class:`FetchError` (→ inconclusive verdict) rather than guessing on a
    transient failure, so a rate-limit never looks like a fee change.
    """
    import io
    import time

    import httpx
    from pypdf import PdfReader

    last = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": _UA, "Accept": "application/pdf,*/*"},
                timeout=timeout,
                follow_redirects=True,
            )
            ctype = resp.headers.get("content-type", "")
            if resp.status_code == 200 and "pdf" in ctype.lower():
                reader = PdfReader(io.BytesIO(resp.content))
                return "\n".join((p.extract_text() or "") for p in reader.pages)
            last = f"HTTP {resp.status_code} content-type={ctype!r}"
        except Exception as exc:  # noqa: BLE001 — surface any transport/parse failure as inconclusive
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2 * attempt)
    raise FetchError(last)


def _emit_github_output(verdict: str, summary: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(f"verdict={verdict}\n")
        fh.write(f"summary={summary}\n")


def _summary(verdict: str, reasons: list[str], current: dict) -> str:
    ex = current["extracted"]
    head = {
        "match": "Kalshi fee schedule unchanged.",
        "soft_drift": "Kalshi fee schedule body changed (rates still match).",
        "hard_drift": "Kalshi fee RATE changed — reconcile evmax/fees.py.",
        "inconclusive": "Kalshi fee watch inconclusive (fetch/parse failed).",
    }[verdict]
    tail = f" taker={ex['taker_rate']} maker_mult={ex['maker_mult']} eff={ex['effective_date']!r}."
    if reasons:
        tail += " " + "; ".join(reasons)
    return head + tail


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Watch Kalshi's fee schedule for drift.")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--offline", metavar="TEXT_FILE", help="Parse this text file instead of fetching.")
    ap.add_argument("--update", action="store_true", help="Fetch and (re)write the committed snapshot.")
    ap.add_argument("--notify-file", metavar="RESULT_JSON", help="POST a prior result's summary via Notifier.")
    ap.add_argument("--json", action="store_true", help="Emit the full result JSON to stdout.")
    ap.add_argument("--today", default=None, help="Override the fetched_at date (YYYY-MM-DD) for --update.")
    args = ap.parse_args(argv)

    # Alert-only mode: read a result file and push it to the webhook(s).
    if args.notify_file:
        result = json.loads(Path(args.notify_file).read_text())
        from evmax.notifications import Notifier

        notifier = Notifier.from_settings()
        if notifier.is_configured():
            notifier.send_text(f":rotating_light: {result['summary']}\n{result['source_url']}")
            print("notified")
        else:
            print("no webhook configured — skipping notify")
        return 0

    # Acquire the schedule text.
    try:
        text = Path(args.offline).read_text() if args.offline else fetch_pdf_text(args.url)
    except FetchError as exc:
        verdict, reasons = "inconclusive", [str(exc)]
        summary = f"Kalshi fee watch inconclusive (fetch/parse failed). {exc}"
        _emit_github_output(verdict, summary)
        RESULT_PATH.write_text(json.dumps(
            {"verdict": verdict, "reasons": reasons, "summary": summary, "source_url": args.url}, indent=2
        ) + "\n")
        print(summary, file=sys.stderr)
        return _VERDICT_EXIT[verdict]

    current = build_current(text, source_url=args.url)

    if args.update:
        today = args.today or __import__("datetime").date.today().isoformat()
        write_snapshot(current, today=today)
        print(f"snapshot written: {SNAPSHOT_PATH}")
        if args.json:
            print(json.dumps(current, indent=2))
        return 0

    snapshot = load_snapshot()
    verdict, reasons = classify(current, snapshot, get_constants())
    summary = _summary(verdict, reasons, current)

    result = {
        "verdict": verdict,
        "reasons": reasons,
        "summary": summary,
        "source_url": args.url,
        "current": current,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    _emit_github_output(verdict, summary)
    print(json.dumps(result, indent=2) if args.json else summary)
    return _VERDICT_EXIT[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
