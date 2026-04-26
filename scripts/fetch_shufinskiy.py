"""Download shufinskiy/nba_data historical archives.

Pulls tar.xz files from the public GitHub repo, decompresses to CSV, and
caches as parquet for fast subsequent reads. Idempotent — already-extracted
files are skipped on re-run unless --force is passed.

Usage:
    # Default: NBA shot details + pbpstats for the last 5 seasons
    python scripts/fetch_shufinskiy.py

    # Custom: specific files (use the keys from list_data.txt)
    python scripts/fetch_shufinskiy.py --keys shotdetail_2025,pbpstats_2024

    # Include WNBA archives (separate use case — see TODO MODEL-12 follow-on)
    python scripts/fetch_shufinskiy.py --wnba

    # Re-download even if files exist
    python scripts/fetch_shufinskiy.py --force

Files land at:
    data/historical_nba/{key}.tar.xz       (downloaded archive)
    data/historical_nba/{key}.csv          (extracted; created on first read)
    data/historical_nba/{key}.parquet      (cached for fast load)

Year convention reminder: `_2025` means the season STARTING in 2025
(i.e., the 2025-26 NBA season). `_2024` = 2024-25, etc.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "historical_nba"
LIST_DATA_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"

# Default download set: 5 most-recent NBA seasons of shot details and the
# 5 most-recent of pbpstats. shot details are through 2025 (current season);
# pbpstats lag one season (last available is 2024 = the 2024-25 season).
DEFAULT_KEYS = [
    f"shotdetail_{y}" for y in (2021, 2022, 2023, 2024, 2025)
] + [
    f"pbpstats_{y}" for y in (2020, 2021, 2022, 2023, 2024)
]


def load_url_index() -> dict[str, str]:
    """Fetch the canonical `key=url` map from the upstream repo."""
    r = httpx.get(LIST_DATA_URL, timeout=15)
    r.raise_for_status()
    out: dict[str, str] = {}
    for line in r.text.strip().splitlines():
        if "=" not in line:
            continue
        key, url = line.split("=", 1)
        out[key.strip()] = url.strip()
    return out


def download(key: str, url: str, force: bool = False) -> Path:
    target = DATA_DIR / f"{key}.tar.xz"
    if target.exists() and not force:
        return target
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {key} ...", end=" ", flush=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        size = 0
        with target.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                size += len(chunk)
    print(f"{size / 1024 / 1024:.1f} MB")
    return target


def extract(archive: Path, force: bool = False) -> Path:
    """Extract the single CSV inside a tar.xz to its sibling location."""
    csv_path = archive.with_suffix("").with_suffix(".csv")
    if csv_path.exists() and not force:
        return csv_path
    with tarfile.open(archive, "r:xz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".csv")]
        if not members:
            raise RuntimeError(f"no CSV in {archive.name}")
        tf.extract(members[0], DATA_DIR)
    return csv_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--keys", default=None,
        help=f"Comma-separated keys (default: {len(DEFAULT_KEYS)} files — recent NBA shotdetail + pbpstats)",
    )
    p.add_argument("--wnba", action="store_true", help="Include WNBA archives in default set")
    p.add_argument("--force", action="store_true", help="Re-download / re-extract existing files")
    args = p.parse_args()

    if args.keys:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    else:
        keys = list(DEFAULT_KEYS)
        if args.wnba:
            keys += [f"wnba_{prefix}_{y}" for prefix in ("shotdetail", "pbpstats")
                     for y in (2021, 2022, 2023, 2024, 2025)]

    print(f"Loading URL index from {LIST_DATA_URL}")
    index = load_url_index()

    missing = [k for k in keys if k not in index]
    if missing:
        print(f"error: keys not in upstream list: {missing}", file=sys.stderr)
        print(f"  available shotdetail keys (sample): "
              f"{[k for k in index if k.startswith('shotdetail_2')][:6]}", file=sys.stderr)
        return 2

    print(f"Fetching {len(keys)} archives → {DATA_DIR}")
    for k in keys:
        archive = download(k, index[k], force=args.force)
        csv_path = extract(archive, force=args.force)
        size_mb = csv_path.stat().st_size / 1024 / 1024
        print(f"  ✓ {csv_path.name} ({size_mb:.1f} MB)")

    print(f"\nDone. {len(keys)} files ready at {DATA_DIR}")
    print("Next: import evmax.backtest.sources.shufinskiy to load them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
