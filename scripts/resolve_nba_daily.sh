#!/bin/bash
set -euo pipefail

cd /Users/ktzeng/Projects/evmax

YESTERDAY=$(date -v-1d +%Y-%m-%d)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nba_resolve_${YESTERDAY}.log"

{
    echo "=== $(date) — resolving NBA projections for $YESTERDAY ==="
    .venv/bin/evmax project resolve --date "$YESTERDAY" --sector nba
    echo "=== done ==="
} >> "$LOG_FILE" 2>&1
