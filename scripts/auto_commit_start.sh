#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$ROOT_DIR/.codex/auto-commit.log"
PID_FILE="$ROOT_DIR/.codex/auto-commit.pid"

mkdir -p "$ROOT_DIR/.codex"

if [[ -f "$PID_FILE" ]]; then
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Auto-commit is already running (pid=$(cat "$PID_FILE"))."
    exit 0
  fi
fi

nohup "$ROOT_DIR/scripts/auto_commit.py" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Auto-commit started (pid=$(cat "$PID_FILE"))."
echo "Log: $LOG_FILE"
