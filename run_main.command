#!/usr/bin/env bash
# Double-click launcher (macOS). Runs the news summarizer interactively.
cd "$(dirname "$0")"
PY="$HOME/.venvs/labs/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" main.py "$@"
echo
echo "Press any key to close..."
read -n 1 -s
