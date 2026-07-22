#!/usr/bin/env bash
# Double-click launcher (macOS). Runs the unit test suite.
cd "$(dirname "$0")"
PY="$HOME/.venvs/labs/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" -m pytest test_summarizer.py -v
echo
echo "Press any key to close..."
read -n 1 -s
