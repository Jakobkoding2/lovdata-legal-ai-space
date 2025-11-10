#!/usr/bin/env bash
# Simple start wrapper. Render will provide $PORT at runtime.
# This script is made executable by the Dockerfile (chmod +x).

if [ -n "$1" ] && [ -z "$PORT" ]; then
  export PORT="$1"
fi

echo "Starting app on PORT=${PORT:-7860}"
exec python app.py
