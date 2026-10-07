#!/usr/bin/env bash
# Start the web front end on http://localhost:8010
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec .venv/bin/uvicorn web.server:app --host 127.0.0.1 --port 8010 --reload
