#!/usr/bin/env bash
# Layer 5 e2e runner — install Chromium, then run smoke.spec.ts against
# the build + preview server. This is the canonical verification step
# before shipping any fix (see memory/development-methodology.md).
#
# Pre-conditions:
#   - FastAPI backend already running on http://127.0.0.1:8765 (with
#     .env containing a real LLM key, BGE-M3 / Reranker preloaded)
#   - Node.js + npm available
#
# Usage:
#   bash tests/e2e/run.sh          # full: install + build + run
#   SKIP_INSTALL=1 bash tests/e2e/run.sh   # skip the playwright install
#                                          # (use on machines that already
#                                          # have Chromium downloaded)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/src/frontend"
E2E_DIR="${REPO_ROOT}/tests/e2e"

echo ">>> [1/3] ensure frontend deps installed"
cd "${FRONTEND_DIR}"
if [[ ! -d node_modules/@playwright/test ]]; then
  npm install
else
  echo "    @playwright/test already present; skipping npm install"
fi

echo ">>> [2/3] ensure Chromium downloaded"
if [[ -z "${SKIP_INSTALL:-}" ]]; then
  npx playwright install chromium --with-deps
else
  echo "    SKIP_INSTALL=1; trusting that Chromium is already present"
fi

echo ">>> [3/3] run Layer 5 smoke"
# Run from FRONTEND_DIR so npx can resolve @playwright/test
# (which is installed in src/frontend/node_modules). Path is
# two `..` because FRONTEND_DIR is `src/frontend/` and the config
# lives at `tests/e2e/playwright.config.ts`.
cd "${FRONTEND_DIR}"
# NODE_PATH lets the config file at tests/e2e/ resolve
# @playwright/test from src/frontend/node_modules. Without this,
# Node's resolver only looks next to the config file.
export NODE_PATH="${FRONTEND_DIR}/node_modules"
npx playwright test --config=../../tests/e2e/playwright.config.ts "$@"

echo ""
echo "Layer 5 finished. If any assertion failed, the fix is NOT shipped."
echo "See tests/e2e/output/ for traces / videos of any failure."