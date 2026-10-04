#!/usr/bin/env bash
# tests/eval/run.sh — wrapper around python -m tests.eval
#
# Mirrors tests/e2e/run.sh convention. Forwarded args are passed
# verbatim to the CLI. Examples:
#
#   bash tests/eval/run.sh --suite golden
#   bash tests/eval/run.sh --suite adversarial --judge programmatic
#
set -euo pipefail

# Resolve repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

exec python -m tests.eval "$@"