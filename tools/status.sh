#!/usr/bin/env bash
# Lightweight status script with graceful fallbacks.
# Runs tests, lint, type-checks, and mutation test summary if tools exist.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

ok=()
skipped=()
failed=()

run_or_skip() {
  local name="$1"; shift
  if command -v "$1" >/dev/null 2>&1; then
    echo "==> $name"
    if "$@"; then
      ok+=("$name")
    else
      failed+=("$name")
    fi
  else
    echo "==> $name: tool not found, skipping"
    skipped+=("$name")
  fi
}

# Tests with coverage
run_or_skip "pytest" pytest -q --disable-warnings --maxfail=1 --cov=. --cov-report=term-missing

# Diff coverage (if git + diff-cover available); do not fail gate yet
if command -v git >/dev/null 2>&1 && command -v diff-cover >/dev/null 2>&1; then
  echo "==> diff-cover (informational)"
  # Generate coverage.xml if not present
  if [ ! -f coverage.xml ] && command -v coverage >/dev/null 2>&1; then
    coverage xml || true
  fi
  git diff -U0 HEAD~1..HEAD | cat >/dev/null 2>&1 || true
  diff-cover coverage.xml --compare-branch=HEAD~1 || true
else
  echo "==> diff-cover: tool or git not found, skipping"
fi

# Lint
run_or_skip "ruff" ruff check .

# Type checks
run_or_skip "mypy" mypy --strict .

# Mutation tests (summary only)
if command -v mutmut >/dev/null 2>&1; then
  echo "==> mutmut (summary)"
  mutmut run --use-coverage || true
  mutmut results || true
else
  echo "==> mutmut: tool not found, skipping"
fi

echo
echo "Summary:"
echo "  OK:       ${#ok[@]} -> ${ok[*]:-}"
echo "  Skipped:  ${#skipped[@]} -> ${skipped[*]:-}"
echo "  Failed:   ${#failed[@]} -> ${failed[*]:-}"

# Always exit 0 for smoke-friendly status (gates come later)
exit 0
