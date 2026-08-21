#!/usr/bin/env bash
# Scan the tracked tree for credential patterns. Run before every push.
# Exits non-zero if anything matches, so it can gate a push.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

PATTERN='hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{10,}'

# git grep searches only tracked files, which is exactly the set that can be pushed.
if git grep -nIE "$PATTERN" -- . ; then
  echo "SECRET SCAN FAILED: credential pattern in a tracked file" >&2
  exit 1
fi

if git ls-files | grep -qE '(^|/)\.env$|\.pem$|\.p12$|(^|/)id_rsa'; then
  echo "SECRET SCAN FAILED: credential-shaped filename is tracked" >&2
  git ls-files | grep -E '(^|/)\.env$|\.pem$|\.p12$|(^|/)id_rsa' >&2
  exit 1
fi

echo "secret scan clean: $(git ls-files | wc -l | tr -d ' ') tracked files"
