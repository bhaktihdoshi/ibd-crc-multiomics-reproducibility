#!/usr/bin/env bash
set -eu
ROOT="${1:-$(git rev-parse --show-toplevel)}"
cd "$ROOT"
EXPECTED="$(mktemp)"
ACTUAL="$(mktemp)"
SCAN="$(mktemp)"
trap 'rm -f "$EXPECTED" "$ACTUAL" "$SCAN"' EXIT
sort final_analysis/focused_validation_v1/release/PUBLIC_STAGE_LIST.txt > "$EXPECTED"
git diff --cached --name-only | sort > "$ACTUAL"
if ! diff -u "$EXPECTED" "$ACTUAL"; then
  echo "ERROR: staged files do not exactly match PUBLIC_STAGE_LIST.txt" >&2
  exit 1
fi
if git diff --cached --name-only | grep -Fxf final_analysis/focused_validation_v1/release/LOCAL_PROVENANCE_ONLY_LIST.txt >/dev/null; then
  echo "ERROR: local-provenance-only file staged" >&2
  exit 1
fi
if git diff --cached --name-only | grep -Fxf final_analysis/focused_validation_v1/release/DO_NOT_COMMIT_LIST.txt >/dev/null; then
  echo "ERROR: do-not-commit file staged" >&2
  exit 1
fi
SLASH="/"
PRIVATE_PATH_REGEX="${SLASH}Users${SLASH}|${SLASH}private${SLASH}|${SLASH}mnt${SLASH}data${SLASH}|bhaktidoshi${SLASH}Downloads"
if git diff --cached --name-only | xargs grep -Il . | xargs grep -nE "$PRIVATE_PATH_REGEX" >"$SCAN" 2>/dev/null; then
  cat "$SCAN" >&2
  echo "ERROR: staged text contains private path markers" >&2
  exit 1
fi
echo "Verified staged public release: $(wc -l < "$ACTUAL" | tr -d ' ') files."
