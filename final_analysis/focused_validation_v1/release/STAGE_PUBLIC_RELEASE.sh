#!/usr/bin/env bash
set -eu
ROOT="${1:-$(git rev-parse --show-toplevel)}"
cd "$ROOT"
test -d .git
test -f final_analysis/focused_validation_v1/release/PUBLIC_STAGE_LIST.txt
while IFS= read -r path; do
  [ -n "$path" ] || continue
  test -f "$path"
  git add -- "$path"
done < final_analysis/focused_validation_v1/release/PUBLIC_STAGE_LIST.txt
if git diff --cached --name-only | grep -Fxf final_analysis/focused_validation_v1/release/LOCAL_PROVENANCE_ONLY_LIST.txt >/dev/null; then
  echo "ERROR: local-provenance-only file staged" >&2
  exit 1
fi
if git diff --cached --name-only | grep -Fxf final_analysis/focused_validation_v1/release/DO_NOT_COMMIT_LIST.txt >/dev/null; then
  echo "ERROR: do-not-commit file staged" >&2
  exit 1
fi
echo "Staged $(git diff --cached --name-only | wc -l | tr -d ' ') files."
