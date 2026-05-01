#!/usr/bin/env bash
set -euo pipefail

BATCH="${1:-300}"           # files per commit
THRESHOLD_MB="${2:-100}"    # size threshold in MiB
USE_LFS="${3:-1}"           # 1 = use Git LFS for large files, 0 = skip large files

echo "==> Batch size: $BATCH files/commit"
echo "==> Threshold: > ${THRESHOLD_MB}MB"
echo "==> Use LFS: $USE_LFS (1=yes, 0=no/skip large files)"

command -v git >/dev/null 2>&1 || { echo "ERROR: git not found."; exit 1; }

# Ensure git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "==> Not a git repo. Initializing..."
  git init
  git branch -M main || true
fi

# Collect large files (for LFS track OR skipping)
declare -A LARGE=()
large_count=0
while IFS= read -r -d '' f; do
  ff="${f#./}"
  LARGE["$ff"]=1
  large_count=$((large_count+1))
done < <(find . -path "./.git" -prune -o -type f -size +"${THRESHOLD_MB}"M -print0)

if [ "$USE_LFS" = "1" ]; then
  # Require Git LFS
  if ! git lfs version >/dev/null 2>&1; then
    echo "ERROR: Git LFS is not installed. Install Git LFS or run with USE_LFS=0 to skip large files."
    exit 1
  fi

  echo "==> Initializing Git LFS..."
  git lfs install >/dev/null

  if [ "$large_count" -gt 0 ]; then
    echo "==> Tracking ${large_count} file(s) > ${THRESHOLD_MB}MB with LFS (by file path)..."
    for p in "${!LARGE[@]}"; do
      echo "   LFS track: $p"
      git lfs track "$p" >/dev/null
    done

    # Commit .gitattributes if changed
    git add .gitattributes || true
    if ! git diff --cached --quiet; then
      echo "==> Committing .gitattributes (LFS rules)..."
      git commit -m "Configure Git LFS for large files" || true
    fi
  else
    echo "==> No files exceed threshold. Skipping LFS tracking."
  fi
else
  if [ "$large_count" -gt 0 ]; then
    echo "==> NOTE: USE_LFS=0, will SKIP ${large_count} large file(s) > ${THRESHOLD_MB}MB:"
    for p in "${!LARGE[@]}"; do
      echo "   SKIP: $p"
    done
    echo "==> Tip: Run with USE_LFS=1 to push these via Git LFS."
  else
    echo "==> USE_LFS=0, but no files exceed threshold."
  fi
fi

# Collect ALL changes (A/M/D/R/C/untracked) and commit in batches
echo "==> Collecting all changes..."
changes=()

while IFS= read -r -d '' rec; do
  xy="${rec:0:2}"
  path="${rec:3}"

  # Skip large file paths if LFS disabled (only when file exists locally)
  if [ "$USE_LFS" != "1" ] && [ -n "${LARGE["$path"]+x}" ]; then
    continue
  fi
  changes+=("$path")

  # Rename/Copy: next token is second path
  if [[ "${xy:0:1}" == "R" || "${xy:0:1}" == "C" ]]; then
    IFS= read -r -d '' path2 || true
    if [ "$USE_LFS" != "1" ] && [ -n "${LARGE["$path2"]+x}" ]; then
      continue
    fi
    changes+=("$path2")
  fi
done < <(git status --porcelain -z)

if [ "${#changes[@]}" -eq 0 ]; then
  echo "==> No changes to commit (after filtering)."
else
  echo "==> Committing all changes in batches..."
  batch=1
  i=0
  total="${#changes[@]}"

  while [ $i -lt $total ]; do
    end=$((i + BATCH))
    if [ $end -gt $total ]; then end=$total; fi

    chunk=( "${changes[@]:i:end-i}" )

    # Stage this chunk (includes deletions via -A)
    git add -A -- "${chunk[@]}"

    if ! git diff --cached --quiet; then
      echo "==> Commit batch $batch ($((end-i)) paths)..."
      git commit -m "Add/Update files batch $batch" || true
      batch=$((batch+1))
    fi
    i=$end
  done
fi

# Push
branch="$(git rev-parse --abbrev-ref HEAD)"
if git remote get-url origin >/dev/null 2>&1; then
  if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
    echo "==> Pushing to existing upstream..."
    git push
  else
    echo "==> Pushing and setting upstream: origin/$branch"
    git push -u origin "$branch"
  fi
  echo "==> Done."
else
  echo "==> NOTE: No remote named 'origin' found."
  echo "   git remote add origin https://github.com/<user>/<repo>.git"
  echo "   then: git push -u origin $branch"
fi

chmod +x push_all_with_lfs_batches.sh
