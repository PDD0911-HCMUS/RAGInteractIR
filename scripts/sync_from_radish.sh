#!/usr/bin/env bash
set -euo pipefail

SRC="${RAIR_RADISH_REPO:-/radish/phamd/duypd-proj/RAGInteractIR}"
DST="${RAIR_LOCAL_REPO:-/home/duypd/ThisPC-DuyPC/RAGInteractIR}"

if [ ! -d "$SRC" ]; then
  echo "Source directory does not exist: $SRC" >&2
  exit 1
fi

if [ ! -d "$DST/.git" ]; then
  echo "Destination does not look like the Git repo: $DST" >&2
  exit 1
fi

rsync -rv --inplace --no-times --no-perms --no-owner --no-group --omit-dir-times \
  --exclude='.git' \
  --exclude='.devcontainer/devcontainer.env' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='OpenAI.key' \
  --exclude='huggingface.key' \
  --exclude='keys.txt' \
  --exclude='e1_*.json' \
  --exclude='e2_*.json' \
  --exclude='e3_*.json' \
  "$SRC/" "$DST/"

echo "Synced radish workspace back to local Git repo:"
echo "  $SRC -> $DST"
echo "Run 'git status' in $DST before committing."
