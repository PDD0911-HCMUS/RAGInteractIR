#!/usr/bin/env bash
set -euo pipefail

SRC="${RAIR_LOCAL_REPO:-/home/duypd/ThisPC-DuyPC/RAGInteractIR}"
DST="${RAIR_RADISH_REPO:-/radish/phamd/duypd-proj/RAGInteractIR}"

if [ ! -d "$SRC" ]; then
  echo "Source directory does not exist: $SRC" >&2
  exit 1
fi

mkdir -p "$DST"

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

echo "Synced local repo to radish:"
echo "  $SRC -> $DST"
