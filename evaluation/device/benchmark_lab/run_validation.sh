#!/bin/zsh
# A11 validation batch run — single heavy process at a time.
set -e
export ECHOGESTURE_MODEL=landmark
export PYTHONDONTWRITEBYTECODE=1
cd $TMP/echogesture-.../A11
$REPO/.venv/bin/python evaluation/device/benchmark_lab/matrix_runner.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir evaluation/device/benchmark_lab/results \
  --repeats 1 --warmup 1 \
  --same-clip-id "6523145816470133-WHAT FOR" \
  --split validation --dataset "ASL Citizen (30-clip validation batch)"
