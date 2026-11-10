#!/bin/zsh
# A11 validation batch run — single heavy process at a time.
set -e
export ECHOGESTURE_MODEL=landmark
export PYTHONDONTWRITEBYTECODE=1
cd /tmp/echogesture-agents/A11
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python evaluation/device/benchmark_lab/matrix_runner.py \
  /tmp/echogesture-validation-30/manifest.csv \
  --output-dir evaluation/device/benchmark_lab/results \
  --repeats 1 --warmup 1 \
  --same-clip-id "6523145816470133-WHAT FOR" \
  --split validation --dataset "ASL Citizen (30-clip validation batch)"
