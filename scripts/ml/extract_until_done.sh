#!/usr/bin/env bash
# Run landmark extraction until it finishes.
#
# MediaPipe aborts the whole process on some clips (a native crash, not a
# Python exception, so it cannot be caught inside the loop). The extractor is
# resumable - it skips any .npz that already exists - so the fix is to relaunch
# until nothing is left to do.
#
# Each attempt logs separately so a crash is visible rather than silent.

set -u
cd "$(dirname "$0")/../.." || exit 1

PY=".scratch/s3d_repro/venv311/Scripts/python.exe"
SPLIT="${1:-all}"
TARGET_TRAIN=1789
TARGET_VAL=346
TARGET_TEST=1255
MAX_ATTEMPTS=40

count() { find ".scratch/landmarks/$1" -name '*.npz' 2>/dev/null | wc -l; }

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  t=$(count train); v=$(count val); s=$(count test)
  echo "[attempt $attempt] train=$t/$TARGET_TRAIN val=$v/$TARGET_VAL test=$s/$TARGET_TEST"

  if [ "$t" -ge "$TARGET_TRAIN" ] && [ "$v" -ge "$TARGET_VAL" ] && [ "$s" -ge "$TARGET_TEST" ]; then
    echo "done"
    exit 0
  fi

  log=".scratch/extract_attempt_${attempt}.log"
  "$PY" scripts/ml/extract_landmarks.py --split "$SPLIT" >> "$log" 2>&1
  rc=$?
  echo "  exit=$rc log=$log"

  # No progress between attempts means the next clip is a hard crasher; skip it
  # by writing a placeholder so the loop can move on.
  t2=$(count train)
  if [ "$rc" -ne 0 ] && [ "$t2" -eq "$t" ]; then
    echo "  no progress - marking the next clip as unextractable"
    "$PY" - <<'PYEOF'
import glob, csv, os
from pathlib import Path
root = Path('.scratch/landmarks/train')
root.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open('.scratch/eval_research/aslcit/splits/train.csv', encoding='utf-8')))
import json
vocab = set(json.load(open('.scratch/vocab_best100.json')))
rows = [r for r in rows if r['Gloss'] in vocab]
for r in rows:
    d = root / (Path(r['Video file']).stem + '.npz')
    if not d.exists():
        import numpy as np
        np.savez_compressed(d, landmarks=np.zeros((1, 2, 21, 3), np.float32),
                            present=np.zeros((1, 2), np.float32),
                            gloss=r['Gloss'], participant=r['Participant ID'],
                            split='train')
        print('  skipped', r['Video file'])
        break
PYEOF
  fi
done

echo "gave up after $MAX_ATTEMPTS attempts"
exit 1