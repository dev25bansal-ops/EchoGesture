# Reproducing every number

Run from the repository root with the project interpreter.

## Model identity

```bash
shasum -a 256 models/landmark_ens_*_fp16.tflite models/hand_landmarker.task
```

Expected: `112279bc…`, `01a261d8…`, `27058a85…`, `08417e72…` for the members
and `fbc2a300…` for the detector.

## Full test suite

```bash
python3 -I scripts/run_all_tests.py --quiet
```

Expected: 572 run, 568 pass, 0 fail, 0 import-blocked, 4 skipped. The 4 skips are
tests that need weights which are not redistributable and therefore absent in a
source-only clone; with weights present they also run.

`unittest discover` is not used: 39 of the 59 test files live outside packages,
so discovery silently ran about half the suite.

## Accuracy, both denominators

```bash
python3 -I evaluation/device/ensemble_evidence/recompute.py
```

909/1249 scored = 72.78%; 72.43% against all 1,255 eligible clips, because six
produced no hand and therefore no prediction.

## Release bundle

```bash
python3 -I -c "
import sys; sys.path.insert(0,'scripts/release')
import build_bundle as bb
bb.build('.', 'models/landmark_ens_rob71_fp16.tflite', 'models/landmark_labels.json',
         'results/results.csv', '/tmp/bundle.zip',
         extra=['scripts/ml/export_tflite_tf.py'])"
```

Then extract to a fresh directory and predict:

```bash
unzip -q /tmp/bundle.zip -d /tmp/bundle
cd /tmp && python3 -I -c "
import sys, json
sys.path.insert(0,'/tmp/bundle/src'); sys.path.insert(0,'/tmp/bundle')
from main import predict
print(json.dumps(predict('/path/to/clip.mp4')))"
```

Expected on the reference clip: `WHATFOR1`, confidence `0.8821`.

## Predictions schema check

```bash
python3 -I -c "
import json
rows=[json.loads(l) for l in open('results/predictions.jsonl') if l.strip()]
assert len(rows)==1249
assert all(set(r)=={'clip_id','prediction','top5','confidence','latency_ms'} for r in rows)
assert all(r['prediction'].isupper() and len(r['top5'])==5 for r in rows)
print('ok', len(rows))"
```

## Latency

Full clip-to-JSON, 1 cold + 12 measured warm, three configurations:

```bash
python3 -I evaluation/device/performance_final/benchmark_ensemble.py
```

Reported p95: 274.2 ms singleton, 154.8 ms parallel-2, 99.8 ms parallel-4.

## What cannot be reproduced here

On-device latency, power draw and 10-minute thermal drift need a physical Android
phone with `adb`. They are reported as NOT MEASURED rather than estimated.
