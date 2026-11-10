# v8_release verification — provisional bundle, isolated copy only

All work done in `/tmp/echogesture-v8-agents/release` (isolated copy,
read-only w.r.t. original checkout). Original checkout and Dev model code
were not edited. This ZIP is **provisional only**.

## Artifacts (verified hashes)

| Artifact | SHA-256 | Bytes |
|---|---|---|
| `models/landmark_classifier_fp16.tflite` | `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` ✅ matches required | 832,888 |
| `models/landmark_classifier_int8.tflite` | `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b` (unselected, not shipped) | 509,720 |
| `models/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | 7,819,105 |
| `models/landmark_labels.json` | `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a` | 1,492 |
| `models/landmark_model_meta.json` | `7f189a16f4d175042b5481afb973217fc21410832ebd14c5bacf2f516473947e` | 123 |
| `src/features.py` | `baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84` | 8,312 |
| `src/landmark_frontend.py` | `c73ef64b72f21a86638c4a126ad642b64c4ee29851cd2f8fd8ea4ed516fa97b0` | 6,633 |
| `src/main.py` | `c695d78b1b1980e3b95b5357e0549385638eda3274a76eba479c6f3dcddf8141` | 10,154 |
| `src/requirements.txt` | `bc7c428f11d4448681454c2ef91382861265de019756bc81810abfb94acdcd64` | 74 |
| `src/README.md` | `aacabcca08fd375332d1a3749cae6ccd413287314c3830f0d53ebf4a03daed28` | 4,826 |
| `scripts/ml/export_tflite_tf.py` | `8abdb6368748babdbf3a74351fe15b8fb8da3348de60dd1a387da51364e96e5f` | 5,582 |

Dependencies (`src/requirements.txt`): `numpy>=1.24.0`, `opencv-python>=4.8.0`,
`ai-edge-litert>=2.3.0`, `mediapipe>=1.2.2`. Present in `.venv`: numpy 2.4.6,
ai-edge-litert ok, mediapipe 1.2.2. No installs performed.

`landmark_model_meta.json`: `{frames:24, feature_dim:255, classes:100, dim:192, backbone:conv, quantization:float16}`.
`landmark_labels.json`: 100 labels, indices 0–99 contiguous (`DOG1:0` … `LONGLINE:99`).

Total `models/` bytes = 10,970,063 < 20,000,000 (builder enforces `< 20 MB`;
shipped bundle models total = 832,888 + 7,819,105 + 1,492 + 123 = 8,653,608 bytes).
Clean ZIP member safety: 13 members, all `0o100644`, no symlinks/devices,
no `..`/absolute/backslash, no duplicates. `verify_zip()` rejects `../evil.py`
with `Unsafe bundle path: ../evil.py` (tested).

## Build / verify commands (actual)

```bash
cd /tmp/echogesture-v8-agents/release
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I -c "<hash all artifacts>"  # FP16 hash matched

/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I scripts/release/build_bundle.py build \
  --source . \
  --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json \
  --results scripts/release/release_lab/results_NOT_MEASURED.csv \
  --output scripts/release/v8_release/provisional.zip \
  --extra models/landmark_classifier_fp16.tflite \
  --extra models/landmark_labels.json \
  --extra models/landmark_model_meta.json \
  --extra models/hand_landmarker.task \
  --extra scripts/ml/export_tflite_tf.py \
  --extra src/features.py \
  --extra src/landmark_frontend.py
# -> {"model_sha256":"a012f2b5...7044a4","status":"structure_verified"} exit 0

/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I scripts/release/build_bundle.py verify \
  scripts/release/v8_release/provisional.zip \
  --model-sha a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4
# -> {"model_sha256":"a012f2b5...7044a4","status":"structure_verified"} exit 0

ECHOGESTURE_MODEL=landmark /Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I scripts/release/build_bundle.py verify \
  scripts/release/v8_release/provisional.zip \
  --clip /tmp/v8work/DOG.mp4 \
  --model-sha a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4
# -> {"clip_id":"DOG","prediction":"ALLOFSUDDEN","top5":["ALLOFSUDDEN","DOG1","AXE1","BELT1","INTUITIVE"],"confidence":0.0372,"latency_ms":491.22} exit 0
```

ZIP: `scripts/release/v8_release/provisional.zip`
sha256 `76f84c6432387c927f878523ddea3637f5217835a0be0d00320e2b77f34ddb5b`,
6,607,661 bytes compressed / 8,690,956 uncompressed, 13 members:
Dockerfile, release.json, results/results.csv,
scripts/ml/export_tflite_tf.py, models/{hand_landmarker.task,
landmark_classifier_fp16.tflite, landmark_labels.json, landmark_model_meta.json},
src/{main.py, requirements.txt, README.md, features.py, landmark_frontend.py}.

## Extracted-tree inference (real permitted clip)

Extracted ZIP to `/tmp/v8extract` (plain `extractall` for test tree after
`verify_zip` had already enforced safe paths; no traversal).
Ran from the extracted tree on one real clip (ASL Citizen validation clip,
already on disk, research-use, not organizer data):

```
cp /tmp/echogesture-v8-agents/release/evaluation/device/v5_accuracy/data/val_346/clips/5579456931257181-DOG.mp4 /tmp/v8work/DOG.mp4
ECHOGESTURE_MODEL=landmark /Users/.../.venv/bin/python -I -c "
import sys; sys.path.insert(0,'/tmp/v8extract/src');
import main, json; print(json.dumps(main.predict('/tmp/v8work/DOG.mp4'), indent=2))"
# -> {"clip_id":"DOG","prediction":"ALLOFSUDDEN","top5":["ALLOFSUDDEN","DOG1","AXE1","BELT1","INTUITIVE"],"confidence":0.0372,"latency_ms":481.54}
# SCHEMA OK: keys {clip_id,prediction,top5(5),confidence,latency_ms}, clip_id==stem, 0<=conf<=1
```

Deterministic-ish: confidence identical; latency varies run-to-run (measured,
not claimed as a target). Note: top-1 is ALLOFSUDDEN, not DOG — single-clip
smoke only, no accuracy claim. `src/main.py` verified from both the source
tree and the extracted tree with identical output.

## Offline behavior

- No `urllib`/`requests`/`socket` imports in `src/*.py`. The only `https`
  string in the bundle is a download-URL docstring in `landmark_frontend.py`
  (`https://storage.googleapis.com/mediapipe-models/hand_landmarker/...`),
  used only if the `.task` file is missing — the bundle ships the `.task`,
  so `get_landmarker()` never fetches. Inference itself is fully local:
  OpenCV decode + MediaPipe HandLandmarker (local `.task`) + TFLite (local).
- No network was used during build/verify/inference (all commands local,
  `-I` isolated, existing `.venv`, no installs).

## HUD / streaming handoff (read-only reuse)

- `evaluation/device/test_offline_demo.py`: 17/17 pass
  (`python -I -m unittest discover -s evaluation/device -p "test_offline_demo.py" -v`).
  Covers CaptionState, HUD render (baseline/offline markers), PredictionWorker
  (predicts once, deletes temp clip), sanitize_result, camera TCC guidance.
- `evaluation/device/test_release.py`: 17/17 pass (generate_results
  hardening + evidence_lab units).
- `evaluation/device/streaming_lab/test_streaming_lab.py`: 13/13 pass.
- Real streaming run (same DOG clip, VIDEO mode sequential):
  `python -I evaluation/device/streaming_lab/run_streaming.py /tmp/v8work/DOG.mp4`
  -> prediction DOG1 (correct top-1), conf 0.107, frame_processing p50 7.4ms /
  p95 7.8ms, camera_to_caption 305.6ms, clip_to_json 578.3ms, dropped_overflow 0.
  NOTE: different sampling (sequential VIDEO vs uniform 32-frame IMAGE in
  `predict()`) explains prediction difference vs `predict()`; not a bug.
  Streaming `clip_to_json_ms` is explicitly **not** a substitute for clip p95.
- Offline demo headless import check: `model_identity()` names
  `landmark_classifier_fp16.tflite` (832,888B, sha prefix `a012f2b5f7a1`),
  `sanitize_result` round-trips. GUI/camera not launched (no display test).

## Negative controls (all as designed)

- `--model-sha 0000...0000` -> `ERROR: Bundle model does not match requested model SHA` exit 2.
- `--official` build -> `ERROR: Official bundle requires organizer-generated predictions.jsonl` exit 2 (never fabricated).
- Malicious ZIP with `../evil.py` -> `Unsafe bundle path: ../evil.py`.

## What is NOT claimed / unresolved blockers (exact)

- **Not official**: `release.json` has `"official_submission": false`,
  `"layout": "provisional pending organizer starter kit"`. Official build
  deliberately blocked without organizer `results/predictions.jsonl` +
  bundle-root `manifest.json` with exact ordered `clip_id` match.
- **Not INT8**: selected candidate is FP16 (`float16`), not INT8.
  `landmark_classifier_int8.tflite` exists but is unselected/unverified.
- **No Android, no actual camera, no official predictions, no power/thermal,
  no license clearance**: none measured, none claimed. `results.csv` rows are
  all `NOT_MEASURED` (no smoke PASS). Device accuracy/p95, power, thermal,
  robustness require representative physical-device runs via
  `run_manifest.py` + `generate_results.py` (smoke runs cannot PASS).
- **No organizer inputs yet**: official filename/ZIP details (inner layout,
  compression, naming), organizer manifest, real predictions file,
  INT8/device target confirmation, submission form/registration/receipt.
- **Frozen by instruction**: `src/features.py`, `src/landmark_frontend.py`,
  `src/main.py` are Dev-owned and reproduced as-is; this pass made no model
  or feature changes.
