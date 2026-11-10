"""Agent D verification commands (run from isolated copy root).

Run with the Python 3.11 venv that already has cv2 / numpy / ai_edge_litert /
mediapipe installed. Paths use the isolated copy root variable ROOT.
"""

ROOT=/tmp/echogesture-v5-agents/D
PY="$REPO/.venv/bin/python"
MANIFEST=/tmp/echogesture-validation-30/manifest.csv

# --- Paired FP16 vs INT8 robustness (identical corruptions, seed 42, severity 0.5) ---
for KIND in low_light crop occlusion motion_blur frame_drop; do
  "$PY" -I "$ROOT/evaluation/device/robustness.py" "$MANIFEST" \
    --output-dir "$ROOT/evaluation/device/v5_release/paired/$KIND/fp16" \
    --predictor "$ROOT/src/main.py" \
    --labels "$ROOT/models/landmark_labels.json" \
    --model "$ROOT/models/landmark_classifier_fp16.tflite" \
    --corruption "$KIND" --severity 0.5 --seed 42 --validation --split validation

  "$PY" -I "$ROOT/evaluation/device/robustness.py" "$MANIFEST" \
    --output-dir "$ROOT/evaluation/device/v5_release/paired/$KIND/int8" \
    --predictor "$ROOT/scripts/release/v5_release/int8_predictor.py" \
    --labels "$ROOT/models/landmark_labels.json" \
    --model "$ROOT/models/landmark_classifier_int8.tflite" \
    --corruption "$KIND" --severity 0.5 --seed 42 --validation --split validation
done

# --- Provisional bundle build + verify (FP16 selected, default unchanged) ---
cd "$ROOT"
"$PY" -I scripts/release/build_bundle.py build \
  --source . \
  --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json \
  --results scripts/release/release_lab/results_NOT_MEASURED.csv \
  --output scripts/release/v5_release/provisional.zip \
  --extra models/landmark_classifier_fp16.tflite \
  --extra models/landmark_labels.json \
  --extra models/landmark_model_meta.json \
  --extra models/hand_landmarker.task \
  --extra scripts/ml/export_tflite_tf.py \
  --extra src/features.py \
  --extra src/landmark_frontend.py

"$PY" -I scripts/release/build_bundle.py verify scripts/release/v5_release/provisional.zip \
  --model-sha a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4

"$PY" -I scripts/release/build_bundle.py verify scripts/release/v5_release/provisional.zip \
  --clip "/tmp/echogesture-validation-30/6523145816470133-WHAT FOR.mp4"

# Official build must fail (no organizer predictions):
"$PY" -I scripts/release/build_bundle.py build --source . \
  --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json \
  --results scripts/release/release_lab/results_NOT_MEASURED.csv \
  --output /tmp/official_attempt.zip \
  --extra models/landmark_classifier_fp16.tflite \
  --extra models/landmark_labels.json \
  --extra models/landmark_model_meta.json \
  --extra models/hand_landmarker.task \
  --extra scripts/ml/export_tflite_tf.py \
  --extra src/features.py --extra src/landmark_frontend.py --official
# Expected: ERROR: Official bundle requires organizer-generated predictions.jsonl

# Clean extraction + extracted predictor smoke:
"$PY" -I -c "import zipfile; zipfile.ZipFile('scripts/release/v5_release/provisional.zip').extractall('/tmp/v5_verify_extract')"
cd /tmp/v5_verify_extract
"$PY" -I src/main.py "/tmp/echogesture-validation-30/6523145816470133-WHAT FOR.mp4"

# Release builder unit tests:
"$PY" -I -m unittest discover -s evaluation/device -p 'test_release.py'
