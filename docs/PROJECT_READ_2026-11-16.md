# EchoGesture — complete project read

**Read completed:** 16 Nov 2026 · **Scope:** every file in the working tree plus every
release/demo/device document on the shared, dist, and release branches.

---

## 1. What was read

| Surface | Count | How |
| --- | --- | --- |
| Working tree, all files | 30,784 | inventoried by extension and directory |
| Working tree, project text/source | **679** | 8 parallel readers + inline reads, **0 uncovered** (verified programmatically) |
| Working tree, binaries | 30,105 | characterised structurally (TFLite magic, npz schema, APK contents, ONNX/PTZ/PNG/MP4 inventories) |
| `shared/v4.4-dev-integration` docs | **86** md | dedicated reader, per-file blocks |
| `shared/...` Android app source | **145** files | dedicated reader, function-level |
| `fix/v4-4-final-release` `evaluation/device/` | **88** md/json/txt (of 250 files) | dedicated reader, per-file blocks |
| Release-branch extras | `submission_final/` (10 files), `results/v4_*`, `NOTICE`, `models/README.md`, `Dockerfile`, `scripts/run_all_tests.py` | inline |

Nothing in the working tree went unread. Everything outside it that bears on the
submission was read as well.

---

## 2. What this project is

**EchoGesture** — Samsung XR Hackathon 2026, Track 2 "Sign Language Prediction on Edge
Device" (Samsung R&D Institute India, Bangalore; XR Perception Group & PRISM).
Two-person team: **Dev** (data, features, training, model export) and **Abhijeet**
(evaluator, Android app, device evaluation, release, submission).
**Deadline 18 Nov 2026, 23:59 IST** (internal target 17 Nov).

Task: recognise 100 isolated ASL glosses from video, fully on-device, no network.

---

## 3. Architecture (the shipped system)

```
video (640×480 native, or 256×256 squash in the Python reference)
  → OpenCV decode, 32 frames sampled evenly across the whole clip (np.unique linspace)
  → MediaPipe HandLandmarker (Tasks API), 2 hands × 21 landmarks × (x,y[,z])
  → canonical [Left, Right] slots by handedness, explicit per-frame presence flags
  → src/features.py: 255 floats/frame (x,y) or 381 (x,y,z)
       2 hands × 21 lms × 2-3 coords × 3 (pos + vel + acc) + 3 inter-hand geometry
       shared wrist→middle-MCP scale, root-centred per hand, inter-hand measured
       BEFORE centring, per-sequence normalisation
  → 9 × FP16 TFLite Conv1D ensembles, mean-logit aggregation, one softmax
       TemporalModel: stem 1×1 conv → BatchNorm → 3× ConvBlock(k=17) or
       4× dilated TCN (1,2,4,8) → attention pooling (learnable query +
       key_padding_mask) → linear head.  dim=192, ~396K–469K params/model.
  → top-5 glosses + confidence + margin
```

**Ensemble composition (shipped):** `tcn21`, `tcnm61`, `tcnm62`, `rob71` (255-wide x,y)
+ `mirror_handdrop_tcn_z_s0/s1/s2/s7/s13` (381-wide x,y,z).
Combined SHA-256 in meta order: `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`.

**Android app** (`org.echogesture.finalapp`, minSdk 24 / target 34, CAMERA-only —
INTERNET/ACCESS_NETWORK_STATE explicitly stripped from the merged manifest):
CameraX `ImageAnalysis.setTargetResolution(640,480)`, `KEEP_ONLY_LATEST` →
`ImageProxy.toBitmap()` (rotation via `ImageProcessingOptions`, never baked) →
`detectForVideo` (VIDEO mode, ts+33/frame) → 3.7 s rolling window ending at the tap,
resampled to 32 real frames by timestamp → `FeatureTransform.build` in both widths →
per-member TFLite interpreters resolved by tensor NAME (input 0 is the mask) →
mean logits → one softmax → top-5, accepted if margin ≥ 0.15 and score ≥ 20 %.
`CAPTURE_SPAN_MS = 3700` is load-bearing: features.py velocity has no dt, so the window
must be real time resampled (32 consecutive 33 ms frames would shrink velocity ~3.5×
and acceleration ~12×).

---

## 4. The measured numbers (verified against artifacts)

I independently recomputed the two headline accuracies by joining the shipped
prediction files against the gloss stored in each `.npz`:

| | artifact | recomputed | claimed |
| --- | --- | --- | --- |
| test top-1 | `results/predictions.jsonl` (1,251 rows) vs native cache | **1,069/1,251 = 85.45 %** | 85.45 % ✓ |
| val top-1 | `results/predictions_val.jsonl` (346 rows) | **309/346 = 89.31 %** | 89.31 % ✓ |

**Current state (9-model FP16, native 640×480 landmark cache):**

| metric | value | verdict |
| --- | --: | --- |
| top-1 test (11 unseen signers, 1,251 clips, 100 classes) | **85.45 %** | PASS (target ≥ 70) |
| top-5 | 96.64 % | — |
| top-1 val (5 unseen signers, 346 clips) | 89.31 % | PASS |
| per-signer | mean 86.02 / min 72.54 / max 99.01 | — |
| best single member | 81.53 % (ensemble adds +3.92) | — |
| model size | 8.51 MB ensemble + 7.82 MB detector = ~16.3 MB | PASS (< 20) |
| quantization | FP16, 0.00 pt vs FP32; INT8 rejected −6.37 pt | PASS |
| classifier latency (9 members, desktop x86) | 4.2 ms/clip | — |
| **per-frame latency, desktop x86** | **p50 133 ms / floor 69 ms / p95 199 ms** | **FAIL** (budget 50) |
| end-to-end warm `predict()` | p50 4,650 ms | — |
| cold start | ~159 s (MediaPipe lazy graph build, now warmed in `_init_model`) | — |
| **robustness, pixel (full split)** | **+4.48** (blur +1.12, low_light +4.48, downscale +1.92, jpeg +2.88) | **PASS** |
| **robustness, landmark (full split)** | **+5.76** (drop_hand_25pct only; 20 % passes at +3.04, 35 % collapses to +10.87) | **FAIL** |
| confidence calibration | ≥ 0.8 → 99.7 % (594 clips); abstain < 0.5 → 95.93 % at 72.6 % coverage | best demo claim |
| **on-device latency / power / thermal** | **NOT MEASURED — no phone ever attached** | 20 % of rubric |

---

## 5. The four findings that define this project

### 5.1 The published pixel model collapses on unseen signers (1.47 %)

The MIT-licensed S3D-Conv1D INT8 weights score **1.47 %** (PTQ, 25/1,705) and **1.00 %**
(QAT) on ASL Citizen's participant-disjoint split — against a 1.02 % chance floor. A
positive control on in-distribution WLASL clips scored **17/17 = 100 %**, proving the
harness correct. Cause: raw 64×64 pixels let the network memorise signer appearance and
background; the paper's own split is 70/20/10 stratified random. This is why the
project uses landmarks.

### 5.2 The attention mask was trained but never exported (+4.24 pt)

`train.TemporalModel.forward` applies `key_padding_mask`; `tf_model.build_tf_model`
declared only `features`. **Every TFLite artifact ever shipped was a different model
than the one trained.** The export's parity gate could not catch it — both sides lacked
the mask and agreed perfectly, so the gate validated the port against PyTorch-*without*
-mask. "Parity is not correctness." Fixing it moved a single member +2.90 val/+1.84
test and the ensemble 72.78 → 77.02.

### 5.3 The evaluation cache was wrong about presence on 29 % of test clips (+8.5 pt)

The 8 Nov landmark cache held a detection *open across the back half* of the clip in
364 of 1,255 test clips (train 0/138, test 29 %). On the 885 clips where cache and
fresh extraction agree on presence, both score **82.49 % to two decimals** — so the
feature pipeline is innocent and the whole 5.5-point gap is the mask. Re-extracting at
the app's real geometry (native 640×480) moved the shipped 9 models from **76.94 % to
85.45 %**. Cause never recovered (five hypotheses disproved; the cache predates the
committed extraction code). A retrain at app geometry *lost* (76.26 %, below a single
member's 78.90 %) — the gain comes from re-feeding the existing models, not retraining.

### 5.4 Four latency figures had no evidence and were 6–16× optimistic

`results.csv` claimed 32.80 ms p95 per-frame PASS, justified by "MediaPipe 15.7 ms/frame"
— a figure in six files with **no artifact anywhere** producing it. Measured properly:
p50 133 ms, floor 69 ms. Cold start 8,800 ms → actually ~159,000 ms; end-to-end 470 ms
→ actually 4,650 ms. The pattern: accuracy numbers were verified repeatedly and held;
latency numbers had never been checked from source and every one was optimistic.

---

## 6. Documentation and branch state — the repo is three projects in one

This is the most important structural finding, and it is invisible from any single branch.

| Track | Where | Model | Accuracy | Latency basis |
| --- | --- | --- | --- | --- |
| **Dev's model work** | `hackathon/dev-model` (this tree) | **9-member masked**, native cache | **85.45 %** | desktop x86, per-frame FAIL |
| **Abhijeet's release pack** | `fix/v4-4-final-release`, `shared/v4.4-dev-integration`, `dist/*` | **4-member**, frozen 11 Nov | **72.78 %** | macOS arm64, p95 99.8 ms best |
| **V7/V8 device evidence** | release branches, `evaluation/device/` | 4-member + single classifier | **78.61 % val** | macOS, parallel-4 79.8 ms p50 |

**The number 85.45 % appears in none of the 86 release docs.** The release docs freeze
72.78 % (test, four members) and 78.61 % (val, V7/V8). The app *code* was migrated to the
9-member bundle (V4.3, parity 9/9) but the *documents* never were.

### Stale-artifact inventory (verified by hashing)

| Document | Says | Reality |
| --- | --- | --- |
| `src/README.md` (evaluator-facing) | "All four files in `models/` are required", 4-member table, 72.78 % | nine members ship, 85.45 % |
| `DEV_HANDOFF_MODEL.md` | 4 old member hashes (`112279bc…`, `01a261d8…`, `27058a85…`, `08417e72…`) | actual are `af031d89…`, `b0a97aa4…`, `1d247397…`, `105218a8…` — **would fail verification** |
| `submission_final/RELEASE_MANIFEST.json` | 4 members, 939,152/832,888 bytes | actual 942,160/835,896 |
| `results/QUANTIZATION.md` | "ship `landmark_classifier_fp16.tflite`" | that file no longer exists |
| `README.md` layout, `TEAM_GUIDE.md` layout | `ptq_100_int8.tflite`, `s3d_conv1d_qat_int8.tflite` | removed from models/ |
| `TEAM_GUIDE.md` §0 | "1.47 % accuracy, 117.8 ms p95, submission not ready" | not marked stale |
| app `strings.xml:132` | "Four-model ensemble" | nine ship |
| app README + all parity fixtures | 4-member logits | tests acknowledge the fixtures are legacy |

`src/main.py::predict()` calls `clip_to_landmarks(video_path, max_frames=...)` with **no
`size` argument**, so it defaults to `FRAME_SIZE = 256` squash. Verified empirically:
default path yields `(256,256,3)` frames, native yields `(480,640,3)`. So the
organizer's runnable entry point measures the **~82.5 % squash operating point**, not
the 85.45 % native figure. `_preprocess_video(size="native")` supports the good path;
`predict()` does not use it.

### `results/results.csv` has a wrong verdict in the `pass` column

Parsed with the `csv` module: 8 columns, no ragged rows — but four rows have their
`runtime` and `pass` values shifted. The dangerous one:

| metric | measured | target | `runtime` | `pass` |
| --- | --: | --- | --- | --- |
| `latency_per_frame_p95` | **199.4 ms** | ≤ 50 | "FAIL (retracted claim replaced)" | **"pass"** |

The organizers' p.11 flow explicitly invites a script to read that `pass` column. It
would report a PASS on a measurement that is 4× over budget. The same shift puts
runtime text in the `pass` column of `top1_accuracy`, `accuracy_drop_robustness` and
`accuracy_drop_robustness_pixel`, and doubles the quotes on the pixel row's device
field. This is the same class of defect the repo fixed once in the four-member era; it
has regressed in the current rows.

Two smaller code findings: `scripts/ml/export_tflite_tf.py`'s INT8 accuracy gate is a
no-op (`drop = 0.0` hardcoded, so the refuse-on-regression check can never fire — the
shipped FP16 path is unaffected because it is parity-gated), and
`results/predictions_test.jsonl` is a stale four-member-era file (constant 1.21 ms
latency, different confidences) sitting beside the current `predictions.jsonl` under an
identical schema.

---

## 7. What is genuinely good

This is an unusually rigorous engineering effort, and that deserves emphasis.

- **Every claim names a script.** `RESULTS.md`, `EXPERIMENTS.md`, `CLAIMS_AND_LIMITS.md`
  separate what is measured from what is inferred, and mark superseded numbers.
- **Negative results are recorded with their numbers.** `EXPERIMENTS.md` documents
  every rejected idea: merged val data (+0.24 = noise), speed perturbation (−5.45),
  strict resample (−8), mirror TTA (−1.01), INT8 (−6.37), wider model (−15), 32 frames
  (−8), mean-prob aggregation (0), larger ensembles (negative). Nothing was quietly
  dropped.
- **Errors were found and corrected in public.** `AUDIT_2026-11-15.md` records 12 real
  errors (a significance test attributed to the wrong measurement, a wrong claim about
  which configuration beat which, an APK hash tail typo, a McNemar p > 1 bug).
  `LANDMARK_CACHE_DISCREPANCY.md` records a bug report that was *nearly filed falsely*
  and the correction. `HARNESS_BUGS` records `--limit 0` silently meaning 120.
- **The rights story is honest.** `NOTICE`, `models/README.md`, `ARTIFACT_RIGHTS.md`,
  `PUBLIC_WEIGHT_EXPOSURE.md` all say the weights are derived from ASL Citizen (MSR
  non-commercial) and WLASL (C-UDA) and are not redistributable.
- **The demo claim is real.** Confidence is calibrated: above 0.8 → 99.7 % over 594
  clips; abstaining below 0.5 answers 72.6 % at 95.93 %. That turns a fluent wrong
  answer into an honest "sign that again" — the strongest thing to show a judge.
- **Parity is verified to 0.000000** on feature transform, exact on masks, 5e-6 on
  logits, 9/9 top-1 between Android and Python.

---

## 8. The open problems, ranked

1. **No phone measurement — 20 % of the rubric.** Not a code problem. Every route is
   documented and blocked: emulator AAR has no x86_64 slice; the app's own tests record
   that TFLite cannot run on the JVM; the phone disconnected mid-install. The APK is
   verified (`3db9d94a…1380896`) and the runbook exists
   (`ON_DEVICE_MEASUREMENT_OPTIONS.md`). This needs hardware and a person.
2. **`results.csv` reports a PASS on a failing latency row** (§6). One-line fix, and
   it is the artefact the judges read. Do this first.
3. **Demo video never recorded.** An explicit deliverable. The 2:50 script, shot list,
   and recording guide all exist; `DEMO_SCENE_SEQUENCE.md` says `NOT_RECORDED`.
4. **The deck exists but is unpersonalised.** `docs/submission/EchoGesture_Submission.pptx`
   (11 slides) awaits team/college strings and the required filename.
5. **Rights unresolved.** Weights sit in a public repo on `hackathon/dev-model`; two
   WLASL-derived research INT8 files are on every branch including `main`; a 1,705-row
   CSV with 11 raw participant IDs is tracked. No visibility change or history rewrite
   has been done. Abhijeet's standing instruction is "avoid further redistribution."
6. **The repo's documentation contradicts its own artifacts** (§6). This is fixable in
   an afternoon and is the difference between a judge verifying the claims and a judge
   catching a mismatch.
7. **`drop_hand_25pct` fails by 0.76 pt** and is a genuine model limitation — the mask
   works, the information is missing.
8. **Per-frame latency fails on desktop** (133 ms vs 50 ms) and the phone path
   (`detectForVideo` with tracker reuse) is unmeasured. Note that the two teams measured
   per-frame latency 30× apart (133 ms Windows x86 IMAGE/VIDEO vs 4.95 ms macOS M5 Pro
   VIDEO) — both are real, different machines and different MediaPipe builds.
9. **Organizer test manifest never arrived.** No starter kit, no vocabulary, no manifest
   in any branch. Predictions use the project's own ASL Citizen split, `clip_id` = video
   stem. `DEV_HANDOFF_MODEL.md` open question 1 asks whether that matches.

---

## 9. Corrections to my own readers

One reader claimed Dev's "133 ms per-frame" was a metric-labelling error (a per-clip p50
coincidentally matching parallel_2's 133.19 ms). **That is wrong.** I read
`scripts/ml/latency_per_frame.py`: it times `detect_hands([f], mode=mode)` — a
one-element frame list — so it genuinely measures per-frame cost, and it records both
IMAGE (150.9 ms p50) and VIDEO (**141.9 ms p50**) modes. Tracker reuse gave this
Windows/x86 MediaPipe build almost nothing. Both teams' per-frame numbers are valid
measurements at very different operating points.

`src/features.py`'s docstring quotes "72.78 → 67.49 if resample changes" — that is the
4-member era; the current figure is 85.45 %.

`results/eval_summary_*.json` (currently modified, uncommitted) were regenerated for the
9-member native era but the diff **dropped every accuracy field**; they now carry latency
and metadata only. `tflite_native_test.json` has the accuracy; the summary writer omits it.

---

## 10. Reproducing anything in this report

```bash
# headline accuracy from the shipped artifacts
python scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native            # test
python scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native --split val

# robustness
python scripts/ml/robustness.py --level landmark --cache .scratch/landmarks_native
python scripts/ml/robustness_pixel_parallel.py --workers 6 --size 512 --letterbox \
       --cache .scratch/landmarks_native

# latency
python scripts/ml/latency_per_frame.py

# hashes
python .scratch/verify_hashes.py            # 995cc8d0… in meta order
```

A working log of every reader result, every contradiction found, and every correction
lives in `.scratch/full_read_progress.md`.
