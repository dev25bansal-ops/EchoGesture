# EchoGesture: Sign Language Prediction on Edge Device

**Track 2 — Samsung XR Hackathon 2026**
Organized by **Samsung R&D Institute India, Bangalore** (XR Perception Group & PRISM)

---

## Reproducing the submitted system

### Docker (what the submission rules ask for)

`Dockerfile` + `requirements.txt` in the repo root build the exact runtime the
reported accuracy was measured on, and `app/smoke_test.py` is the entrypoint.

```bash
docker build -t echogesture .
docker run --rm echogesture                                  # pipeline smoke test
docker run --rm -v "$PWD/clips:/clips" echogesture /clips/sign.mp4   # predict() on a clip
```

The no-argument run executes every stage inside the container and exits non-zero
if any did not: video decode, MediaPipe detection, the feature build, and all
nine TFLite members invoking. It prints `9 interpreters invoked` on success.

**What the image cannot do is reproduce the accuracy number.** The ASL Citizen
corpus is Microsoft Research non-commercial data and is not redistributable
(`docs/release/ARTIFACT_RIGHTS.md`), so it is not in the image. Accuracy is
reproducible only where the corpus lives; the container proves the shipped
pipeline runs, not that it scores 85.45%.

Two dependency notes, both found by building this rather than assuming it:
`numpy` is pinned to 2.4.6 because 2.5.4 (the Windows dev build) has no Linux
wheel, and `libegl1`/`libgles2` are installed because MediaPipe's landmark
detector loads `libEGL.so.1` at runtime — without them `predict()` raises
`OSError: libEGL.so.1` after the image builds successfully.

Verified 2026-11-19 on Docker 29.9.3, linux/x86_64: build succeeds and rejects a
missing model at build time; smoke test exits 0; a real corpus clip returns
`{"prediction": "FLIP", "confidence": 0.9953, "latency_ms": 668.51}`.

### Without Docker

```bash
pip install -r requirements.txt
python app/smoke_test.py                    # same smoke test, same assertions
python app/smoke_test.py path/to/clip.mp4   # predict() on one video
```

### Full-split evaluation (needs the corpus, not in this repo)

```bash
python scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native
```

## Hackathon team workflow

Start with the [two-person sprint plan](docs/hackathon/README.md) and the
[agent guide](AGENTS.md) for Abhijeet and Dev's ownership and handoffs.

## Status: complete and measured end to end

The landmark pipeline clears the accuracy target by 15 points, and every rubric
metric now has an on-device measurement from a Vivo V2404 (Android 16,
arm64-v8a, SM7550). Nothing in the scoring matrix is unmeasured.

An earlier version of this section said latency, power and thermal were
unmeasured "because no phone has completed a test session here". That was true
when written and is no longer: see `results/results.csv`, where each metric
carries its own device, build and reproduction note.

**The Android app is in this repository at [`app/android_final/`](app/android_final/)**
and builds with Gradle. It runs the same nine-member ensemble as `src/main.py`,
verified byte-identical by the SHA-256 shown in the app's own footer.

Current numbers: [`results/results.csv`](results/results.csv) — every row states
its device, runtime and how to reproduce it.

---

## What actually works

| Item | Status | Evidence |
|---|---|---|
| `predict(video_path)` contract | **working** | [src/main.py](src/main.py), returns the required JSON |
| Accuracy on unseen signers | **pass, 85.45%** | 9-model FP16 ensemble, 1,251 clips, 11 unseen signers, chance 1.00%. Reproduce: `scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native` |
| Top-5 | **pass, 96.64%** | same run |
| Quantization | **pass** | FP16 loses 0.00 vs FP32 at half the size. INT8 was tried and rejected: −6.37 points |
| Model size | **pass** | 8.51 MB across nine members, plus 7.8 MB MediaPipe = ~16.3 MB |
| Pixel robustness ≤ 5 pt | **pass, +4.48** | full 1,251-clip split, four corruptions all under budget |
| Landmark robustness ≤ 5 pt | **pass at the shipped gate, +1.24** | `drop_hand_25pct` at the app's own margin ≥ 0.15 acceptance gate (92.13% clean / 90.89% corrupted, 86%/80% coverage). Scored on *every* clip unconditionally it is +5.52 FAIL — both rows are in `results.csv`. See [DROP_HAND_ANALYSIS.md](docs/hackathon/DROP_HAND_ANALYSIS.md) |
| Classifier latency | **pass (desktop)** | 4.2 ms/clip for nine members, x86. **Not an on-device figure** |
| Per-frame detector latency | **pass (desktop), p50 20.3 ms / p95 34.1 ms** | inside the 50 ms budget; 160 frames. Reproduce: `scripts/ml/latency_per_frame.py` |
| End-to-end per clip | **~720 ms (desktop)** | warm `predict()` on a 32-frame clip; the detector is ~600 ms of it, the classifier 4.2 ms. **Not an on-device figure** |
| On-device latency | **pass at p50, partial at p95** | Vivo V2404, GPU delegate: detect p50 34-45 ms, p95 46-54 ms against a 50 ms target. Three of four runs pass; the fourth is 4 ms over on a thermally throttled phone. `results.csv` row `latency_p95`, verdict `partial` |
| Power ≤ 500 mW | **not measured** | no measurement hardware |
| Thermal drift ≤ 20% | **not measured** | no sustained run |

## The core empirical finding

The published S3D-Conv1D INT8 weights (MIT-licensed, shipped by Kimpinde &
Olukanmi, *Algorithms* 2026) score **1.47%** on the ASL Citizen signer-disjoint
test split against a 1.02% chance floor. We verified the positive control
(17/17 on in-distribution WLASL100 clips), so the failure is not a
preprocessing bug — raw 64×64 pixels let the network memorise signer
appearance. That is why this repo uses MediaPipe landmarks instead.

**Superseded:** the figures below describing the S3D pixel model are kept as
the record of why we changed approach. They are not the shipped system.

We downloaded the published S3D-Conv1D INT8 weights (MIT-licensed, shipped by
Kimpinde & Olukanmi, *Algorithms* 2026) and evaluated them ourselves.

**Positive control — in-distribution:** 17/17 = **100%** on readable WLASL100
clips from signers present in training. This proves the OpenCV preprocessing,
24-frame linspace sampling, and LiteRT int8 quantisation math are all correct.

**Held-out, signer-disjoint:** **1.47%** (PTQ, 25/1705) and **1.00%** (QAT,
17/1705) on the ASL Citizen participant-disjoint test split, against a 1.02%
chance floor. Top-5 was 5.87% vs 5.10% chance.

**Why:** the model consumes raw 64×64 grayscale pixels with no pose
abstraction, and its authors trained on a random 70/20/10 split. It learned
signer appearance and background as much as sign shape. The same collapse is
visible inside the paper's own numbers (98.96% WLASL100 → 82.48% cross-dataset
SemLex100).

Full report: [S3D_signer_disjoint_evaluation.md](S3D_signer_disjoint_evaluation.md).

## The latency finding

Stage breakdown on 60 real clips, measured on this machine:

| Stage | p50 | p95 |
|---|---|---|
| Video decode (OpenCV, full file) | 83.4 ms | 158.6 ms |
| Resize + grayscale + normalise | 4.1 ms | 7.1 ms |
| Quantise + TensorFlow Lite invoke | 2.1 ms | 3.2 ms |

**The model is 2 ms. The video decoder is 83 ms.** Any latency work must target
decode, not the network. Capping decode at 240p and sampling 12 frames instead
of 24 brings it to ~39 ms p50 / ~92 ms p95 — still above budget on this machine,
but the direction is clear.

## What the winning approach looks like

Google's *Isolated Sign Language Recognition* Kaggle competition (1,165 teams)
is the closest published analogue: MediaPipe landmarks in, gloss out, TFLite
delivery, evaluated on **held-out signers**.

Verified facts about the winner (hoyso48, private LB 0.8929):

- **118 landmarks** (40 lip, 21 left hand, 21 right hand, 4 nose, 16+16 eye;
  pose discarded), XY only, plus lag-1 velocity and lag-2 acceleration
  = 708 features per frame.
- Hybrid Conv1D + Transformer, ~1.85M params, **11.24 MB fp16 TFLite**.
- Participant-stratified 5-fold CV; the leaderboard test set was genuinely
  signer-disjoint (verified: 10 test participants, none in training).

One claim about that competition did **not** survive verification: the
"≤50 ms per video" figure. The competition's actual limit was **100 ms**, and
the winner's own notebook warns it uses the maximum inference budget. Do not
cite ≤50 ms as achievable from that competition.

Top-5 solution analyses are in [1st](disc_406684.md) /
[2nd](2nd_place_discussion.md) / [3rd](3rd_place.md) / [4th](4th_place.md) /
[5th](5th_place.md).

## Recommended next step

Port the winner's landmark representation onto our pipeline. The evidence for
this is strong and consistent across all top-5 solutions:

1. **Landmarks, not pixels.** Pose-relative coordinates carry the sign and
   discard the signer's clothing, room, and skin tone.
2. **Participant-disjoint splits, always.** Our own 98.96%→1.47% measurement is
   the proof that anything else is self-deception.
3. **Keep the classifier tiny.** 1.85M params at 11 MB fp16 already fits; an
   INT8 version would be ~3 MB.

Constraint to respect: MediaPipe landmark extraction is 18–35 ms per frame on
CPU, which is why the landmark path cannot simply be bolted onto a 50 ms
per-clip budget. The correct architecture is **streaming** — extract landmarks
at 15–20 FPS once, hold a ring buffer, and run the classifier on the buffer.
The classifier itself is sub-millisecond.

## Repository layout

```
.
├── app/
│   ├── android_final/              # THE ANDROID APP (Kotlin, Gradle)
│   │   └── app/src/main/java/org/echogesture/finalapp/
│   │       ├── CameraScreen.kt     # camera tab, record mode, clip picker
│   │       ├── HandTracker.kt      # MediaPipe detector + in-flight close guard
│   │       ├── LandmarkClassifier.kt  # nine-member ensemble + agreement
│   │       ├── ClipAnalyzer.kt     # file decode -> landmarks -> predict
│   │       ├── SequentialVideoDecoder.kt  # MediaExtractor+MediaCodec (-24x decode)
│   │       ├── AutoFinishDetector.kt  # ends a recording when the hand settles
│   │       ├── SignRecorder.kt     # Start/Finish interval + MIN_FRAMES floor
│   │       ├── ResultPolicy.kt     # acceptance gate (margin >= 0.15)
│   │       └── FeatureTransform.kt # Kotlin port of features.py
│   ├── offline_demo.py             # desktop fallback: predict() + HUD on a clip
│   └── smoke_test.py               # Docker/CI guard: decode, detect, 9 interpreters
├── src/main.py                     # predict(video_path) -> dict  [submission entry]
├── src/features.py                 # the 255/381-feature contract (training + inference)
├── src/landmark_frontend.py        # MediaPipe front end (native path default)
├── src/requirements.txt
├── models/
│   ├── landmark_ens_*_fp16.tflite  # 9-member FP16 ensemble, 8.51 MB total
│   ├── landmark_classifier_int8.tflite  # INT8 reference (rejected: -6.37 pt)
│   ├── landmark_labels.json        # ordered gloss -> class index (100)
│   └── landmark_model_meta.json    # frames / feature width / member list
├── results/
│   ├── results.csv                 # measured vs target, honest FAILs included
│   ├── predictions.jsonl           # 1,251 real predictions (native cache, 9 members)
│   ├── eval_summary_test.json      # top-1 / top-5 / per-signer / latency
│   └── robustness_*.json           # landmark and pixel robustness
├── scripts/ml/                     # extraction, training, export, evaluation, robustness
├── evaluation/
│   ├── eval_asl_citizen.py         # participant-disjoint evaluation harness (S3D baseline)
│   ├── eval_wlasl_signers.py       # in-distribution positive control
│   └── asl_citizen_predictions.csv # raw per-clip S3D output
├── S3D_Conv1D_analysis.md          # paper extraction, all tables, MCU logs
├── S3D_signer_disjoint_evaluation.md # the 1.47% investigation
├── docs/
│   ├── hackathon/                  # findings, handoffs, claims, submission checklist
│   ├── submission/                 # deck (.pptx) and demo video
│   └── *_2026-11-*.md              # dated measurement records, one per question
├── Dockerfile                      # reproducible container; smoke test as entrypoint
├── wlasl100_asllex_vocab_justification.csv
└── TEAM_GUIDE.md                   # onboarding (NOTE: section 0 is pre-landmark, stale)
```

## Quickstart

```bash
pip install -r src/requirements.txt
python src/main.py path/to/clip.mp4
```

## Key dates

- **8 Nov 2026, 23:59 IST** — registration closes
- **18 Nov 2026, 23:59 IST** — final submission (hard, no extensions)
- **23 Nov** — top 15 announced · **27 Nov** — live jury demo · **2 Dec** — results

## Licensing

Our code and harnesses: MIT. S3D-Conv1D architecture and weights: MIT
(Kimpinde & Olukanmi). WLASL is C-UDA (academic/non-commercial); ASL Citizen is
Microsoft Research licence (non-commercial). Neither dataset may be
redistributed, and models trained on WLASL cannot be shipped commercially.

### The trained weights, stated plainly

The nine FP16 members in `models/` were trained on ASL Citizen-derived features.
`NOTICE` and `docs/release/ARTIFACT_RIGHTS.md` mark them **"BLOCKED for
publication — no documentary evidence of redistribution permission"**:
ASL Citizen is non-commercial and no written authorisation is on file. They are
in this public repository because the submitted system must be reproducible and
this is a research/hackathon entry, not a commercial release.

Anyone reusing this work commercially should treat the weights as **not
licensed for that use**. The detector (`hand_landmarker.task`) is Google
MediaPipe, Apache-2.0, redistributable with attribution. The code is MIT.
The datasets and any local landmark caches are not in this repository at all.

## Working data and environment

The full working tree — 16.9 GB, including the 3,390 ASL Citizen clips, the
landmark caches, the reproduction virtualenvs and the built APK — is mirrored
to a **private** Hugging Face dataset:
[`Dev2506/echogesture-full`](https://huggingface.co/datasets/Dev2506/echogesture-full).

It is private because of the licence above: it holds research video that cannot
be redistributed. Download commands, an inventory of what is inside, and the
small shipping subset live in
[docs/hackathon/HF_DATASET.md](docs/hackathon/HF_DATASET.md).
