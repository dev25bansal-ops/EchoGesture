# EchoGesture: Sign Language Prediction on Edge Device

**Track 2 — Samsung XR Hackathon 2026**
Organized by **Samsung R&D Institute India, Bangalore** (XR Perception Group & PRISM)

---

## The four submission items, and where they are

Track 2 asks for four things (organizer deck V3-2, slide 11). Every one is in
this repository:

| # | What the organizers asked for | Where it is |
| --: | --- | --- |
| 1 | `predict()` implementation, one JSON object per test clip | [`src/main.py`](src/main.py) — `predict(video_path, clip_id=None)`. Predictions: [`results/predictions.jsonl`](results/predictions.jsonl), 1,251 lines. Each line carries the five fields the deck's example shows (`clip_id`, `prediction`, `top5`, `confidence`, `latency_ms`) **plus one documented extra, `latency_scope: "classifier_only"`** — because this file is produced from cached landmarks, so its `latency_ms` times the nine interpreters only and does not include video decode or MediaPipe. Recording the scope keeps a ~13 ms number from being misread as end-to-end; `predict()` itself returns exactly the five required keys. |
| 2 | Results sheet: one row per metric, measured vs target, pass/fail | [`results/results.csv`](results/results.csv) — six scored rows plus supporting measurements, each with its device and reproduction command |
| 3 | Repository with a README showing how to run it, a quantized model, and the conversion script | This file (run instructions below) · nine FP16 `.tflite` members in [`models/`](models/) · [`scripts/ml/export_ensemble.py`](scripts/ml/export_ensemble.py) |
| 4 | Demo video (2–3 min) or deployed app URL | **Pending** — the recording script is [`submission/DEMO_SCRIPT.md`](submission/DEMO_SCRIPT.md); the link goes in the submission form and on slide 7 of the deck |

Also required by the naming rule: the deck must be named
`CollegeName_TeamName`. It is
[`submission/presentation/CollegeName_TeamName.pptx`](submission/presentation/CollegeName_TeamName.pptx)
with the matching PDF beside it, built inside the organizer's own 9-slide
template. The team+college string and the demo link are the two placeholders
still to be filled.

**The APK to install** is the signed release build, published as a release asset
(80.8 MB is too large for a reliable git push):
[release `v2026-11-22-submission`](https://github.com/dev25bansal-ops/EchoGesture/releases/tag/v2026-11-22-submission).
Its digest is pinned by
[`submission/apk/EchoGesture-2026-11-22-release.apk.sha256`](submission/apk/EchoGesture-2026-11-22-release.apk.sha256).

**Current numbers, in one line:** 85.45% top-1 / 96.64% top-5 on 1,251
signer-disjoint clips; 16.33 MB total; on-device GPU p95 46–54 ms with an empty
frame and 68–70 ms with a hand, on a debug build. Three of the six scored rows
pass and three fail — the table in
[`submission/FINAL_DELIVERY/RESULTS.md`](submission/FINAL_DELIVERY/RESULTS.md)
states which, in its own words.

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

The landmark pipeline clears the accuracy target by 15 points. **Latency, power
and thermal** each have an on-device measurement from a Vivo V2404 (Android 16,
arm64-v8a, SM7550). **Accuracy and robustness are laptop measurements** — they
are corpus evaluations on pre-extracted landmarks, not phone figures, and no
phone accuracy exists. The `device` column of `results.csv` says which is which
on every row; read it before quoting a number.

An earlier version of this section said latency, power and thermal were
unmeasured "because no phone has completed a test session here". That was true
when written and is no longer. An even earlier revision of this paragraph then
overclaimed the other way — that *every* rubric metric was on-device — which the
`device` column contradicts. Both are fixed here.

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
| Model size | **pass** | 8.51 MB across nine members, plus 7.82 MB MediaPipe = 16.33 MB (under the ≤25 MB budget) |
| Pixel robustness ≤ 5 pt | **pass, +4.48** | full 1,251-clip split, four corruptions all under budget |
| Landmark robustness ≤ 5 pt | **1 of 6 fails raw, passes gated** | `drop_hand_25pct` is +5.76 FAIL scored on every clip; **+2.39 PASS** at the app's own margin ≥ 0.15 gate (92.13% clean / 89.74% corrupted at 86%/80% coverage). All five other corruptions pass (camera-further −0.24, half-framerate −1.52, drop-landmarks −2.16, jitter σ0.03 −3.20, jitter σ0.01 +0.16). Both rows are in `results.csv`; the honest reading is in [HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md](docs/HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md) |
| HUD clarity (25% of the rubric) | **pass, verified on a phone** | The app declines rather than guesses: a no-hands or thin capture gets a specific refusal (*"No hands in view"*, *"Hands were visible for too little of that recording"*) and is **not** added to History. A low-margin reading is still shown but labelled *"Uncertain — this is the model's best guess"*. The gate is one function, `ResultPolicy.evaluate`, worth 86.1% coverage at 96.0% accuracy. Evidence: [HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md](docs/HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md) |
| Classifier latency | **pass (desktop)** | 4.2 ms/clip for nine members, x86. **Not an on-device figure** |
| Per-frame detector latency | **pass (desktop), p50 20.3 ms / p95 34.1 ms** | inside the 50 ms budget; 160 frames. Reproduce: `scripts/ml/latency_per_frame.py` |
| End-to-end per clip | **~720 ms (desktop)** | warm `predict()` on a 32-frame clip; the detector is ~600 ms of it, the classifier 4.2 ms. **Not an on-device figure** |
| On-device latency | **FAIL at p95** | Vivo V2404, GPU delegate, DEBUGGABLE build: detect p50 34–45 ms, p95 46–54 ms (no hand) / 68–70 ms (hand in frame) against a ≤50 ms target. Passes in the empty-frame regime, fails once a hand is present. `results.csv` row `latency_p95`, verdict `FAIL` |
| Power ≤ 500 mW | **FAIL, 2666–2898 mW** | **This is the camera hardware floor, not the model.** batterystats prices "camera on" at a fixed PowerProfile constant of **749.4 mA** — five independent blocks in one on-device dump agree to under 0.2% across screen-on, screen-off and doze states, which no real current can do. At 3.87 V that is ~2.9 W, about **5.8× the budget**, so no live-camera app on this device can pass. 84% of the app's own attributed drain is that same camera line; app compute is ~157 mW. `docs/POWER_THERMAL_2026-11-19.md` |
| Thermal drift ≤ 20% | **PASS, 19.2%** | SoC 45.8→56.7°C over 10 min, thermal status LIGHT throughout. Only 0.8 pt of headroom. `docs/POWER_THERMAL_2026-11-19.md` |

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

Source: the organizer's consolidated deck V3-2, slides 23–26.

- **8 Nov 2026, 23:59 IST** — registration closes
- **22 Nov 2026, 23:59 IST** — final submission. *(This line previously read
  18 Nov, as did several docs; V3-2 slide 24 moves it to 22 Nov and v1's "no
  extension" sentence is deleted from V3-2.)*
- **29 Nov** — top 12–15 announced · **9 Dec** — final demo · **12 Dec** — results

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
