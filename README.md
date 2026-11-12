# EchoGesture: Sign Language Prediction on Edge Device

**Track 2 — Samsung XR Hackathon 2026**
Organized by **Samsung R&D Institute India, Bangalore** (XR Perception Group & PRISM)

---

## Hackathon team workflow

Start with the [two-person sprint plan](docs/hackathon/README.md) and the
[agent guide](AGENTS.md) for Abhijeet and Dev's ownership and handoffs.

## Setup and quickstart

Requires **Python 3.11** (the pinned `ai-edge-litert` / `mediapipe` wheels).

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r src/requirements.txt

# Model weights are NOT in this repo (see models/README.md for why and how).
python src/main.py /absolute/path/to/clip.mp4
```

Run the tests:

```bash
.venv/bin/python -I scripts/run_all_tests.py --quiet
```

`unittest discover` is **not** used: 39 of the 59 test files sit outside packages,
so discovery silently runs about half the suite. The runner loads every file by
path, one subprocess each.

Run the offline demo (real inference, no camera, no network):

```bash
.venv/bin/python -I app/offline_demo.py --video /path/to/clip.mp4
```

See [models/README.md](models/README.md) for the artifact contract and
[TEAM_WORKFLOW.md](docs/TEAM_WORKFLOW.md) for branching and handoff rules.

---

## Status: integrated and packaged, external actions remaining

The default runtime is Dev's **four-model FP16 TFLite ensemble**: every member
consumes the same `[1, 24, 255]` float32 feature tensor, logits are averaged
element-wise, and a single softmax runs over the mean. It has been executed
through the shared `predict()` API and from a freshly extracted release bundle
on real ASL Citizen clips — `WHATFOR1` at 0.8821 confidence.

That is an integration proof, not a compliance claim: the ensemble does not
meet the 50 ms clip-p95 target on the machine it was measured on, and no
physical Android device was available. See
[submission_final/FINAL_CHECKLIST.md](submission_final/FINAL_CHECKLIST.md) for
the current state of every deliverable, and
[docs/submission/COMPLIANCE_FINAL.md](docs/submission/COMPLIANCE_FINAL.md) for
the requirement-by-requirement audit against the organizer deck.

**Accuracy summary (signer-disjoint held-out test, 100 glosses).** The shipped
runtime is a **4-member FP16 ensemble** and scores **909/1249 = 72.78% top-1 /
87.51% top-5** across 11 unseen signers, clearing the ≥70% internal target. Six
of 1,255 eligible clips produced no prediction (no hand detected); counting
them as wrong gives 72.43%. The previous single-model runtime scored 62.61% on
the same clips. Validation topline for the single model was 272/346 = 78.61%
(INT8 254/346 = 73.41%).
The test split was inspected during model selection, so it is not a pristine
blind test. See [FINAL_EVIDENCE.md](docs/release/FINAL_EVIDENCE.md).

Read [candidate integration](evaluation/device/CANDIDATE_INTEGRATION.md),
[RESULTS.md](RESULTS.md) for the old S3D baseline, and
[FINAL_EVIDENCE.md](docs/release/FINAL_EVIDENCE.md) for remaining release gates.

---

## What actually works in the integrated candidate

| Item | Status | Evidence |
|---|---|---|
| `predict(video_path)` contract | **working** | Real landmark and S3D paths, optional exact manifest-ID override |
| Landmark candidate | **working, 4-model FP16 ensemble** | Real `WHATFOR1` (0.8821) prediction; ensemble-aware release ZIP extracts and runs |
| Stage profiling | **working** | `evaluation/device/run_manifest.py --profile` |
| Representative accuracy/p95 | **measured, disclosed** | 72.78% top-1 / 87.51% top-5 on a signer-disjoint held-out split; p95 274.2 ms singleton, 99.8 ms parallel-4 on a laptop. 50 ms not met. |
| Live camera demo | **blocked locally** | Real worker/HUD implemented; macOS camera permission denied |
| Android demo | **APK built and verified; on-device untested** | `app/android_final/app/build/outputs/apk/debug/app-debug.apk`, 75,072,782 B, signed, zipaligned, all 4 members + detector embedded, 9/9 unit tests. No phone attached, so no camera/power/thermal result is claimed. |
| INT8 compliance | **AMBIGUOUS in organizer material** | Slides p.9/p.10 say "quantized model (INT8)"; the scored rubric p.11 says "Compilation to TFLite/ONNX <20MB" with no INT8 mention. Shipped ensemble is FP16 (11.37 MB). The INT8 single-model alternative scores 73.41% vs 78.61%. Treat as an open 20%-weight scoring risk pending written clarification |

The table below is the **old S3D research baseline**, not the landmark model.

| Item | Status | Evidence |
|---|---|---|
| `predict(video_path)` contract | **working** | [src/main.py](src/main.py), returns the required JSON |
| INT8 model size ≤ 20 MB | **pass** | 880.1 KB measured with `ls -l` |
| Model inference latency | **pass** | 2.1 ms p50 / 3.2 ms p95 measured, N=200, this machine |
| End-to-end `predict()` latency | **fail** | 54.1 ms p50 / 117.8 ms p95 measured, N=60 real clips |
| Accuracy on unseen signers | **fail** | 1.47% (25/1705) vs 1.02% chance, ASL Citizen participant-disjoint |
| Robustness drop ≤ 5 pt | **not measured** | no occlusion/lighting test script exists yet |
| Power ≤ 500 mW | **not measured** | no measurement hardware or capture |
| Thermal drift ≤ 20% | **not measured** | no sustained 10-min run performed |

## The core empirical finding

We downloaded the published S3D-Conv1D INT8 weights (shipped by Kimpinde &
Olukanmi, *Algorithms* 2026) and evaluated them ourselves.

**On the licence:** the upstream authors describe the code as MIT, but the
weights derive from WLASL, whose terms are academic / non-commercial. This repo
holds **no documentary permission to redistribute the weight files**, which is
why `NOTICE` marks them local/private only. An earlier revision of this line
said "MIT-licensed", which overstated what we can prove. Two of those files
nevertheless appear in this repository's history from an early commit — see
[LICENSING_REVIEW.md](docs/submission/LICENSING_REVIEW.md).

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
├── src/
│   ├── main.py                      # predict(video_path) -> dict  [submission entry]
│   ├── features.py                  # 255-feature transform (Dev-owned)
│   ├── landmark_frontend.py         # MediaPipe hand landmarks (Dev-owned)
│   └── requirements.txt
├── models/
│   ├── README.md                    # artifact contract + local setup (weights not committed)
│   ├── landmark_labels.json         # 100 ordered glosses (committed)
│   ├── landmark_model_meta.json     # frames/feature-dim/classes (committed)
│   └── *.tflite, *.task             # restricted weights — local only, see NOTICE
├── app/
│   ├── offline_demo.py              # tap-to-capture HUD
│   ├── v7_demo/                     # reliable demo wrapper (file + camera)
│   └── android_final/               # offline Android app source
├── scripts/
│   ├── ml/                          # Dev: export/convert
│   └── release/                     # build_bundle.py, compliance gates
├── evaluation/device/               # benchmarks, robustness, test suites
├── results/
│   ├── results.csv                  # measured vs target, honest FAILs included
│   └── predictions.jsonl            # 1,705 predictions (signer IDs pseudonymised)
└── docs/                            # demo, release evidence, team workflow
```

## Quickstart

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r src/requirements.txt
python src/main.py path/to/clip.mp4     # needs locally supplied weights; see models/README.md
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
