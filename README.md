# EchoGesture: Sign Language Prediction on Edge Device

**Track 2 — Samsung XR Hackathon 2026**
Organized by **Samsung R&D Institute India, Bangalore** (XR Perception Group & PRISM)

---

## Status: research complete, submission NOT ready

This repository contains a working inference pipeline, an empirically-validated
diagnosis of why the obvious approach fails, and an honest measurement harness.
It does **not** yet contain a model that meets the accuracy target.

Read [RESULTS.md](RESULTS.md) before reading any number in this file.

---

## What actually works

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
├── src/main.py                     # predict(video_path) -> dict  [submission entry]
├── src/requirements.txt
├── models/
│   ├── s3d_conv1d_qat_int8.tflite  # 880.1 KB
│   ├── ptq_100_int8.tflite         # 883.1 KB
│   └── label100.json
├── results/
│   ├── results.csv                 # measured vs target, honest FAILs included
│   └── predictions.jsonl           # 1,705 real predictions from the ASL Citizen run
├── evaluation/
│   ├── eval_asl_citizen.py         # participant-disjoint evaluation harness
│   ├── eval_wlasl_signers.py       # in-distribution positive control
│   └── asl_citizen_predictions.csv # raw per-clip output
├── S3D_Conv1D_analysis.md          # paper extraction, all tables, MCU logs
├── S3D_signer_disjoint_evaluation.md # the 1.47% investigation
├── wlasl100_asllex_vocab_justification.csv
└── TEAM_GUIDE.md                   # onboarding, plan, work streams
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
