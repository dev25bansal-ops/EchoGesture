# EchoGesture: Sign Language Prediction on Edge Device

**Track 2 — Samsung XR Hackathon 2026**
Organized by **Samsung R&D Institute India, Bangalore** (XR Perception Group & PRISM)

---

## Hackathon team workflow

Start with the [two-person sprint plan](docs/hackathon/README.md) and the
[agent guide](AGENTS.md) for Abhijeet and Dev's ownership and handoffs.

## Status: submission-ready model, on-device numbers outstanding

The landmark pipeline works and clears the accuracy target by 15 points. What is
**not** done is on-device measurement: latency, power and thermal are unmeasured
because the only available emulator cannot run MediaPipe (no x86_64 native
library) and no phone has completed a test session here.

Current numbers, measured on the FP16 `.tflite` files that ship:
[docs/hackathon/FINAL_NUMBERS_2026-11-15.md](docs/hackathon/FINAL_NUMBERS_2026-11-15.md).

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
| Landmark robustness ≤ 5 pt | **fail, +5.76** | `drop_hand_25pct` only; the other five pass. A model limitation, not a measurement artefact — see [DROP_HAND_ANALYSIS.md](docs/hackathon/DROP_HAND_ANALYSIS.md) |
| Classifier latency | **pass (desktop)** | 4.2 ms/clip for nine members, x86. **Not an on-device figure** |
| On-device latency | **not measured** | needs a phone; the emulator cannot run MediaPipe |
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

## Working data and environment

The full working tree — 16.9 GB, including the 3,390 ASL Citizen clips, the
landmark caches, the reproduction virtualenvs and the built APK — is mirrored
to a **private** Hugging Face dataset:
[`Dev2506/echogesture-full`](https://huggingface.co/datasets/Dev2506/echogesture-full).

It is private because of the licence above: it holds research video that cannot
be redistributed. Download commands, an inventory of what is inside, and the
small shipping subset live in
[docs/hackathon/HF_DATASET.md](docs/hackathon/HF_DATASET.md).
