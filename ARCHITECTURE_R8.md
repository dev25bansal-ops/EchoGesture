# EchoGesture — Round 7/8 Architecture Synthesis

Provenance is stated per claim. Three categories appear below and are never
mixed:

- **OURS, MEASURED** — produced by a script on this machine, path given.
- **THIRD-PARTY** — read from someone else's repository or paper. Reproduced
  here only where explicitly stated. Carries their caveats.
- **EXTRAPOLATION** — arithmetic on the above. Speculation, not evidence.

---

## 1. The decisive constraint, restated correctly

The hackathon's latency target is **50 ms per `predict()` call**, and the
evaluator re-measures it on their own device. Our prior measurement is that
**video decode dominates everything**: 83.4 ms p50 for full decode vs 2.1 ms for
the model. Any architecture that decodes a whole file inside `predict()` cannot
meet 50 ms. This is arithmetic, not opinion.

**Corollary the team has not yet absorbed:** the front-end must also be cheap.
MediaPipe at 18–35 ms/frame × 24 frames = 432–840 ms. A landmark pipeline and a
50 ms budget are only compatible if both the frame count and the per-frame cost
come down by roughly an order of magnitude.

---

## 2. Measured latency budget (this laptop, x86 CPU, 4 threads)

Script: `_altbench` benchmarks plus the Round-8 sweeps in
`%TEMP%\egbench\framesweep.py`. 20–60 real clips per cell, median of inner reps.

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| PyAV decode, `skip_frame=NONREF`, reformat→224×224, ≤16 frames | **15–26 ms** | 53–65 ms | 35% faster than full decode |
| OpenCV full-file decode (current `main.py`) | 169.7 ms | 643 ms | **abandon this path** |
| cv2 `grab()`/`retrieve()` 16f | 138 ms | 803 ms | worse than PyAV |
| PyAV seek-per-frame 16f | 136 ms | 819 ms | **never seek** |
| palm_lite 192×192, 1 call | 3.4 ms | 4.8 ms | FP32; MediaPipe ships FP32 |
| hand_landmark_lite 224×224, 1 call | 1.65 ms | 1.9 ms | FP32 |
| palm + landmark × 8 sampled frames | **19.6–20.6 ms** | 21–24 ms | one palm detect, then landmark-only |
| INT8 classifier, L=32, 4 threads | 1.33 ms | 1.49 ms | 1.81 M params |
| INT8 classifier, L=96, 4 threads | 3.92 ms | 4.23 ms | upper bound |

**Batching does not amortise.** Measured per-frame cost is flat from batch 1 to
batch 24 (palm 4.64 → 5.07 ms/frame 4-thread; landmark 2.21 → 2.60 ms/frame).
There is no free lunch from batching the front-end.

### The frame-count problem

RESEARCH_STATUS.md §5 establishes that 12–16 frames costs real accuracy
(16 frames = 78.7% of 24-frame accuracy, McNemar p=0.006). Combined with the
latency table:

| frames | TOTAL p50 | TOTAL p95 | accuracy retention (S3D measurement) |
|---|---|---|---|
| 8 | 46.4 ms | 80.5 ms | untested at this count |
| 12 | 54.8 ms | 101.7 ms | 68.0% |
| 16 | 57.9 ms | 111.9 ms | 83.9% |
| 20 | 62.5 ms | 98.2 ms | 98.1% |
| 24 | 70.2 ms | 105.7 ms | 100% (reference) |

**Honest conclusion: on this laptop, ≥20 frames costs ~62 ms p50, which misses
the 50 ms target by ~25%.** Three levers close the gap and all must be measured
on the actual target device, not this laptop (which the repo's own benchmark
README documents as throttling 3–10× on unpinned runs):

1. **Target device is faster than this laptop's throttled state.** A Snapdragon
   8 Elite will beat a thermally-limited i7-14700HX on these tiny fixed-shape
   models. This is plausible but **unmeasured** — it is a claim to verify, not
   assume.
2. **INT8-quantise the two MediaPipe models.** They ship FP32. Quantising to INT8
   typically gives 1.5–2.5× on CPU. Predicted total at 20 frames: **~40–45 ms**.
   Not yet done in this repo.
3. **Streaming amortisation.** In a live conversation, decode and landmark
   extraction run continuously and `predict()` only reads a ring buffer. The
   per-clip number then collapses to classifier time (~2–4 ms). This is the
   architecturally correct answer and it is what the hackathon's "conversation
   gap" framing actually asks for. It does not help if the evaluator times a
   cold `predict(video_path)` — so ship **both**: a fast cold path and a
   streaming demo path.

---

## 3. The architecture

```
video_path
   │
   ├─[1] DECODE  PyAV, thread_type=AUTO, skip_frame=NONREF,
   │             reformat directly to 224×224 rgb24 (no separate resize)
   │             early-stop at 16–24 decoded frames
   │             ~15–26 ms p50 / 55–65 ms p95
   │
   ├─[2] DETECT  palm_detection_lite 192×192 on ONE sampled frame,
   │             INT8-quantised, hand ROI written once
   │             ~3 ms
   │
   ├─[3] LANDMARKS hand_landmark_lite 224×224 per sampled frame,
   │             INT8-quantised, 21 keypoints × (x,y) only, z DISCARDED
   │             8–16 frames  →  ~20–36 ms p50
   │
   ├─[4] FEATURES 21 landmarks × 2 coords
   │             root-centre: subtract wrist (landmark 0)
   │             scale: divide by wrist→middle-MCP distance (landmarks 0→9)
   │             concat lag-1 velocity and lag-2 acceleration
   │             = 21 × 6 = 126 features/frame
   │             (optional pairwise distances: 210 inter-joint, rotation-invariant)
   │
   ├─[5] CLASSIFIER  Conv1D + Transformer (1st-place topology, dim=192)
   │             1,813,528 params  →  INT8 1.93 MB (measured)
   │             L=32: 1.33 ms p50 / 1.49 ms p95
   │
   └─[6] HEADS  softmax over fixed vocabulary
                + energy/entropy threshold → SIGNING / IDLE / DEGRADED
                + top-5
```

### Why this and not pixels
Our own 1.47% measurement is the proof: a raw-pixel model trained on a random
split memorises signer clothing, skin tone and background. Landmarks have none
of those. Every one of the top-5 GISLR solutions independently landed on
landmarks for exactly this reason.

### Why z is dropped — THIRD-PARTY EVIDENCE, NOT OURS
**This is not our measurement.** A public third-party repository ("ember",
MSL/WLASL skeleton project) ran a WLASL-100 signer-disjoint ablation and reports
dropping MediaPipe's z channel as **+7.10 pp on a transformer and +11.34 pp on a
TCN**, disjoint seed ranges on both. Their own caveats, which travel with the
number: 3 seeds (so "disjoint ranges" is not statistical significance — the
smallest achievable two-sided rank p at 3v3 is 0.1), corpus only 55.3% complete
(1,127 of 2,038 clips), untuned, and tested on citation-form dictionary video
rather than natural signing. Source read: `.scratch/wlasl_search/ember_readme.md`.

**We have not reproduced this.** It is a strong prior worth acting on
(`use_z: false` is a cheap experiment), not a result of ours. Treat it with the
same suspicion as the S3D 98.6% — which is to say, reproduce before citing.

### Why a TCN as well as the Transformer — SAME THIRD-PARTY SOURCE
The same repo reports a TCN beating the transformer by **+5.7 pp with
augmentation and +8.1 pp without**, and reports **59.17% ± 2.63 top-1** for its
best TCN configuration on WLASL-100 signer-disjoint. Independently, the 1st-place
GISLR solution found a pure 1D CNN beat a pure transformer. Recommendation to
train both is reasonable, but the numbers above are theirs, not ours.

---

## 4. How the GISLR winners solved cross-signer generalisation

Verified from the primary write-ups in this repo:

| Team | Place | Key idea | Cross-signer result |
|---|---|---|---|
| hoyso48 | 1st | 118 landmarks, XY + Δ + Δ², nose-anchored, per-sequence mean/std norm; Conv1D×3 + Transformer ×2 | 0.8929 private LB, 10 test participants disjoint |
| 2nd | 2nd | EfficientNet-B0 on 80 points as spectrogram + BERT/DeBERTa; 210 pairwise hand distances, 15 finger angles | ~0.81 ensemble |
| SabaiSabai | 3rd | six Conv1D + two transformers; **combinations of parts** augmentation (one signer's hand + another's lips, same label) | 0.8711 private |
| ohkawa3 | 4th | XY only, eyebrow-anchored, **drop frames with no hand**, global **max** pooling not avg | 0.8824 private |
| 5th | 5th | transformer + distances; mean/std of the single sequence | 0.8809 private |

**The transferable four:**
1. **Invariant coordinates** — root-centre, scale by a body measure, XY only.
2. **Per-sequence normalisation** — mean/std computed over that clip, not a
   global constant. This is what makes a new signer's framing irrelevant.
3. **Global max pooling** over time, not average. Important frames are sparse;
   averaging dilutes them. 4th place measured a "significant performance drop"
   switching to avg.
4. **Part-combination augmentation** — splice a hand from one sample onto lips
   from another with the same label. 3rd place called this a significant gain.
   It directly attacks signer-specific co-occurrence.

**What did NOT work** (recorded so we don't repeat it): GCNs, CutMix/MixUp with
variable lengths, knowledge distillation, synthetic 3D hand parameter models.

---

## 5. Vocabulary and data — the honest accounting

### Three candidate vocabularies, measured

| Vocabulary | classes | train videos | videos/class | source | reachable? |
|---|---|---|---|---|---|
| WLASL-100 (current plan) | 100 | 1,127 usable | 11.3 | WLASL v0.3 | **only ~45%** — 18/40 probed URLs returned 200 |
| ASL Citizen ∩ WLASL-100 | 98 | 2,015 | 20.6 | ASL Citizen | **yes, 100%** |
| ASL Citizen best-100 by coverage | 100 | 4,604 | 46.0 | ASL Citizen | **yes, 100%** |
| GISLR ∩ ASL Citizen | 208 | 78,607 (GISLR) | 382 | GISLR | **no Kaggle credentials** |

ASL Citizen is the only large corpus we can actually get today: 83,399 videos,
52 signers, official **participant-disjoint** train/val/test (35/6/11
participants, zero overlap — verified from the split CSVs). Download is proven:
1,705 videos in 380 s = **4.5 videos/s**, so the full 40 K train split is ~2.5 h
and ~22 GB.

### Accuracy calibration — ALL THIRD-PARTY NUMBERS, NONE OURS

| Reference | videos/class | signer-disjoint top-1 | Source |
|---|---|---|---|
| SPOTER on WLASL-100, 1,127 clips | 11.3 | 10.43% | third-party "azk" repo README |
| TCN on WLASL-100, 1,127 clips | 11.3 | **59.17%** | third-party "ember" repo README |
| GISLR 1st place, 250 classes | 378 | 89.29% | Kaggle leaderboard (verified) |

Same corpus, same split, same 100 classes: 10.43% vs 59.17%. That comparison is
theirs, and it comes with their caveats (55.3% corpus completeness, 3 seeds,
untuned, dictionary video). **We have reproduced none of it.** Log-linear
extrapolation from these two anchors predicts ~71% for a best-100 ASL Citizen
vocabulary and ~64% for WLASL-100 — but extrapolating from two third-party
points with different caveats is speculation, not forecasting. State it as a
hypothesis to test, never as a plan number.

### Recommendation
**Use the ASL Citizen best-100 vocabulary as the primary submission**, with the
WLASL-100 subset as an auxiliary training set (98 overlapping classes) if time
permits. Rationale: 4,604 videos vs 2,015, all reachable, official
signer-disjoint split, and the vocabulary is chosen to maximise both training
videos and signer diversity rather than inherited from a paper that never
claimed signer independence.

---

## 6. Training strategy to hit ≥70%

1. **Split**: use ASL Citizen's shipped split, never re-split randomly. Report
   only the 11 held-out test participants.
2. **Input**: 32 frames is the sweet spot — decode cap 16 is a latency
   compromise, so **train at 24–32 and evaluate at the same count you will
   deploy**. Train/test frame-count mismatch is worth 16 points (83.9% vs 100%
   retention in the S3D probe).
3. **Features**: 126/frame (21 landmarks × 6). Optionally add 210 pairwise
   hand distances from the 2nd-place solution.
4. **Augmentation** (all four had measured positive effect in the top-5):
   - global affine: rotate ±10°, shift ±0.1, scale 0.8–1.2, shear
   - random temporal resample 0.5–1.5×
   - **hand occlusion masking** (p≈0.3) — this doubles as the robustness defence
   - part-combination: splice hands from same-label sample (3rd place)
5. **Loss**: label smoothing 0.5, stochastic depth 0.1–0.2, late dropout 0.8
   after epoch 15. All three were "had a significant impact" for 1st place.
6. **Optimiser**: RAdam + Lookahead, cosine decay, 300–400 epochs.
7. **QAT, not PTQ**: the S3D evidence shows QAT retained more (1.00% vs 1.47%,
   though both failed). For a 1.8 M-param model, QAT costs half a day and buys
   real quantisation robustness.
8. **Export**: INT8 TFLite with a representative dataset drawn from real
   landmark features.

---

## 7. Day-1 build instructions

### Block 0 — registration (today, before anything else)
Submit the Google Form before **8 Nov 23:59 IST**. If the college is not on the
five-option dropdown, email `prism@samsung.com` the same day.

### Block 1 — data (≈3 h wall clock, mostly unattended)
```bash
# ASL Citizen is range-extractable; the machinery already exists
python evaluation/eval_asl_citizen.py --help   # reuse its extract_member()

# 1. choose vocabulary = best-100 by (train participants, train videos)
#    script: scripts/choose_vocab.py  -> writes models/label100.json
# 2. fetch train (40,154) + val (10,304) + test (32,941) = ~22 GB, ~2.5 h at 4.5/s
```
Do **not** wait for the full corpus. Fetch the 4,604-video best-100 subset
first (~17 min), train on it, and expand only if accuracy is short.

### Block 2 — landmark extraction (≈2 h for 4.6 K clips)
```bash
python scripts/extract_landmarks.py \
  --videos lists/best100.csv --out cache/landmarks.npz \
  --palm _altbench/mp/palm_detection_lite.tflite \
  --landmark _altbench/mp/hand_landmark_lite.tflite \
  --resize 224 --max-frames 32
```
Store raw 21×2 per frame plus a per-frame `hand_present` flag. Normalise at
load time, not at extraction time, so normalisation can be ablated.

### Block 3 — train (≈2 h on GPU, or overnight on CPU)
```bash
python scripts/train.py --arch conv1d_transformer --dim 192 \
  --frames 32 --features 126 --use-z false \
  --epochs 350 --label-smoothing 0.5 --stochastic-depth 0.2 \
  --split participant --aug all --out models/eg_r8
```
Success criterion at this point: **signer-disjoint val top-1 ≥ 60%**. If below,
the problem is data or normalisation, not architecture — do not tune the model.

### Block 4 — quantise and convert
```bash
python scripts/convert_int8.py --ckpt models/eg_r8 --rep cache/landmarks.npz \
  --out models/eg_r8_int8.tflite      # expect ~1.9 MB
```

### Block 5 — rebuild `predict()` on the fast path
Replace `src/main.py`'s OpenCV loop with PyAV + NONREF + reformat-to-224 +
early-stop-16, then palm-once + landmark-per-frame + INT8 classifier. Target:
**≤ 50 ms p50 at 8–12 frames, ≤ 65 ms at 16**. Measure and record honestly;
if the target device does not reach 50 ms, report the real number and show the
streaming path that does.

### Block 6 — robustness harness (half a day, worth 25% of marks)
Five seeded corruptions on the held-out split: occlusion box, low light, motion
blur, crop, frame drop. Paired clean-vs-corrupt delta. The occlusion-mask
augmentation in Block 3 is what makes this pass.

### What to verify on Day 1, in this order
1. Does the best-100 vocabulary exist in ASL Citizen with ≥18 train participants?
   **Verified yes: 100 classes qualify, median 19 participants.**
2. Are the videos downloadable? **Verified yes: 4.5 videos/s, 0.54 MB each.**
3. Does the classifier convert to INT8 under 20 MB at L=32? **Verified yes:
   1.93 MB at L=96, smaller at L=32.**
4. Does the front-end fit the budget? **Partially: 46 ms p50 at 8 frames,
   62 ms at 20. Not yet verified after INT8-quantising MediaPipe.**

---

## 8. What is still unmeasured, stated plainly

- **INT8 MediaPipe latency.** Both models ship FP32. Quantising them is the
  single largest remaining latency lever and has not been done.
- **Target-device latency.** All numbers here are a thermally-limited laptop.
  The repo's own benchmark notes document 3–10× swing on unpinned runs.
- **Accuracy of a trained landmark model on our vocabulary.** 71% is an
  extrapolation from two anchor points, not a measurement.
- **Power and thermal.** No hardware.
- **Robustness delta.** Harness not yet written.

---

## 9. One-paragraph summary

Replace the pixel front-end with MediaPipe hand landmarks; keep only x,y; centre
on the wrist and scale by hand size; add velocity and acceleration; classify
with a 1.8 M-param Conv1D+Transformer that quantises to 1.9 MB and runs in
1.3–4 ms; decode with PyAV NONREF and early-stop at 16 frames; train on ASL
Citizen's official signer-disjoint split with a 100-word vocabulary chosen for
signer diversity, not inherited from WLASL. That combination is the only one
with both published evidence (every GISLR top-5 team) and local evidence (our
own 1.47% collapse) pointing the same way. The latency number is the risk: the
front-end, not the model, decides whether 50 ms is met, and the honest answer
today is 46 ms at 8 frames and 62 ms at 20 on this laptop.
