# Handoff to Dev — recognition accuracy, no-retrain evidence

Status: **NOT YET REQUESTED.** This records what has been established and what
is missing. Per the phase gate, a model-work request needs phone evidence that
does not exist yet.

## What was verified on this side (no model change)

- **Android VIDEO mode matches Python IMAGE mode.** Same authorized clips,
  both modes, separate processes: identical top-1 on 4/4 (EAT1, WHATFOR1,
  DOG1, BASKETBALL1), hand counts within ±1. Feature tensors differ slightly
  (mean |Δ| 0.017) without moving the decision.
- **Hand slots, rotation and mirroring are consistent** between the two code
  paths. Slots key on MediaPipe handedness; neither path mirrors; rotation is
  applied once. Single-hand slot flicker measured at 0.62 per clip over 40
  clips — present but not systematic.
- **Feature construction is fixture-asserted** against `src/features.py` to
  1e-4, and the 32→24 resample matches `_resample()` exactly.
- **The shipped pipeline scores 84.2% top-1 / 93.9% top-5** on 247 in-vocabulary
  ASL Citizen validation clips, with **96.8% accuracy at 62.8% accepted
  coverage**.

None of this identifies a model defect. It establishes that the pipeline is
not the reason phone accuracy is unknown.

## Measured model behaviour worth your attention

From all 345 in-vocabulary validation clips:

| Outcome | Share |
| --- | ---: |
| Correct top-1 | 75.36% |
| Correct at rank 2–5 | 11.9% |
| Never in top-5 | 12.8% |

**Attractor labels** (predicted wrongly most often): BREAKFAST1 (7),
INTUITIVE (6), TAIL1 (5), JEWELRY (4), then MOVIE1, RAZOR2, CHEER, SHAVE1,
SCARF2 at 3 each.

**Weakest classes**: DOG1 1/4, LOCK1 2/5, BEE1 3/6, CHEESEGRATER 2/5,
FOREIGNER1 1/4, CHEEK 1/4.

**Repeated confusion pairs**: DOG1→INTUITIVE (2), ZEBRA→JEWELRY (2),
FOREIGNER1→SCARF2 (2).

## What would justify model work

The gate is: phone capture shows **high hand coverage** (≥40% of the window)
**and** a wrong label above the acceptance margin. That combination rules out
capture and pipeline as causes and leaves the model.

Until that exists, the evidence does **not** support retraining, and the
variants below are listed for discussion rather than requested.

## Ordered candidates if the gate is met

1. **Variable-duration temporal augmentation.** Training samples 32 frames
   across each whole clip (~115 ms steps); the app now samples across a fixed
   3.7 s span. A signer faster or slower than the corpus lands on a different
   time base. `linspace` over a randomly truncated span, resampled to a random
   step in ~60–200 ms, directly targets that mismatch.
2. **Frame-drop augmentation.** `FINAL_EVIDENCE.md` already records 13.33 pt
   degradation under frame_drop at severity 0.5. A real phone's variable
   analysis FPS is frame dropping, so this is the known-weakest axis and it is
   already measured as such.
3. **Phone-camera domain adaptation.** Perspective, illumination and white
   balance augmentation. Justified only if phone failures show high coverage
   with consistent errors, i.e. the model sees the gesture and still misses.
4. **Confusable-class analysis** for the attractor labels above. Needs several
   signers per pair before it is worth acting on; 2–6 clips per class is not
   enough to build a class-balanced intervention.
5. **Confidence calibration.** Not a training change: the app now gates on
   top-1/top-2 margin, measured to give the same accuracy at higher coverage
   than the old score cut. Calibration would replace that ranking heuristic
   with a fitted quantity, and requires held-out data the app does not have.

## What I need from you

- **Confirmation of what the frozen `predictions_ensemble_val.jsonl` was
  produced by.** It does not match the shipped pipeline on the clips tested
  (WHAT FOR: PENNY 0.547 frozen vs WHATFOR1 0.567 shipped). Without knowing its
  provenance, it cannot serve as a like-for-like baseline.
- **Signer-level breakdown** for the validation split, if it exists. Per-signer
  variance is the largest unknown in the current evidence: 3–6 clips per class
  from 5 signers cannot separate a bad class from a bad signer.

## Data handling

No clip, landmark array, frame or diagnostic from this work has been committed.
Clips are research-use, fetched on demand, and deleted after use. Nothing here
redistributes data or weights.