# Robustness: the two hand-dropout numbers, and which one is the harness

Written 2026-11-16 auditing `hackathon/dev-model` @ `465166f`.

---

## The short version

| Figure | Meaning | Use it? |
| --- | --- | --- |
| **+5.76 pp** | Both coordinates **and** the presence flag zeroed — what `scripts/ml/robustness.py` actually does | **Yes. This is the unconditional result.** |
| +5.51 pp | Presence flag zeroed, **coordinates kept** — a diagnostic variant | Only to prove the mask works |
| +5.52 pp | Same as 5.51, with 0.01 arithmetic slippage | **No — mislabelled** |
| +1.24 pp | Both zeroed, scored **only on clips the app accepted** (86% clean / 80% corrupted coverage) | Only with the coverage stated |

**The submission's honest unconditional robustfigure is +5.76 pp, and it FAILS
the ≤5 pp criterion.**

## Why 5.76 and 5.52 differ — the actual cause

They are **not two remeasurements of the same thing.** They are two different
corruptions.

`docs/hackathon/DROP_HAND_ANALYSIS.md` states the decomposition outright:

| Corruption applied | Accuracy | Delta |
| --- | --: | --: |
| clean | 85.45% | — |
| presence flag zeroed, coordinates kept | 79.94% | +5.51 |
| **both zeroed (what the harness does)** | **79.70%** | **+5.76** |
| coordinates zeroed, presence kept | 77.94% | +7.51 |

`scripts/ml/robustness.py` zeroes both. So **79.70 / +5.76 is the harness
measurement**, and `results/results.csv` correctly labels it:

> `accuracy_drop_robustness, …, 5.76, …, FAIL`

But `results/robustness_gated.json` labels the **diagnostic** as the raw row:

```json
{ "mode": "raw (every clip)",
  "clean_pct": 85.45, "drop_pct": 79.94, "delta_pp": 5.52, "verdict": "FAIL" }
```

- `drop_pct: 79.94` is the **coords-kept** variant, not the harness corruption.
- 85.45 − 79.94 = **5.51**, not 5.52. The file is also off by 0.01 against its
  own two numbers.

So the repo publishes one figure under two names. **5.76 is correct.**

### The population discrepancy

- `DROP_HAND_ANALYSIS.md` and `GEOMETRY_FINDING.md` say **1,249 clips**.
- `eval_summary_test.json`, `results.csv`, `robustness_gated.json` and
  `predictions.jsonl` say **1,251** (1,255 minus 4 with no hand in any frame).

The 5.76 delta is computed against the 1,251-clip clean baseline. The 1,249
figure is stale text.

## The gated +1.24 — read the coverage with it

| Operating point | Clean | drop_hand_25pct | Delta | Coverage clean / corrupted |
| --- | --: | --: | --: | --- |
| raw, every clip | 85.45% | 79.70% | **+5.76 FAIL** | 100% / 100% |
| app gate, margin ≥ 0.15 | 92.13% | 90.89% | **+1.24 PASS** | **86% / 80%** |
| stricter gate, margin ≥ 0.30 | 96.11% | 94.78% | +1.33 PASS | 76% / 69% |

The pass is earned by **answering fewer clips**: 14% of clean clips and 20% of
corrupted clips are rejected. That is a legitimate design choice, but it is not
unconditional robustness, and the two must never be separated.

**Whether the competition permits abstentions in its robustness metric is
UNKNOWN.** No rubric language on this has been verified. Until it is, do not
claim a robustness PASS.

## What the gate actually is

The gate is **top-1 minus top-2 probability margin ≥ 0.15**, plus a 20% score
floor that the app documents as a degenerate-distribution guard.

This was verified directly in the Android source:

```kotlin
// CameraScreen.kt:958
val certain = result.margin >= 0.15f && percent >= 20f
```

The margin is computed as `probs[order[0]] - probs[order[1]]`
(`LandmarkClassifier.kt:238`).

**It is not a 0.8 confidence threshold, and it is not a 0.5 abstention.** Any
document claiming those are the app's live behaviour is wrong.

The gate does **not** withhold a prediction. When the margin is below 0.15 the
card still shows the top-1 plus the label:

> *"Uncertain — this is the model's best guess, not a confirmed reading"*

## Pixel robustness is a different experiment

The +4.48 pp pixel PASS (blur, low light, downscale, JPEG) was measured at
**512-pixel letterbox** preprocessing with the Python detector.

**The Android app does not use 512 letterbox.** It feeds CameraX native frames to
MediaPipe `detectForVideo`. A pixel-robustness PASS therefore cannot be
transferred to the phone. Phone robustness is **unmeasured**.

## What to say

> "Hand dropout is the criterion we fail. Losing a hand for a quarter of a clip
> costs 5.76 points unconditionally. At the margin gate the app actually uses, it
> costs 1.24 points — but that is on the 86% of clean clips it accepts, so it is
> not the same claim. I'm not going to present the gated number as if it were
> unconditional."

## Actions for Dev

1. Fix `results/robustness_gated.json`'s "raw" row to use the harness corruption
   (79.70 / +5.76), or relabel it clearly as the coords-kept diagnostic.
2. Correct 5.52 → 5.51 wherever it appears with 79.94.
3. Reconcile 1,249 vs 1,251 across `DROP_HAND_ANALYSIS.md` and
   `GEOMETRY_FINDING.md`.
4. State coverage in every sentence that quotes +1.24.

No retraining is required or implied by any of this.
