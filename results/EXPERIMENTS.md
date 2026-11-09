# Experiments tried and rejected

Every entry here was measured, not assumed. Each says what was tried, what it
scored, and why it was not shipped. Kept so nobody repeats the work.

All numbers are signer-disjoint test top-1 on 1,249 clips unless noted. The
shipped baseline is a 4-model ensemble at **72.78%**.

---

## Data availability: there is no more of it

Before trying to train more, I checked how much data actually exists.

The ASL Citizen archive holds **83,399 clips across 2,731 glosses**. For our
100 glosses the official split contains exactly **3,390 clips**, and all 3,390
were already extracted. The other 80,009 archive clips belong to the other
2,631 glosses — they are not our vocabulary, so they cannot be added.

Per gloss in the training split: **min 17, median 17, max 24 clips.**
ASL Citizen simply did not film these 100 signs with more than those 32
participants. There is no hidden reserve.

### The only way to add data, and what it costs

| option | training clips | test set | verdict |
| --- | --- | --- | --- |
| official split | 1,787 | 11 unseen signers | the shipped setup |
| **merge val into train** | **2,132 (+19%)** | **11 unseen signers, untouched** | tried |
| merge all three splits | 3,390 (+89%) | none left | rejected |

Merging val is legitimate: val's 5 signers do not overlap the test 11
(verified at both signer and clip level — 2,135 + 1,255 = 3,390 exactly, zero
intersection). It buys +19% training data and costs the val split as a
selection set.

**Tried it. Result: +0.24 points, which is noise.**

| | test top-1 |
| --- | --- |
| shipped 4-model ensemble | 72.78% |
| + merged-data model | 73.02% |

With n=1,249 the standard error is ±1.26 points, so +0.24 is **0.19 standard
errors** — indistinguishable from zero. Training-set accuracy also stayed at
~98% for both, so the extra 345 clips did not change the memorisation regime.

This is the clearest evidence for the real conclusion: **the problem is not
"not enough data" in the sense that 19% more would fix it.** Doubling the
clips per class (17 → 32) might; a fifth more does not.

Merging all three splits was rejected. It would leave no held-out signers, so
the headline number would stop being comparable to anything — and with 18 of
48 signers contributing fewer than 30 clips (some as few as 2), any re-split
would either strand those signers or make val/test too small to be stable.

## Rejected: the `_resample` "phantom hand" fix — 64.61%

`features._resample` interpolates coordinates and the presence flag
independently, so a frame straddling a hand appearing or disappearing blends a
real hand with an absent one into a half-size phantom. Making that consistent
(zero the slot when either source frame lacks the hand) looked obviously
correct, and `STRICT_RESAMPLE` was added so it could be tested.

Full retrain with the corrected transform:

| | clean | hand dropped |
| --- | --- | --- |
| shipped 4-model ensemble | **72.78%** | **58.21%** |
| strict91, own retrain | 64.61% | 47.64% |

Worse on **both** axes. It does not even fix the hand-drop case it was
designed for — because removing the blended frames also removes the
transition frames that carry the sign's onset, and 17–24 clips per class
cannot pay for that.

Adding it to the shipped ensemble gives 72.62%, no gain. The flag stays in
`features.py` (default False, documented) so the experiment is reproducible,
but the shipped transform is correct as-is. This entry exists because the
fix *looked* right and measuring it is what stopped it shipping.

## Rejected: speed perturbation — 63.65%

Stretched or compressed each training clip's timeline by 0.8x–1.25x before the
resample, so the model would see the same sign at different tempos.

**Result: 63.65% vs 69.10% for the same seed without it. −5.45 points.**

ASL Citizen clips are already trimmed to a single sign of similar duration, so
the perturbation mostly destroyed timing that was genuinely informative rather
than adding variation the model needed.

## Rejected: temporal multi-crop TTA — 72.22–72.94% vs 72.78%

Evaluated 3 or 5 windows of each clip and averaged their logits.

It looked like a +2 point win when measured on a 400-clip subset (74.70% and
75.10% at 3 and 5 crops). On the full 1,249-clip set it does not hold:

| window | val | test |
| --- | --- | --- |
| **shipped (full-clip resample)** | 83.77% | **72.78%** |
| crop at centre | 85.80% | 72.22% |
| 3-window | 85.51% | 72.94% |
| 5-window | 84.93% | 72.86% |
| crop at start | 84.93% | 68.94% |

The val and test columns disagree: cropping wins on val and loses on test,
which is what selection noise looks like. **The subset measurement was the
error** — 400 clips is ±2 points, enough to make a null result look real.

One finding *is* solid: cropping at the START of a clip costs 3.8 points
(68.94% vs 72.78%). Hand presence by clip decile peaks in the middle
(0.77 at decile 4) and is sparse at both ends, so a start-anchored window
misses the sign. The shipped `read_frames` already samples evenly across the
whole clip, so it does not have this problem.

## Rejected: mirror TTA on the mirror-trained member — −1.01 points

Averaging each test clip with its mirror. Helps a model that was *not* trained
with mirror augmentation; on `mirror05_s0`, which was, it costs a point —
the model is already mirror-invariant, so the mirrored copy is redundant and
its errors are correlated rather than complementary.

## Rejected: INT8 quantization — 68.99% vs 75.36% float on val

6.37 points lost, and two TFLite runtimes disagreed on the same file
(`ai_edge_litert` 72.0%, `tf.lite` 4.0% on 50 clips). Full analysis in
[QUANTIZATION.md](QUANTIZATION.md). FP16 loses zero points, so it ships.

## Rejected: wider model dim=256 — 61.25%

More capacity made it worse, which the overfitting measurement explains:
the model already reaches **99.4% on training data** while managing 72% on
unseen signers. The constraint is data (17–24 clips per class), not capacity.

## Rejected: 32-frame input — 60.85–65.09%

Feeding 32 frames instead of 24. Never beat the 24-frame models as a member of
the ensemble.

## Measurement errors worth remembering

Two of these cost more time than any model change:

1. **Measuring on a subset.** The TTA "gain" above was real on 400 clips and
   gone on 1,249. Any comparison under ~1,000 clips is not trustworthy at this
   effect size; always confirm on val as well as test.
2. **Reading a checkpoint mid-training.** Concurrent training runs starved this
   15.7 GB machine, and I evaluated partially-written checkpoints as if they
   were results — producing nonsense like 9.21% and 33.79%. A checkpoint is
   only valid once its run has finished and printed a best-val line.

Both are now why every number in `results.csv` names the split and the n.