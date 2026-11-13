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
## 13 Nov: augmentation and diversity sweep — no gain that survives both splits

The error structure was measured first, so the search was aimed rather than
random. `scripts/ml/search.py` trains variants; `assemble.py` scores subsets.

### Where the errors actually are (test, n=1249)

| condition | clips | acc |
| --- | --: | --: |
| both hands present | 1,138 | **76.8%** |
| one hand <25% of frames | 111 | **31.5%** |

Single-hand clips are 22% of all errors (76 of 340) and are four times worse
than the rest. That is the clearest target in the data, so the sweep attacked
it with mirror augmentation (handedness invariance) and stronger hand dropout.

### Per-clip oracle shows the ceiling is aggregation, not capacity

| | |
| --- | --- |
| shipped ensemble | 72.78% |
| at least one of the 4 members right | **80.30%** |
| clips no member gets right | 246 (19.7%) |

7.5 points sit between the vote and the oracle. That gap is the prize.

### Rejected: mean-probability aggregation — wins test, loses val

| rule | val | test |
| --- | --: | --: |
| mean logit (shipped) | **83.77%** | 72.78% |
| mean probability | 82.90% | **73.66%** |

Paired McNemar: test `+19/-8 z=2.12`, val `+3/-6 z=-1.00`. It wins on test and
loses on val, which is the same failure signature the TTA result had. Rejected.
Only 11 clips change on val, so val cannot resolve this — but it cannot confirm
it either, and an unconfirmed change does not ship.

### Rejected: mirror augmentation, stronger hand dropout, both

| recipe | val | test |
| --- | --: | --: |
| base (shipped recipe, retrained) | 81.74% | 66.13% |
| mirror | 82.61% | 68.37% |
| handdrop | 82.32% | 65.41% |
| mirror+handdrop | 81.74% | 67.89% |
| mirror+handdrop, tcn | **82.90%** | **70.62%** |
| mirror+handdrop, dropout 0.3 | 78.55% | 66.93% |

Every single model lands **below** the shipped ensemble (72.78%). Mirror
consistently helps the conv backbone (+2.2 test over base) but does not reach
the ensemble. The tcn backbone is the best single family.

### Rejected: the best new 4-member ensemble — +1.92 pts is selection noise

Searching every subset of 10 trained members and ranking by `min(val, test)`:

| ensemble | val | test |
| --- | --: | --: |
| shipped | 83.77% | 72.78% |
| tcn/s0 + tcn/s7 + base/s7 + drop3/s0 | **84.06%** | **74.70%** |

It wins on both splits, which no other candidate this session did — and it is
still rejected, for three measured reasons:

1. **Paired McNemar on test is z=1.99**, and val is `+14/-13 z=0.19`. Val moved
   by one clip. The 1.99 is the threshold, not a margin.
2. **The whole subset distribution is 0.35 SD wide** (30 subsets, 73.34–74.70%).
   The winner is not separated from the crowd; it is the top of a narrow pile.
3. **The gain is 1.56 standard errors** at n=1249 (SE 1.26). Under the 1.96 bar
   used everywhere else in this file, it is not a result.

The hypothesis test that settles it: TCN was a candidate *mechanism*, so it was
tested directly rather than through subset search. TCN-only ensembles score
71.58% and **lose** to shipped with `z=-1.16` (test) and `z=-0.49` (val). So
the 74.70% is the luckiest mix of nine, not a backbone effect.

### The real ceiling: ensemble size saturates, then hurts

Members ranked by mean accuracy, added cumulatively:

| members | val | test |
| --: | --: | --: |
| 1 | 82.90% | 70.62% |
| 2 | 82.61% | 73.66% |
| 3 | 82.90% | 73.90% |
| 5 | **84.64%** | **74.06%** |
| 6 | 84.64% | 73.26% |
| 8 | 84.64% | 72.70% |
| 10 | 84.64% | 72.86% |

Accuracy peaks at 5 members and then **falls below the shipped 4-member
ensemble**. Why: of 1,249 test clips, the union of clips any member gets wrong
is 669, and **182 are wrong for every single member**. The members fail on the
same clips, so adding them adds variance to the vote without adding
information. That correlated-error core is the binding constraint, not
capacity, not augmentation, not aggregation.

### What this closes

Every lever that a small-data ensembling problem normally offers has now been
measured and rejected on this corpus: more data (+0.24), capacity (−15),
frames (−8), speed perturbation (−5.45), strict resample (−8), mirror TTA
(−1.01), INT8 (−6.37), multi-crop TTA (0), mean-prob aggregation (0),
mirror/handdrop augmentation (0), and larger ensembles (negative).

The shipped 72.78% stands. The remaining 7.5-point oracle gap is not reachable
by combining these models; it needs either a different input representation or
signers whose data does not exist in ASL Citizen's 3,390 clips for these 100
glosses.
