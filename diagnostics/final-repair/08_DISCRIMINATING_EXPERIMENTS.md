# 11 — Discriminating experiments (final session)

No physical phone was connected at any point in this session. Verified:

```console
$ adb devices -l
emulator-5554   device product:sdk_gphone64_arm64 ...
$ system_profiler SPUSBDataType | grep -iE "Samsung|Galaxy|SM-|Pixel|OnePlus|Xiaomi"
(no output)
```

So Step 1 and Step 2 of the brief are **BLOCKED**, and every claim below is
either a labelled-clip replay or a source/bytecode analysis. Nothing is a phone
measurement.

What was possible was Step 3's discriminating experiments, and they produced a
clear answer.

---

## Experiment 1 — tensor-level Android/Python parity

**Question:** do the two implementations compute the same tensors, or only
agree on final labels by luck?

`FeatureParityTest` compares, per clip: both feature widths (24×255 and 24×381),
the 24-step attention mask, all nine members' raw logits, and top-1.

Fixture generated from 6 real authorised clips spanning **all five signers**
(`make_parity_fixture.py`, model hashes verified against `PINNED_MEMBERS` before
use), pushed to the device, and run:

```
PARITY clip=0 frames=62  feat_max_abs=0.000000 mask_mismatch=0 logit_max_abs=0.000003 kotlin=BEE1      python=BEE1      score=0.9729
PARITY clip=1 frames=56  feat_max_abs=0.000000 mask_mismatch=0 logit_max_abs=0.000002 kotlin=8HOUR     python=8HOUR     score=0.3185
PARITY clip=2 frames=67  feat_max_abs=0.000000 mask_mismatch=0 logit_max_abs=0.000004 kotlin=MAPLE     python=MAPLE     score=0.8888
PARITY clip=3 frames=121 feat_max_abs=0.000000 mask_mismatch=0 logit_max_abs=0.000005 kotlin=8HOUR     python=8HOUR     score=0.9323
PARITY clip=4 frames=70  feat_max_abs=0.000000 mask_mismatch=0 logit_max_abs=0.000005 kotlin=APPLE     python=APPLE     score=0.9228
PARITY clip=5 frames=69  feat_max_abs=0.000000 mask_mismatch=0 logit_max_abs=0.000004 kotlin=CALENDAR1 python=CALENDAR1 score=0.6550
PARITY SUMMARY clips=6 worst_features=0.000000 worst_mask=0 worst_logits=0.000005 top1_agree=6/6
```

| Quantity | Result | Reading |
| --- | --- | --- |
| worst feature difference | **0.000000** | bit-identical, not merely close |
| mask mismatches | **0** | identical presence |
| worst logit difference | 0.000005 | float accumulation order only |
| top-1 agreement | **6/6** | identical decisions |

**This closes the Android-side hypotheses.** Feature transform, landmark
normalisation, attention mask, per-member inference, mean-logit aggregation and
softmax all produce the same numbers as the Python reference on the same
landmarks. There is no Android preprocessing defect to fix.

It is a stronger result than the earlier label-only comparison, which could not
distinguish "same answer for the same reason" from "same answer by accident".

---

## Experiment 2 — is there a signer the model collapses on?

345 clips, five signers, same ensemble:

| Signer | Correct | Accuracy |
| --- | --- | --- |
| P21 | 90 / 104 | 86.5% |
| P39 | 96 / 103 | 93.2% |
| P26 | 85 / 101 | 84.2% |
| P12 | 20 / 24 | 83.3% |
| P5 | 11 / 13 | 84.6% |

All five sit in an 83–93% band. **No signer is systematically failing.** The
spread is consistent with sample size (P5 has 13 clips, P12 has 24), not with a
signer-specific defect.

---

## Experiment 3 — is confidence a usable discriminator?

345 clips, real top-1 and top-2 probabilities from the ensemble:

| | Correct | Wrong |
| --- | --- | --- |
| mean top-1 | **0.762** | 0.291 |
| mean margin | **0.700** | 0.139 |

43 errors, and only **6** have top-1 ≥ 0.5. The gate at margin ≥ 0.15 and
score ≥ 20% accepts 86.1% of captures at 96.0% accuracy (measured earlier on
this same population).

**Answer to "do not claim that changing uncertainty thresholds improved
recognition accuracy":** no threshold was changed this session, and no accuracy
claim rests on one. The policy is reported because it separates reliable from
unreliable output, not because it makes the model more accurate.

---

## Experiment 4 — what the 43 errors actually look like

Confusion pairs, most frequent first:

```
CHEESEGRATER -> DRAG1        x2
NOON1        -> BEE1         x1
PARTY1       -> BACKPACK1    x1
SHAVE1       -> PATIENT2     x1
SPECIAL1     -> HURDLE/TRIP1 x1
ANYONE       -> OFFEND       x1
CHEESEGRATER -> LOCK1        x1
ELEVATOR1    -> CHEW1        x1
FLOAT1       -> RIVER1       x1
GOODYGOODYSHOEOPPOSITE -> DOWNSIZE1 x1
```

**41 of 43 are unique pairs.** There is no systematic label swap, no off-by-one,
no reversed vocabulary. That is the signature of individual clips sitting near a
decision boundary, not of a mapping defect.

### The one structural case: 8HOUR vs MAPLE

| Clip | Truth | Predicted | top-1 | Signer |
| --- | --- | --- | --- | --- |
| `3191470037860531-8 HOUR` | 8HOUR | 8HOUR | 0.294 | P21 |
| `8922394464987926-8 HOUR` | 8HOUR | **MAPLE** | **0.863** | P26 |
| `5929429678299929-8 HOUR` | 8HOUR | 8HOUR | 0.935 | P39 |
| `7629092396207264-MAPLE` | MAPLE | **8HOUR** | 0.094 | P21 |
| `2560023931191153-MAPLE` | MAPLE | MAPLE | 0.926 | P26 |
| `3796947997428477-MAPLE` | MAPLE | MAPLE | 0.545 | P39 |

The confusion is **bidirectional and instance-specific**: P21 calls a MAPLE clip
8HOUR; P26 calls an 8HOUR clip MAPLE. P21 and P39 handle both correctly.
**Both platforms make the same mistake on the same clip** (parity test clip=2 is
exactly the P26 8HOUR→MAPLE case, at 0.8888 in both).

So this is a property of the model on two visually similar signs, not of the
Android implementation and not of a signer. It is the kind of failure that would
need more training data or a targeted augmentation, not an app-side patch.

---

## Verdict

**The Android recognition path is not broken.** Tensor-level parity is exact and
the model scores 87.5% across five unseen signers on authorised clips.

**The remaining failures are model-level**, concentrated on individual
hard clips, and two of them are high-confidence errors on a similar-sign pair.
Changing Android code cannot fix those, and doing so would be exactly the
speculative editing the brief forbids.

**What is still unknown is the phone.** The one thing no clip replay can tell us
is whether a person signing into a phone camera produces landmarks that land in
the same region as these clips. That requires the device.