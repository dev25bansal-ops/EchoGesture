# V3 model decision

**Verdict: MODEL CANDIDATE VALIDATED — the frozen FP16 ensemble stays.**

No fine-tuning, no re-conversion, no threshold change. The reason is not that
the model is proven good on a phone — it is not, and that evidence does not
exist. The reason is that **no valid phone input has been measured**, and the
rules for touching the model require it.

## The rule this decision follows

From the V3 directive: fine-tuning is justified only when *"real-phone data
shows repeated errors across signers with valid input"*. Low scores alone are
not a trigger. Neither is an emulator run. Neither is a suspicion.

None of those conditions is met:

| Condition | State |
| --- | --- |
| Real-phone data exists | **No.** `adb devices -l` shows only `emulator-5554` |
| Repeated errors observed | Not on phone |
| Across multiple signers | Only one signer has ever been recorded, and not on phone |
| With valid input (adequate coverage, in-span window) | The capture path only just became valid — defect A meant windows could be entirely post-sign |

Fine-tuning on anything available today would mean fitting the model to
*emulator-rendered* landmarks or to a single signer's training clips. Both are
the overfitting the directive explicitly forbids, and both would be justified
by a number that describes the pipeline defect rather than the model.

## Why the model is not the current blocker

The off-phone evidence already points away from the model:

- **97 of 100 genuine training clips** across the 22 previously-failing classes
  are classified correctly. No class fails on every clip.
- **Android and Python agree on the identical file** (EAT1, same runner-up),
  so the two pipelines are computing the same thing.
- **Rejection lifts 73.6% micro top-1 to 94.1% accuracy among the 64.2% of
  windows it accepts.** The gate is doing real work.
- The dominant remaining defect was **the capture window selecting the wrong
  seconds**, not the classifier scoring them wrong. That is now fixed and
  covered by 22 behavioural tests.

A model that scores 97/100 on genuine clips is not the thing to retrain while
the input path was, until this branch, capable of feeding it an empty room.

## What would change this decision

The decision flips to **MODEL_FINE_TUNING_REQUIRED** if and only if a phone run
shows:

1. **valid captures** — high `hand_frames`/`frames` ratio in `diagnostics.txt`,
   so the window really contained the sign; **and**
2. **consistent errors on the same signs** across attempts and across at least
   two signers; **and**
3. those errors are **not** explained by coverage, lighting or framing.

Condition 1 is why `accuracy_report.py` prints rejection rate and coverage
alongside accuracy: a run with low coverage is a pipeline finding, not a model
finding, and must be reported as such.

## Constraints preserved by this decision

| Constraint | State |
| --- | --- |
| Four FP16 members frozen | Yes — combined SHA-256 `0e4c3927…` unchanged |
| No hash or vocabulary change without an approved model version | No change made |
| INT8 not shipped | Still not shipped. Measured cost was 5.2 pt; FP16 stays |
| No tuning against the held-out test set | Nothing was tuned at all |
| No model weights uploaded | Nothing uploaded |

## Consequences for the submission

The Samsung submission still reports the model as an FP16 four-member ensemble
with a measured INT8 trade-off, exactly as before. This branch changes **no**
model artifact. What changed is the contract between the camera and the model:
the model now receives the 3.7 s that ended at the tap, with real timestamps and
a known coverage, instead of possibly receiving the seconds after it.