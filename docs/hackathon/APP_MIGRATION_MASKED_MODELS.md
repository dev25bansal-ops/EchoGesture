# The app must change to use the new models

**This is a breaking interface change. `LandmarkClassifier.kt` will fail its
own preconditions at startup with the current `models/`.**

Written 13 Nov 2026 by Dev. `src/main.py` and the exported artifacts are done
and verified; the Kotlin side is Abhijeet's and is not touched here.

## What now fails, and where

`LandmarkClassifier.kt` asserts two things that are no longer true:

```kotlin
require(interpreter.inputTensorCount == 1) { ... }          // now 2
check(inputTensor.shape().contentEquals(intArrayOf(1, 24, 255))) { ... }  // now 381 for 5 of 9
```

Both fail *loudly*, at member construction, before any inference. That is the
good outcome — the alternative would have been silently scoring a different
model. But it does mean the app will not start until this is addressed.

## Why the models changed

The classifier was **always trained** with a hand-presence mask applied to its
attention: `train.TemporalModel` takes `key_padding_mask`, and `train.py`
always supplies it, so frames with no detected hand get zero attention weight.
The export never declared a mask input, so every artifact shipped before today
was a *different model* from the one trained — it attended over absent-hand
frames with `present=0` and all-zero coordinates.

Supplying the mask is worth, measured on one checkpoint with identical weights:

| | val | test |
| --- | --: | --: |
| no mask | 80.87% | 70.30% |
| with mask | 83.77% | 72.14% |

The second change is the z channel: five of the nine members take 381-wide
x,y,z features instead of 255-wide x,y. That is a different input
representation, and it is recorded per member in the meta sidecar.

Together, on the full ensemble and re-verified by reloading every `.tflite`
from disk: **87.25% val / 77.02% test**, up from 83.77% / 72.78%.

## What needs to change in the app

### 1. Accept two input tensors, resolved by name

```kotlin
require(interpreter.inputTensorCount == 2) {
    "member '$name': expected 2 inputs (features + mask), got ${interpreter.inputTensorCount}"
}

// Match on the NAME. The converter does not preserve declaration order -
// the mask is reported as input 0. Indexing positionally feeds the mask as
// features, and that fails only as a shape mismatch several layers deeper.
val inputs = (0 until interpreter.inputTensorCount).map { interpreter.getInputTensor(it) }
val featuresTensor = inputs.first { it.name().contains("features") }
val maskTensor     = inputs.first { it.name().contains("mask") }
```

### 2. Read the feature width per member instead of hardcoding 255

```kotlin
// From landmark_model_meta.json -> members_detail[]: {"file": ..., "use_z": true}
// A member with no entry, or use_z false, is 255-wide.
val featureDim = if (memberUseZ[name] == true) 381 else 255
check(featuresTensor.shape().contentEquals(intArrayOf(1, 24, featureDim))) { ... }
```

`ModelContract.kt` has `EXPECTED_FRAMES`/`EXPECTED_FEATURES` style constants;
`FEATURE_DIM` will need to become a per-member value rather than a single one.

### 3. Build features once per width, and feed the mask

`FeatureTransform.buildFeatures` currently returns the 255-wide matrix. It
needs a `useZ` flag whose `true` path appends the z coordinates exactly as
`features.py` does — the same layout, in the same order, or the member gets
garbage. **This is the part to be most careful with**: the feature vector is a
contract, and a reordering produces a model that runs perfectly and predicts
nonsense.

`buildFeatures` already returns a `mask` as its second value; that mask is what
now goes into `maskTensor`. Only extract landmarks once, then build the two
widths from them — MediaPipe is the expensive part.

## Suggested order

1. Two-input plumbing + name matching, still 255-only, against the **four
   legacy members** — this alone should take the app from 72.78% to 74.06% and
   proves the mask path works before width is introduced.
2. Per-member width and the `useZ` branch of `FeatureTransform`.
3. Re-run the on-device parity test; all nine members should agree with the
   Python reference to within FP16 rounding.

Step 1 is worth doing even if step 2 is not ready: it is a smaller change and
captures most of the gain from the mask alone.

## What is verified, and what is not

**Verified on the Python side:** every `.tflite` reloaded from `models/`,
`src/main.py` driven through its real wiring, and three real clips predicted
end to end (SAIL2 0.877, PULLCONVINCE 0.660, PONDER 0.254 — all correct).

**Not verified:** anything on Android. The app has not been built or run
against these models. The latency figure in the handoff (1.21 ms p50) was
measured on four members and has not been re-timed for nine.
