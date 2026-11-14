# V4.3 — model migration: nine-member masked ensemble

Branch `integration/v4-3-masked-ensemble`, base `ab10d54` (V4.2).
Dev source `hackathon/dev-model` @ `11bbddf`. Date 2026-11-14.

**Verdict: nine-member masked ensemble loads and runs. Android/Python parity
PASSES. Accuracy improvement MEASURED. Physical phone NOT TESTED.**

## 1. What changed in the contract

The V4.2 app asserted two things that are no longer true of Dev's artifacts:

```kotlin
require(interpreter.inputTensorCount == 1)                              // now 2
check(inputTensor.shape() == intArrayOf(1, 24, 255))                    // now 381 for 5 of 9
```

Both failed at member construction, before any inference. The app could not
start on the new bundle at all.

| | V4.2 | V4.3 |
| --- | --- | --- |
| Members | 4 | 9 |
| Inputs per member | 1 (`features`) | 2 (`features`, `mask`) |
| Feature widths | 255 only | 255 ×4, 381 ×5 |
| Mask | none | hand-presence mask at every member |
| Combined SHA-256 | `0e4c3927…` | `995cc8d0…` |

## 2. Implementation

### Inputs resolved by name, never position

The converter reports `serving_default_mask:0` as input **0**, not the features.
Positional code therefore feeds the mask where the features belong, and fails
only as a shape mismatch several layers down. Both the classifier and the
structural test now resolve `features` and `mask` by tensor name and assert they
are different tensors.

The test that originally indexed positionally caught this exact hazard when it
was updated — it read input 0 expecting `[1,24,255]` and got the mask's `[1,24]`.

### Attention mask

`FeatureTransform.build()` now returns `(features, mask)`. The mask is computed
**after** temporal resampling, exactly as `src/features.py` does:

```
mask[t] = 1 if present_0[t] > 0.5 or present_1[t] > 0.5 else 0
```

over the interpolated presence, not the original frames. Both feature widths are
built from one landmark pass, so the mask is identical for both and MediaPipe is
never run twice.

### 381-wide z path

`use_z` comes from `members_detail` in the sidecar, per member. A member with no
entry is a hard error — inferring width from a filename is the guess that feeds
a 381-wide model a 255-wide tensor.

The z path reads `landmarks[..., :3]` instead of `[..., :2]`, giving blocks of
126 (2 hands × 21 landmarks × 3 coords) with the same 3 inter-hand geometry
values appended: 126×3 + 3 = 381. Inter-hand geometry stays **x/y only** even in
the z path, which is what makes 126×3 + 3 rather than 126×3 + 4.

### A real defect found by parity

`src/features.py::_hand_scale` calls

```python
np.linalg.norm(hand_xy[MIDDLE_MCP] - hand_xy[WRIST])
```

on whatever array it was handed. In the z path that is a **3-vector**, so the
norm **includes z**. The docstring still says "one 21x2 hand"; the code is the
contract and the code uses three coordinates.

Kotlin used x,y only, which rescales every z-member feature by a different
factor per frame. Measured effect: **feature max abs diff 1.047** — and the
model still ran, still produced a plausible label and still agreed 6/6 on top-1.
Only a numeric comparison found it.

After the fix:

| Metric | Value |
| --- | --- |
| Feature max abs diff | **0.000000** |
| Mask mismatches (of 144) | **0** |
| Logit max abs diff | **5 × 10⁻⁶** |
| Top-1 agreement | **6/6** |

Test tolerances were tightened from 1e-4/0.05 to **1e-3/1e-2** so this class of
error cannot pass silently again. The old 0.05 logit tolerance *passed* the
1.047 feature defect.

## 3. Model identities

### New bundle (shipped)

| Member | SHA-256 |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `af031d898c98c97abcf66ae7538166316faac458b7b938fe9e3390fce31e5d4c` |
| `landmark_ens_tcnm61_fp16.tflite` | `b0a97aa45c6797b4b85e2f6a1b03945881a94dad5519cb71be47e474fdea54f0` |
| `landmark_ens_tcnm62_fp16.tflite` | `1d247397d57e982196e68f3f230adeeabbfcdf673ba01ed541eadd54d874ea8d` |
| `landmark_ens_rob71_fp16.tflite` | `105218a89fde1d107441422254805dfb38cc7df25a216ff986801bc15dccb89a` |
| `landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite` | `387749f81673719bd4a24813f33c6880c6117ed7be9d6bd50853250f8082901e` |
| `…_z_s1_fp16.tflite` | `02a2dc70fbd769f793c8f8915c8a30cc568d901bf2e35bffe1752aa762000a35` |
| `…_z_s2_fp16.tflite` | `3b97a2d416faec002bb84a6d2e47c131bb97941f074feed248a2c5a99c1c3b3d` |
| `…_z_s7_fp16.tflite` | `2f13d0eba1f34e68f3e3ac1b2c645d9f75d1953918d69be6d58ca00a1f2b36b2` |
| `…_z_s13_fp16.tflite` | `3d3a61e08d6a4d2a0a8064ae3b63692a55b55a2004ba37a428308a22b6336fba` |
| **Combined, meta order** | **`995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`** |
| Detector (unchanged) | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Vocabulary | 100 labels, unchanged |

### Rollback bundle (V4.2, preserved)

Extracted from `dist/echogesture-v4.2.apk` and hash-verified against the V4.2
contract: `112279bc…`, `01a261d8…`, `27058a85…`, `08417e72…`, combined
`0e4c3927…`. The V4.2 APK itself is untouched.

**Filenames are not identity.** The four legacy filenames hold completely
different binaries — `landmark_ens_tcn21_fp16.tflite` was `112279bc…` and is now
`af031d89…`. Anything keying on filename alone will silently load the wrong
model.

## 4. Two gated steps, as required

**Step A** — the four masked 255-wide members — works. Verified by
`FeatureParityTest` (mask parity, per-member logits) and by structural tests.

**Step B** — all nine including the five 381-wide z members — works. Verified the
same way, with per-member width read from the sidecar.

No partially loaded ensemble can ship: `ModelContract` fails hard if any listed
member is missing, unpinned, or has no `members_detail` entry, and
`LandmarkClassifier.create` closes every already-constructed member if a later
one throws.

## 5. Preserved from V4.2

Start → Sign → Finish recording, `SignRecorder`, `RecordingFeatures`,
`SignSegmenter`, timestamp-based sampling, offline operation, Camera UI,
Glossary, History, Diagnostics, confidence and retry behaviour. No UI redesign.

The segmenter's own doc comments were carried over unchanged; its behaviour is
re-evaluated under the new ensemble in `V4_3_MODEL_COMPARISON.md`.