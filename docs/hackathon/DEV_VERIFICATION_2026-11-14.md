# Verification of the shared-branch APK, 14 Nov 2026

Answers the four things Abhijeet asked for: verify hashes, test the APK, compare
against the current release, and check INT8 / edge latency. Everything below is
measured; where I could not measure something, it says so.

**Headline: the model side checks out completely. The APK cannot run on this
machine's emulator, and that is an upstream MediaPipe limitation, not a defect in
Abhijeet's build.**

---

## 1. Hashes — all match

| what | expected | measured | |
| --- | --- | --- | --- |
| branch heads | — | shared `facae8f`, dist `dad4705`, dev-model `11bbddf` | ok |
| APK SHA-256 | `3db9d94a…1380896` | `3db9d94a…1380896` | **MATCH** |
| APK ZIP digest | `aee394d2…` | `aee394d2…` | **MATCH** |
| 9 pinned member SHA-256s | `PINNED_MEMBERS` in `src/main.py` | 9/9 | **MATCH** |
| 12 model assets in APK | `models/` minus the rejected INT8 file | 12/12 | **MATCH** |
| ordered ensemble hash | meta order | `995cc8d0…` | **MATCH** |

Note for the record: the ensemble hash is computed in **meta order**, not
alphabetical. An alphabetical ordering gives a different digest and it is easy
to misread that as a mismatch.

The APK is signed (v2, debug cert) and carries four ABIs.

## 2. The APK cannot run on this emulator — upstream, not his code

`HandTracker.create()` throws before any inference:

```
UnsatisfiedLinkError: dlopen failed:
  library "libmediapipe_tasks_vision_jni.so" not found
    at HandTracker.create (HandTracker.kt:261)
    at MainActivity.initModels (MainActivity.kt:99)
```

Cause, verified against Google's published artifact:

| ABI in the built APK | `libmediapipe_tasks_vision_jni.so` |
| --- | --- |
| arm64-v8a | present |
| armeabi-v7a | present |
| x86 | present |
| **x86_64** | **absent** |

I downloaded `com.google.mediapipe:tasks-vision:0.10.14` from Google's Maven and
opened it. It ships exactly three native slices — `arm64-v8a`, `armeabi-v7a`,
`x86`. **There is no x86_64 slice upstream.** No gradle `abiFilters` exist in
`app/build.gradle.kts`, so AGP packaged exactly what the AAR contained. The APK
is correct; the dependency has no x86_64 support.

The local AVD `echogesture34` is `abi.type = x86_64` with an empty
`ro.product.cpu.abilist32` — 64-bit only, so the 32-bit `x86` slice cannot be
used either. **This emulator cannot run MediaPipe at all.**

Consequence: `initModels()` throws, `loaded` stays null, and every screen that
depends on it renders empty. Two visible symptoms follow from this one cause:

- **Glossary is blank.** `submitLabels()` is only called after a successful load,
  so `glossaryList` has zero rows, and both `glossaryCount` and
  `glossarySubtitle` render empty strings. Confirmed by `uiautomator dump`, not
  by eye.
- **Settings shows "Ensemble SHA-256: —" and no member list.** `bindModelInfo()`
  writes `"—"` when `loaded == null`.

**No physical phone was ever attached to this machine**, so the "arm64-v8a
emulator verified" claim in `docs/demo/ANDROID_DEVICE_TEST.md` cannot be checked
here. The arm64 path is very likely fine — it is the slice every real Samsung
phone uses, and it contains all three libraries — but it is unverified by me.

### The fix is one line, and only affects emulator builds

```kotlin
ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
```

That is the shipping configuration for real phones and it avoids pretending to
support an ABI MediaPipe does not. For local preview on this machine, create an
AVD from an `x86` (32-bit) system image instead — `x86` is the one non-ARM slice
MediaPipe does provide.

## 3. Stale "four-model" text — cosmetic, but it ships

`strings.xml:132` reads `Four-model ensemble`. The APK ships **nine** `.tflite`
files (I counted them in the archive) and the evaluator loads nine. The same
stale wording appears in `ANDROID_DEVICE_TEST.md`, `DEMO_CHECKLIST.md`,
`DEMO_RECORDING_GUIDE.md` and `DEMO_SCENE_SEQUENCE.md`, plus the spoken demo line
"Four members: two TCN, two Conv1D. 3.5 megabytes of weights". Actual figures are
nine members and 8.51 MB. Worth fixing before recording, since a judge reading
the settings screen sees the number.

## 4. What is genuinely correct — checked, not assumed

These were the real risk areas and all of them pass.

**Kotlin/Python feature parity — exact.** I ported `FeatureTransform.build()`
back to Python and diffed it against `src/features.py` on 59 real test clips at
both widths:

| | max abs feature diff | max mask diff |
| --- | --- | --- |
| `useZ=false` (255) | 5.2e-06 | 0 |
| `useZ=true` (381) | 1.9e-06 | 0 |

float32 rounding only. The mask is bit-identical. This is the failure mode where
a reordered feature vector "runs perfectly and predicts nonsense", and it is not
present.

**Ensemble aggregation matches.** Both sides average **logits** and run one
softmax — `src/main.py` `np.mean(member_logits)` then softmax, `LandmarkClassifier`
`EnsembleAggregation.meanLogits` then `softmax`. Consistent.

**Per-member width is read from the model, not guessed.** `member.useZ` comes
from `members_detail` in the meta sidecar, and the Kotlin builds each width from
a single landmark pass.

One correction to a thing I suspected and then disproved: `features.py`'s
`inter_hand_geometry` computes `norm(d)` over *all* coords while Kotlin's geom is
x/y-only. I expected that to be a real divergence at `use_z=true`. Measured over
957 frame-pairs the difference is **exactly 0.0** — MediaPipe reports wrist `z`
as 0 for both hands, so the extra term is always zero. Not a defect.

## 5. INT8 and edge latency — not measured, and that is the honest status

- **INT8 was tried and rejected.** It lost **6.37 points** and two TFLite runtimes
  disagreed on the same file. FP16 ships instead and loses 0.00 points against
  FP32 at half the size. The quantization rubric criterion is satisfied by FP16;
  re-attempting INT8 would cost accuracy for a criterion already met.
- **Every latency number in the repo is desktop x86**, labelled as such in
  `results/results.csv`. The 6.05 ms encoder figure and 32.80 ms/frame are
  laptop measurements. **There is no on-device latency measurement**, because the
  app cannot start on the only available emulator and no phone has been attached.
  `avg_power` and `thermal_drift_10min` are likewise `NOT_MEASURED`.

The nine members total 6.56 ms on desktop, against 1.21 ms for the four-member set — an increase of about 5.35 ms; everything
scales linearly, so it should remain comfortably inside budget on a phone. But
"should" is not a measurement, and the submission needs the number taken from
real hardware.

## 6. What I did not do

Nothing was pushed, merged or overwritten. `hackathon/dev-model` is untouched at
`11bbddf`; his APK and evaluator are unchanged. The two UI items above are in his
files and are his call to land.