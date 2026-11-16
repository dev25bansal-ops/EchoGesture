# Reproducing the same build — Dev and Abhijeet

Two people must be able to produce the **same APK** and the **same model
results** from the same source and the same authorised weights. This file is
the single procedure for both.

Everything here was executed on the branch it describes. Where a step could not
be run on a given machine, it says so instead of implying it passed.

---

## 0. What is in Git and what is not

| | In Git | Where it actually lives |
| --- | --- | --- |
| Source, tests, docs | **yes** | the branch |
| `models/landmark_labels.json`, `models/landmark_model_meta.json` | **yes** | the branch |
| The nine `.tflite` members | **no** | private shared storage |
| `hand_landmarker.task` (detector) | **no** | private shared storage |
| APKs | **no** | private shared storage |
| ASL Citizen clips, landmark caches | **no** | never in Git |

`*.tflite` and `*.apk` are gitignored on every branch where that rule exists.
If a clone is missing weights, that is the intended state, not a broken setup.

**Redistribution is not cleared.** See `NOTICE` and `models/README.md`. Keep
weights and APKs in private storage until written permission is recorded.

---

## 1. Get the source

```bash
git clone https://github.com/dev25bansal-ops/EchoGesture.git
cd EchoGesture
git checkout shared/v4.4-dev-integration
```

## 2. Supply the weights

Put these in `models/`. Eleven files.

```text
models/hand_landmarker.task
models/landmark_ens_tcn21_fp16.tflite
models/landmark_ens_tcnm61_fp16.tflite
models/landmark_ens_tcnm62_fp16.tflite
models/landmark_ens_rob71_fp16.tflite
models/landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite
models/landmark_ens_mirror_handdrop_tcn_z_s1_fp16.tflite
models/landmark_ens_mirror_handdrop_tcn_z_s2_fp16.tflite
models/landmark_ens_mirror_handdrop_tcn_z_s7_fp16.tflite
models/landmark_ens_mirror_handdrop_tcn_z_s13_fp16.tflite
```

### Verify every file before building

A wrong file produces a model that runs perfectly and predicts nonsense. The
evaluator refuses to load an unpinned binary, so this check is enforced, but
verify by hand too:

```bash
cd models
shasum -a 256 -c <<'EOF'
af031d898c98c97abcf66ae7538166316faac458b7b938fe9e3390fce31e5d4c  landmark_ens_tcn21_fp16.tflite
b0a97aa45c6797b4b85e2f6a1b03945881a94dad5519cb71be47e474fdea54f0  landmark_ens_tcnm61_fp16.tflite
1d247397d57e982196e68f3f230adeeabbfcdf673ba01ed541eadd54d874ea8d  landmark_ens_tcnm62_fp16.tflite
105218a89fde1d107441422254805dfb38cc7df25a216ff986801bc15dccb89a  landmark_ens_rob71_fp16.tflite
387749f81673719bd4a24813f33c6880c6117ed7be9d6bd50853250f8082901e  landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite
02a2dc70fbd769f793c8f8915c8a30cc568d901bf2e35bffe1752aa762000a35  landmark_ens_mirror_handdrop_tcn_z_s1_fp16.tflite
3b97a2d416faec002bb84a6d2e47c131bb97941f074feed248a2c5a99c1c3b3d  landmark_ens_mirror_handdrop_tcn_z_s2_fp16.tflite
2f13d0eba1f34e68f3e3ac1b2c645d9f75d1953918d69be6d58ca00a1f2b36b2  landmark_ens_mirror_handdrop_tcn_z_s7_fp16.tflite
3d3a61e08d6a4d2a0a8064ae3b63692a55b55a2004ba37a428308a22b6336fba  landmark_ens_mirror_handdrop_tcn_z_s13_fp16.tflite
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  hand_landmarker.task
EOF
```

**The identity that matters is the ordered hash**, over the nine members in
`members` order from `landmark_model_meta.json`:

```bash
cd models
for f in $(python3 -c "import json;print(' '.join(json.load(open('landmark_model_meta.json'))['members']))"); do cat "$f"; done | shasum -a 256
# expect: 995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb
```

Both branches produce this value from the same nine files. Verified.

---

## 3. Python evaluator — `predict(video_path)`

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r src/requirements.txt

PYTHONPATH=src .venv/bin/python -c "
from main import predict
print(predict('/path/to/clip.mp4'))
"
```

Expected shape — exactly five keys:

```json
{"clip_id": "...", "prediction": "DOWNSIZE1", "top5": [...], "confidence": 0.915, "latency_ms": 848.97}
```

### Requirements note

`src/requirements.txt` pins **`opencv-python-headless`**, not `opencv-python`.
The GUI build links `libxcb`/`libGL`, which `python:3.11-slim` does not ship,
and `predict()` only decodes video. A desktop install of the same version works
too; the headless pin is what makes the container reproducible.

### Failure behaviour you should expect

The evaluator refuses to run rather than guess:

| Condition | Result |
| --- | --- |
| Any member bytes differ from the pins | `ValueError`, names the hash |
| Member order changed | `ValueError`, ordered hash mismatch |
| A member listed in meta but absent on disk | `ValueError`, names the file |
| `members_detail` missing an entry | `ValueError` — width is never inferred from a filename |
| Model declares a class count ≠ vocabulary | `ValueError` |
| No hand detected in any sampled frame | `ValueError` — an empty clip is not a prediction |

To evaluate a deliberately unpinned candidate, set
`ECHOGESTURE_ALLOW_UNPINNED_MODELS=1`. Off by default: an unpinned binary is an
unverified binary.

---

## 4. Android APK

### Toolchain that was actually used

| Component | Version | Notes |
| --- | --- | --- |
| JDK | **17** (Temurin 17.0.21) | **Required.** AGP 8.5.2 rejects JDK 21+ with `26.0.1` as the whole error |
| Android SDK | platform android-34, build-tools 34.0.0 | |
| Gradle | 8.10.2 | wrapper handles it |
| Kotlin / AGP | 2.0.20 / 8.5.2 | |

### Assets must be present or the build refuses

```bash
mkdir -p app/android_final/app/src/main/assets
cp models/hand_landmarker.task \
   models/landmark_labels.json \
   models/landmark_model_meta.json \
   models/landmark_ens_*.tflite \
   app/android_final/app/src/main/assets/
```

All nine members, not four. The build validates the shipped set against
`ModelContract.kt`.

### SDK path

```bash
echo "sdk.dir=/absolute/path/to/android-sdk" > app/android_final/local.properties
```

Forward slashes. On Windows, backslashes fail AGP's `validateSdkPath` with a
misleading `java.io.IOException`.

### Build and test

```bash
export JAVA_HOME=/path/to/jdk-17
export ANDROID_HOME=/path/to/android-sdk
cd app/android_final
./gradlew clean testDebugUnitTest assembleDebug
```

Expected: **110 JVM tests, 0 failures.** Output at
`app/build/outputs/apk/debug/app-debug.apk`.

---

## 5. What "the same APK" means here

An APK is not byte-reproducible across machines (timestamps, signing, constant
pool ordering). What is reproducible is the **content**:

- same nine member hashes, verified above
- same `landmark_model_meta.json` and labels
- same source commit

Record the APK SHA-256 and the commit it came from every time you hand one over:

```bash
shasum -a 256 app/android_final/app/build/outputs/apk/debug/app-debug.apk
git rev-parse HEAD
```

**Do not commit the APK.** It embeds the weights.

---

## 6. Verification status of this procedure

| Step | Status |
| --- | --- |
| Weights hash to the pinned values | **VERIFIED** on both branches |
| Ordered ensemble hash `995cc8d0…` | **VERIFIED**, computed from Dev's committed files and from the pin |
| `predict()` loads nine members, correct `use_z` split | **VERIFIED** (`[F,F,F,F,T,T,T,T,T]`) |
| Release bundle builds with all nine + correct hashes | **VERIFIED** `status: structure_verified` |
| JVM tests | **VERIFIED** 110 / 0 failures |
| Python suite | **424 / 493 pass**; 26 fail for missing gitignored artifacts (see §7) |
| Android instrumentation | **11 / 11** at V4.4; not re-run on this branch |
| Physical phone | **NOT TESTED** — no device attached |

---

## 7. Known test failures and why they are not defects

`python3 -I scripts/run_all_tests.py` reports 3 failed + 23 errors on this
branch. **Identical on the untouched V4.4 tip** — the integration changed
nothing here. All 26 trace to three artifacts that are gitignored by design:

| Missing artifact | Tests affected | Why absent |
| --- | --- | --- |
| `evaluation/device/v5_accuracy/results/val_346/summary.json` | quantization_final | measurement output, gitignored |
| `models/landmark_classifier_fp16.tflite` | release_ensemble, release_final, quantization_final | restricted; correctly not in Git |
| `evaluation/device/v7_android/parity_chew2.npy` | feature contract | binary fixture, gitignored |

Provide the weights and the caches and these run. Until then they are
**blocked on assets**, not failing on code.

---

## 8. Numerical parity

`src/main.py` and `src/features.py` are byte-identical to the state that
produced the 9-clip parity fixture:

```
main.py       386ec80413384475277c7c240731a49af4782a12a85b90cc8e184aabdc793a29
features.py   ecf555fbfe5b70d09b5aeba92f95748338d64d10591abaac92cf57d1fd8bbba6
```

Recorded result at that state: feature max abs diff **0.000000**, mask
mismatches **0**, logit max abs diff **5e-6**, top-1 agreement **9/9** across
both feature widths and five signers.

To regenerate:

```bash
python evaluation/device/v4_3_parity/make_parity_fixture.py \
    --clips DIR --manifest manifest.json --out fixture.bin --limit 9
```

It refuses to write unless all nine member hashes and the ordered ensemble hash
match the pins.