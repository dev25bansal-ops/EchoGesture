# Dev and Abhijeet handoff

## Branches and reviews

Abhijeet owns integration into `hackathon/two-person-sprint`. Work and push
from your own branch, then request the other teammate's review. See the
[GitHub workflow](GITHUB.md) for publishing the plan, branch commands, PR
targets and recovery steps.

| Files or proposed area | Primary editor | Boundary |
| --- | --- | --- |
| `src/features.py`, `models/`, `scripts/ml/` | Dev | Model interface, features, training and export |
| `src/main.py`, `app/`, `scripts/release/` | Abhijeet | Evaluator API, phone app and packaging |
| `evaluation/model/` | Dev | Data checks, unseen-signer model comparison and reproducible corruption fixtures |
| `evaluation/device/`, `results/` | Abhijeet | Full-path timings, end-to-end robustness and final submission outputs |
| `src/requirements.txt`, `src/README.md`, sprint docs | Abhijeet coordinates | Dev hands off dependency, model and reproduction changes for the same PR or a follow-up |

The proposed paths do not exist until the owner adds them. Agree on any
change that crosses these boundaries before editing. For interface changes,
merge Dev's model-side PR first, then have Abhijeet sync and integrate against
the exact artifact SHA. Abhijeet regenerates final results after the model
is pinned.

## Abhijeet → Dev · Evaluator inputs

Provide the starter-kit version, vocabulary and exact label spellings, allowed
sources, manifest shape, and unresolved organizer questions. For benchmarks,
include the model SHA, runtime and delegate, device, clip set and sample count,
end-to-end p95 and stage breakdown. Send any parity failures with a reproducible
fixture or a local clip reference both teammates can access.

## Dev → Abhijeet · Model bundle

| Field | Include |
| --- | --- |
| Candidate | Branch, commit SHA, artifact SHA-256 and bytes |
| Artifacts | TFLite file, exact ordered label map and versioned feature config |
| Input contract | Shape, dtype and scale/zero point, number of frames, sampling, coordinate convention, masks and normalization |
| Output contract | Class axis and whether values are logits or probabilities, plus dequantization parameters where applicable |
| Reproduction | Commands for feature extraction, training and conversion, plus one permitted smoke fixture and expected feature or inference shape |
| Evidence | Data version, split, clip and signer counts, float and INT8 validation top-1, known failure cases |
| Data scope | Sources used and what can be shared in the release |

A handoff is ready when Abhijeet can run the candidate from the documented
artifacts and verify feature and label parity. A change to label order or
features requires a new config version and a new benchmark of the actual
candidate.

## Daily update template

```text
Date (IST):
Owner and task ID:
Branch and commit SHA:
What changed:
Command(s) and result:
Dataset/split or device/runtime:
Artifact or config SHA:
Blocker or decision:
Next action and owner:
```

## Final release sign-off

- Dev verifies model bytes and format, vocabulary and split integrity, and
  reproducibility from the release bundle.
- Abhijeet verifies the actual evaluator API, manifest coverage, device and
  robustness logs, demo and deck, and final form requirements.
- Both agree which commit and artifact SHAs the submitted numbers describe.
  Abhijeet submits the package and keeps the receipt.

## Android V5.9 automatic camera response

Abhijeet built the installable debug APK from `48e02c28f60faa9e7b60a7b74869fa5f25b86186` on `codex/android-capture-fixes`. The app uses the pinned nine-member FP16 ensemble (`995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`), MediaPipe 0.10.14 and TensorFlow Lite 2.16.1 with four XNNPACK threads. The upgrade from V5.8 was verified on an offline Android 14 ARM64 emulator with both virtual cameras. The APK SHA-256 is `2842d2be9cfe816f35a8bdf5635c42112ec48ba02ad39e422a6e19a1d1e81b27`.

On 120 labeled portrait replays, V5.9 ranked the correct sign first on 106 clips and confirmed 54 correct captions. V5.8 ranked 98 correctly and confirmed 52 correct captions. Neither version confirmed an incorrect caption on this set. Across the 115 clips that yielded a label in both versions, V5.9 triggered inference a median 467 ms earlier. All 118 V5.9 endpoints in this replay followed hand disappearance. The one-tap UI test exercised the transition from Start to a visible result. A physical rear-camera check by the original signer remains open.

Build in `app/android_final` with `./gradlew :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest`. The local Desktop delivery folder `EchoGesture-V5.9` contains the APK, checksum, per-clip results, the 120 exact clip IDs and labels, and staging instructions. After staging the named clips and manifest, run the benchmark with:

```sh
adb shell am instrument -w -r -e requireResponse true \
  -e class org.echogesture.finalapp.PortraitRecognitionTest#automaticResponseOnLabeledPortraitClips \
  org.echogesture.finalapp.test/androidx.test.runner.AndroidJUnitRunner
```

All 152 JVM tests passed. Across the full instrumentation run and the corrected one-tap test rerun, all 51 distinct Android tests passed with fixtures present. Two earlier cancellation tests had used a stationary hand for 1.2 seconds, by which time V5.9 had completed recognition. Test commit `621c688` supplies sustained motion for cancellation and checks held-pose completion separately. The final APK's median feature and inference time was 24 ms (p95 30 ms) on scored clips. The older CameraX 1.3.4 initialization-recovery call still raises a RestrictedApi error in Android lint.

## Dev’s November 22 submission and Auto 6.0

The local delivery is in Desktop/EchoGesture-Final-2026-11-22. The evaluator and organizer baseline come from [Dev’s submission/final branch](https://github.com/dev25bansal-ops/EchoGesture/tree/submission/final) at a02248142a3a92e98759472a0063f7d579ddde38. The bundled Dockerfile.dockerignore makes the published Dockerfile runnable by including the pinned models and app/smoke_test.py. The extracted image ran offline and the local 30-clip check yielded 29 predictions, including 26 correct first matches. One clip had 12.5% hand coverage and triggered the 15% evaluator gate.

The [published reference APK](https://github.com/dev25bansal-ops/EchoGesture/releases/tag/v2026-11-22-submission) matches SHA-256 a56ba20f3b8a88c9e0be49c196a0ca29e66f0e69ad4e80d9bd4803dfa7e8ada7. The local Auto 6.0 APK was built from 3f53b1de4b5dec73f14588fec4a51b538cb4aba3 using assembleRelease. It uses the same pinned models and a dedicated signing identity and package ID so the apps can coexist. Its SHA-256 is 56a31080e22d52afe542f160c7592f3ef7db17670946f915f29eb7a20cf9239b.

Auto 6.0 keeps the V5.9 endpoint and confirmation policy and retries GPU tracking errors on CPU. The signed build imported WHATFOR1 successfully five times. In the 120-clip labeled emulator replay, 106 first matches were correct. The median endpoint was 467 ms earlier than V5.8 on 115 paired clips. Measurement conditions and raw traces are in Verification. Dev’s earlier Vivo V2404 debug build had 68–70 ms p95 with a hand, 2,666–2,898 mW attributed to the camera subsystem, 19.2% thermal drift and 5.76-point robustness loss for 25% hand dropout. Auto 6.0 phone metrics require a new physical run. The deck links to the included local 2:14 walkthrough. Organizer-form prerequisites include the exact registered team string and a shareable demo URL.

## Auto 6.1 rear framing and local video imports

The signed Auto 6.1 APK comes from 61d708bdd7cf612339f6247c0976a8620703f7fe. Its SHA-256 is 07cd81f2f701d4c6eee199c96b3795857b72013e8066da6fda6dc34f87a54e23. The complete package is in Desktop/EchoGesture-Final-6.1. It upgrades Auto 6.0. The organizer package retains the evaluator and FP16 weights pinned by Dev’s submission/final revision a02248142a3a92e98759472a0063f7d579ddde38.

On Android 14 ARM64, the paired 30-clip video import benchmark measured median 1,197 to 908 ms and p95 1,552 to 1,014 ms at identical output and confirmation scores. Across three imports per signed build, a rotation-tagged WHATFOR1 clip took median 5,550 ms on Auto 6.0 and 1,002 ms on Auto 6.1. The rear framing replay placed each of the 30 clips in the center of a portrait frame, testing 1× and 1.5×. Correct first matches improved from 23 to 26, and correct confirmed results improved from nine to 11. The signed app switched rear zoom levels and hid the control when switching to the front lens.

Build with ./gradlew :app:assembleRelease :app:testReleaseUnitTest in app/android_final. Reproduce the Android import measurements with ClipCorpusTest using requireCorpus and compareImport arguments set to true. Reproduce the framing replay with PortraitRecognitionTest#compareWideAndCloseFramingForDistantSigners and requireRearFraming set to true. Stage the 30 manifest clips in files/file-corpus. The package carries the per-clip CSVs and the six rubric rows. The prior Vivo results and the new emulator results represent different test conditions. Physical Auto 6.1 latency, rear accuracy, power, and thermal checks remain open.
