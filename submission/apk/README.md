# EchoGesture APK

**Ship this one: `EchoGesture-Auto-6.2.apk`.** It is a signed, non-debuggable
release build and it is the app the submission is scored on.

| | Auto 6.2 (ships) | Auto 6.1 | Release (superseded) | Debug (historical) |
|---|---|---|---|---|
| File | `EchoGesture-Auto-6.2.apk` | `EchoGesture-Auto-6.1.apk` | `EchoGesture-2026-11-22-release.apk` | `EchoGesture-2026-11-21.apk` |
| Size | 81,281,198 bytes | — | 80,816,048 bytes | 82,894,510 bytes |
| SHA-256 | `b1c0e6710e24352dc24a9ebe930527cb7667f67abca905915ea8c49a9ac2dcf8` | `07cd81f2f701d4c6eee199c96b3795857b72013e8066da6fda6dc34f87a54e23` | `9930894280967c0809b52595b103e0a36b93097768f7f61e401b7e50a3b4b397` | `96f00458f54de715c7a6f489f9ae92425bc73a089043a91ccbe0ae6d4f060035` |
| Package | `org.echogesture.finalapp.auto` | same | `org.echogesture.finalapp` | same |
| Version | vc 9 / 6.2 | vc 8 / 6.1 | vc 1 / 1.0 | vc 1 / 1.0 |
| Signature | `CN=EchoGesture Auto` | same | `CN=measurement` | debug key |
| `android:debuggable` | **absent** | absent | absent | `true` |

The Auto variants carry their own package ID and signing identity precisely so
they install **alongside** the older builds rather than replacing them. That is
why both can be on the phone at once.

## Where it comes from

Auto 6.2 is built from branch **`codex/android-auto-6.2`**, commit
**`33e82b5dafff08df7cfa92088f81aacc2418af8d`**. That commit is not just
claimed — `docs/hackathon/HANDOFF.md` on the branch records it beside this exact
SHA-256, and the build itself refuses to package unless the model bytes match:

`app/android_final/app/build.gradle.kts` defines a `verifyPinnedAssets` task
that hard-fails `assembleRelease` when any of the nine `*_fp16.tflite` members
or `hand_landmarker.task` is missing or hashes differently from its pin. All ten
pins plus `landmark_labels.json` were re-verified against this repository's
`models/` on 2026-11-22. **A swapped classifier cannot get past the build.**

## Install

Signed with a local key, not a Play Store certificate, so Android warns about an
unknown source. Enable "Install unknown apps" for your file manager, or:

```
adb install -r EchoGesture-Auto-6.2.apk
```

Verify the bytes first:

```
sha256sum -c EchoGesture-Auto-6.2.apk.sha256
```

**The 81 MB binary is published as a GitHub release asset, not tracked in git** —
an 80 MB blob repeatedly timed out the HTTPS push (HTTP 408), so the canonical
copy lives at:

https://github.com/dev25bansal-ops/EchoGesture/releases/tag/v2026-11-22-auto6.2

The `.sha256` file in the tree pins the bytes you should get. GitHub's own
server-side digest for the uploaded asset matches this hash exactly, so the
release asset and the tree agree.

## What it does

Offline sign language recognition for 100 isolated ASL signs. No network
permission at all — the manifest removes INTERNET and ACCESS_NETWORK_STATE that
MediaPipe's transitive dependency injects, so the installed app cannot open a
socket. The Settings screen states this and the idle screen carries a green
**Offline** badge.

Three ways to use it:

- **One tap.** Tap **Start Recognition**, perform one sign, and the app finds
  the sign's own start and end and shows the result — no Finish tap. This is the
  Auto variant's headline change.
- **Pick a video clip.** Classifies a file directly, with no camera involved.
  This is the deterministic path and the one to use when checking it works.
- **Speak.** A confirmed sign can be played aloud from 100 bundled
  pronunciations, entirely offline.

## Known limits, stated rather than buried

- **85.45% top-1** on 11 unseen signers (1,251 clips). That is **corpus
  accuracy**, not phone accuracy, and it is unchanged by this build.
- **Latency fails on this build, measured on the path a user takes.** Nineteen
  live-camera sessions with a hand in frame gave per-frame detect p95 of
  **72–100 ms (median 85)** against a 50 ms target — over in **19 of 19**.
  Analysis consumed 0.94–1.02× the inter-frame interval, so the analyser
  discarded frames the camera delivered (2,135 delivered, 863 processed). The
  detector is the bottleneck; convert never exceeded 50 ms and the classifier is
  54–211 ms per clip, once. See
  `docs/evidence/LATENCY_AUTO62_LIVE_CAMERA_2026-11-22.md`.
- **Robustness fails unconditionally.** Landmark `drop_hand_25pct` costs
  **+5.76 pt**. The app's margin ≥ 0.15 gate recovers it to +2.39 at 86% clean /
  80% corrupted coverage, and whether the competition permits abstentions is
  unknown.
- **Power fails at the camera hardware floor.** `batterystats` prices
  camera-on at a fixed PowerProfile constant of 749.4 mA (~2.9 W), roughly 5.8×
  the 500 mW budget, so **no** live-camera app on this device can pass that row.
- **A held hand that is not a sign still gets a name.** A closed fist held at
  the chin is reported as RAZOR2 at 0.794 with 9 of 9 members agreeing, because
  that is what a fist looks like in this lexicon. The gate is the only defence.
- **Two demo clips are training clips.** NOON and VOICE were seen during
  training; DARK and MECHANIC are held out. Only the held-out pair says anything
  about generalisation.
- **A declined reading looks different from an accepted one, everywhere.** An
  accepted reading reads **"RECOGNISED SIGN"** in green; a low-margin one reads
  **"POSSIBLE MATCH"** in amber with *"Low confidence. Compare the options below
  or try the sign on its own."*; and a capture with no hand reads **"No hands in
  view"** with *"No sign started. Tap Start Recognition and perform one sign."*
  A refusal is not recorded to History.

## Rights — read this before redistributing

The APK embeds the nine trained `.tflite` members, which are **ASL Citizen
derived and carry no redistribution permission**. `NOTICE` and
`docs/release/ARTIFACT_RIGHTS.md` mark them and this APK **BLOCKED for
publication**. They are in this public repository so the submitted system is
reproducible; they are **not licensed for commercial reuse**. The MediaPipe
detector (`hand_landmarker.task`) is Apache-2.0 and redistributable with
attribution. The 100 pronunciation MP3s are bundled app assets. The app code is
MIT.

## Rebuilding it yourself

`app/src/main/assets/` is gitignored, so a clean clone cannot assemble an APK
until the model files are put there. The build then verifies them against the
pins:

```
cp models/*.tflite models/*.task models/*.json \
   app/android_final/app/src/main/assets/
cd app/android_final
JAVA_HOME=<jdk17> ./gradlew :app:assembleRelease
```

The asset directory needs `hand_landmarker.task`, the nine `*_fp16.tflite`
members, `landmark_labels.json` and `landmark_model_meta.json` — the same set
`results.csv` row `model_size` counts. Signing is opt-in through
`ECHOGESTURE_KEYSTORE`; with it unset the release variant stays unsigned.