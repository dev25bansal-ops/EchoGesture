# EchoGesture APK

Built 2026-11-20 from `submission/final` at commit `ebadd26`.

| | |
|---|---|
| File | `EchoGesture-2026-11-20.apk` |
| Size | 83,128,337 bytes (83.1 MB) |
| SHA-256 | `bb682468bb065233d8baa5e322907730eb03a8200a7ffad0a541389041fdac17` |
| Package | `org.echogesture.finalapp` |
| Min SDK | 24 (Android 7.0) |
| Target SDK | 34 |
| Arch | arm64-v8a |

The `.sha256` file beside it is the same digest in `sha256sum` format. Verify
before installing:

```
sha256sum -c EchoGesture-2026-11-20.apk.sha256
```

## What it does

Offline sign language recognition for 100 isolated ASL signs. No network
permission at all — the manifest removes INTERNET and ACCESS_NETWORK_STATE that
MediaPipe's transitive dependency injects, so the installed app cannot open a
socket.

Two ways to use it:

- **Start sign → perform → stop.** Recording ends by itself ~900 ms after your
  hand settles (`AutoFinishDetector`), or immediately if you tap Finish.
- **Pick a video clip.** Classifies a file directly, with no camera involved.
  This is the deterministic path and the one to use when checking the app works.

## Install

Debug-signed, so Android will warn about an unknown source. Enable "Install
unknown apps" for your file manager, or:

```
adb install -r EchoGesture-2026-11-20.apk
```

## Known limits, stated rather than buried

- **85.45% top-1** on 11 unseen signers (1,251 clips). That is corpus accuracy;
  live-camera accuracy is **not** measured and the few data points are worse.
- **The weakest signs are genuinely weak:** CHEESEGRATER 40%, SINK and SCARF2
  50%. Do not demo those.
- **Clip analysis takes about 1 s**, not the 10–20 s it used to. The fix was
  replacing per-frame seeking with a sequential `MediaExtractor` + `MediaCodec`
  decode (`SequentialVideoDecoder.kt`), measured at 8,700–21,200 ms before and
  ~870 ms after.
- **Two demo clips are training clips.** NOON and VOICE were seen during
  training; DARK and MECHANIC are held out. Only the held-out pair says anything
  about generalisation.
- **The acceptance gate declines rather than guesses** when the top-2 margin is
  under 0.15, and a held hand that is not signing still gets a named gloss about
  43% of the time. Both are documented in `docs/`.
- **A declined reading now looks different from an accepted one, everywhere.**
  The live pill, the card header and the score badge all follow the gate: an
  accepted reading reads "Recognised sign" in green with a check, a declined one
  reads "Uncertain reading" in amber with a warning triangle. Verified on device
  in both directions — EAT1 at 22.2% declined, MECHANIC1 at 89.3% accepted.

## Rights — read this before redistributing

The APK embeds the nine trained `.tflite` members, which are **ASL Citizen
derived and carry no redistribution permission**. `NOTICE` and
`docs/release/ARTIFACT_RIGHTS.md` mark them and this APK **BLOCKED for
publication**. They are in this public repository so the submitted system is
reproducible; they are **not licensed for commercial reuse**. The MediaPipe
detector (`hand_landmarker.task`) is Apache-2.0 and redistributable with
attribution. The app code is MIT.

## Rebuilding it yourself

`app/src/main/assets/` is gitignored, so a clean clone cannot assemble an APK
until the model files are put there:

```
cp models/*.tflite models/*.task models/*.json \
   app/android_final/app/src/main/assets/
cd app/android_final
JAVA_HOME=<jdk17> ./gradlew :app:assembleDebug
```

The asset directory needs `hand_landmarker.task`, the nine `*_fp16.tflite`
members, `landmark_labels.json` and `landmark_model_meta.json` — the same set
`results.csv` row `model_size` counts.