# On-device latency, EchoGesture Auto 6.2, live camera with a hand in frame

**Measured 2026-11-22 11:53–11:58 IST**, on the shipping APK, by the app's own
`AnalysisProfiler`, exported through Settings → Export diagnostics.

This is the measurement the release build never had. Every previous on-device
number in this repository came from either the **file-pick** path or a
**debug** build; `RESULTS_SUMMARY.md` said so explicitly: *"Still unmeasured:
the live camera with a hand in frame on the release build."* It is measured now,
on the app that ships.

## The artifact

| | |
| --- | --- |
| File | `EchoGesture-Auto-6.2.apk` |
| SHA-256 | `b1c0e6710e24352dc24a9ebe930527cb7667f67abca905915ea8c49a9ac2dcf8` |
| Package | `org.echogesture.finalapp.auto` |
| versionCode / versionName | 9 / 6.2 |
| Source commit | `33e82b5dafff08df7cfa92088f81aacc2418af8d` on `codex/android-auto-6.2` |
| Signer | `CN=EchoGesture Auto`, cert SHA-256 `984cfbaeedb31f08080d525e75183d6b45c9f749a694453c33336eb24e5cde70` |

Device: Vivo V2404, Android 16, arm64-v8a, SM7550, 1080×2392.
Delegate: GPU (`Delegate.GPU`, MediaPipe 0.10.14, TFLite 2.16.1, four XNNPACK threads).
Ensemble on screen: nine FP16 members, SHA-256 `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`.

## Sample

**19 live-camera sessions**, each carrying one `profile` line written by
`AnalysisProfiler` when the attempt ended. Every session had a real hand in
frame during the analysed span (`hand_frames` 5–91 of 17–91 frames). Four
additional `source=clip` records from the file-pick path are in the same export
and are reported separately below.

The raw export is committed beside this file as
`auto6.2_diagnostics_live_camera_2026-11-22.txt`.

## Result: every per-frame p95 is over the 50 ms target

| Stage, per frame | min | p50 | max | over 50 ms |
| --- | --: | --: | --: | --- |
| **detect p95** | **72** | **85** | **100** | **19 / 19** |
| detect p50 | 40 | 57 | 76 | 16 / 19 |
| convert p50 (YUV→Bitmap) | 14 | 28 | 37 | 0 / 19 |
| analyse p50 (convert+detect) | 54 | 83 | 105 | 19 / 19 |
| arrival interval p50 | 56 | 86 | 107 | — |

Every one of the 19 detect p95 samples:

```
72  75  77  77  80  82  84  84  84  85  86  86  89  92  92  93  93  97  100
```

**The detector — not the classifier — is the cost, and it is 22 to 50 ms over
the per-frame target.** The nine-member classifier on the same runs took
**54–211 ms, median 91 ms**, per whole clip rather than per frame.

## The finding that matters more than the number

`analyse_p50 / interval_p50` is **0.94 to 1.02, median 0.98**, and
`limit=ANALYSER` in **19 of 19** sessions.

Analysis occupies essentially the entire interval between frames arriving, so
the camera's frames are discarded while the analyser is still busy. Across the
19 sessions the camera **delivered 2,135 frames and the analyser processed 863
— 40.4%**. The drop is not a camera limit and not a decoder limit:

- `convert_p50` is 14–37 ms and never once exceeded 50 ms, so the YUV→Bitmap
  copy is not the problem.
- The camera delivered frames on a 56–107 ms cadence, so it had frames to give.
- Detection consumed 72–100 ms p95 of every interval.

**This attributes the failure to a single stage with evidence rather than
assumption.** The correct optimisation target is MediaPipe `detectForVideo` on
the GPU delegate, and no amount of classifier work will move this row: the
classifier is 91 ms per clip, once.

## What this means for the scored row

`results/results.csv` row `latency_p95` is **FAIL**, and this measurement makes
it fail on better evidence than before:

| Build and path | p50 | p95 | Verdict |
| --- | --: | --: | --- |
| DEBUG, CPU delegate | 95 | 166 | FAIL |
| DEBUG, GPU, no hand in frame (before the 640 pin) | 34–45 | 46–54 | at target, not the shipping config |
| DEBUG, GPU, hand in frame, diagnostics on | 50–58 | 68–70 | FAIL |
| SIGNED RELEASE, GPU, file-pick (4 runs) | — | 58 / 65 / 67 / 72 | FAIL, 8–22 ms over |
| **SIGNED AUTO 6.2, GPU, LIVE CAMERA, hand in frame (19 sessions)** | **57** | **72–100, median 85** | **FAIL, 22–50 ms over** |

The live-camera number is **worse** than the file-pick number, which is the
opposite of what a reader would assume from the old table, and it is the path
a user actually takes.

## What this does not measure

- **No accuracy claim.** One device, one signer, an unscripted session. In it,
  **7 of 18** predictions with a score reached the 0.5 confirmation floor. That
  is a description of one sitting, not a measured accuracy, and it does not
  replace the 85.45% corpus figure.
- **No power or thermal measurement.** Both need a separate run.
- **End reasons are capture behaviour, not latency:** 17 `HAND_LOST`,
  1 `QUIET`, 1 `TIMEOUT`. The `TIMEOUT` session (91 frames, no prediction) is
  the one case where the capture ran long enough to be abandoned.

## Reproducing it

1. Install `EchoGesture-Auto-6.2.apk`.
2. Settings → **Capture diagnostics** on, **Keep history on this device** on.
3. Settings → **Export diagnostics…**, share the `.txt` off the device.
4. Each `profile` line is one session; `detect_p95` is the per-frame figure the
   criterion scores.

The app writes these numbers itself, so the export cannot be produced by
anything but a real run on the device.
