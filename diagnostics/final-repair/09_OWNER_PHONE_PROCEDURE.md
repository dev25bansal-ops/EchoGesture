# 12 — Owner-operated phone diagnostic procedure

No phone is attached to the build machine, so Steps 1–2 of the brief could not be
run here. This is the exact procedure to run yourself. It is written to be
copy-pasteable and to produce the numbers that decide the remaining question.

## 0. Install

```bash
adb install -r ~/Desktop/EchoGesture-Final-Fixed/EchoGesture-Final-Fixed.apk
```

Expected SHA-256: `9aebf4cbd8f2722ba119e43761e1e40e39d2405d4836177c4bc9eabe7ca3fa40`

If `adb` is not on your PATH, use the full path:

```bash
/Users/abhijeetmiskin/AppData/ActionMesh-fork/android-toolchain/sdk/platform-tools/adb
```

## 1. Confirm the device and the build

```bash
adb devices -l
adb shell getprop ro.product.model
adb shell dumpsys package org.echogesture.finalapp | grep -E "versionName|lastUpdateTime"
adb shell dumpsys package org.echogesture.finalapp | grep "android.permission.CAMERA"
```

Expect a real model name (not `sdk_gphone64_arm64`) and
`android.permission.CAMERA: granted=true`.

## 2. Enable diagnostics

Open the app → **Settings** → scroll to **Privacy** → turn on
**Capture diagnostics**.

Verify:

```bash
adb shell run-as org.echogesture.finalapp cat shared_prefs/echogesture_prefs.xml
```

Expect `<boolean name="diagnostics" value="true" />`. If it is absent, the toggle
did not take and no record will be written.

## 3. Capture the five signs

For each of five supported signs, three attempts:

1. Frame both hands inside the guide, good light, arm's length.
2. Tap **Start sign** once.
3. Perform the sign at a natural pace.
4. Tap **Finish sign** when the sign is complete.

After all fifteen, dump the record:

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt
```

Also copy it off the device so it survives:

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > phone-diagnostics.txt
```

## 4. What each line means

**Capture line** — one per attempt:

```
frames=32 dur_ms=12061 fps=29.5 hand_frames=26 presence=11011011... pred=BEE1 score=0.9412 top5=... margin=0.7310 infer_ms=48 total_ms=...
```

| Field | What it tells you |
| --- | --- |
| `hand_frames` | Frames where MediaPipe found a hand. **This is the first thing to check.** |
| `presence` | One char per model step: `0` = no hand, `1` = left, `2` = right |
| `pred` / `score` / `margin` | What the model said and how sure it was |
| `fps` | Sampling density of the model window, **not** camera throughput |
| `infer_ms` | Inference time |

**Profile line** — one per attempt:

```
profile fps=29.4 interval_p50=34 analyse_p50=13 convert_p50=1 detect_p50=12 delivered=684 dropped=0 limit=CAMERA
```

| Field | What it tells you |
| --- | --- |
| `fps` / `interval_p50` | **The real analyser rate.** This is the number that matters |
| `analyse_p50` | Per-frame cost against the arrival interval |
| `dropped` | Frames the analyser never saw |
| `limit` | `CAMERA` = sensor is the constraint; `ANALYSER` = the app is too slow |

## 5. How to read the result

**If `hand_frames` is low** (say under 40% of `frames`) — the detector is losing
the hands. Check framing, lighting, and whether both hands are inside the guide.
This is the most likely failure and the most fixable by technique alone.

**If `limit=ANALYSER`** — the app is the bottleneck. Record `analyse_p50` and
`fps`; that points at CameraX or MediaPipe and is an app-side fix.

**If `limit=CAMERA` and `hand_frames` is healthy but predictions are wrong** —
the landmarks are arriving fine and the model is misreading them. Compare the
offending clip against the 345-clip results: if the confidence is low, the app
is correctly declining; if it is high, it is a model error of the same kind as
8HOUR→MAPLE.

**Send back:** the full `phone-diagnostics.txt` plus, for each attempt, the sign
you intended. With those fifteen rows I can tell which of the three cases above
you are in, and the fix follows directly.

## 6. Do not do these

- Do not enable diagnostics and then film the screen — that adds the recapture
  degradation on top of the live one and confounds both.
- Do not sign faster than natural to "make it work"; the velocity features are
  trained on natural signing.
- Do not treat a low `score` as proof the app is broken. 96% of accepted readings
  are correct; a low score is the app declining, which is working as designed.