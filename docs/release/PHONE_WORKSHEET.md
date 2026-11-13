# Phone testing worksheet — NOT TESTED

Status: **no Android device has been attached to this machine.** Everything
below is a script for you to run, not a result. Any phone claim in the
submission must be traced back to a filled copy of this worksheet.

Companion documents: [ACCURACY_PROTOCOL.md](ACCURACY_PROTOCOL.md) (the
measurement method), `evaluation/device/accuracy_report.py` (turns the log into
the numbers).

## 1. Connect

On the phone: Settings → About phone → tap **Build number** 7 times → Developer
options → **USB debugging** on. Connect by USB, then:

```bash
adb devices -l
```

Expected: one line ending in `device`. `unauthorized` means the RSA prompt on
the phone has not been accepted. `offline` means the cable or port is bad.

Record the phone identity for the submission:

```bash
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.product.cpu.abi
```

## 2. Install

```bash
adb install -r dist/echogesture-v2-cc00f6a.apk
adb shell pm grant org.echogesture.finalapp android.permission.CAMERA
adb shell am start -n org.echogesture.finalapp/.MainActivity
```

If install fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, the phone holds a
build signed with a different key. Uninstall first:

```bash
adb uninstall org.echogesture.finalapp
```

## 3. Turn on diagnostics

In the app: **Settings → Capture diagnostics → on**. This deletes any previous
log and starts a fresh one. Nothing leaves the device; the app has no internet
permission.

## 4. Sign inventory

Fill this in *before* testing. Use the exact spelling from
`models/landmark_labels.json`.

| # | Official label | Source of the correct demonstration | Notes |
| --: | --- | --- | --- |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

Pick 10 labels from the 100 in `models/landmark_labels.json`. Include at least
one of the known-weak classes so the result is informative: **DOG1** (1/4
correct on validation), LOCK1, BEE1, CHEESEGRATER, FOREIGNER1, CHEEK.

## 5. Attempts

5 attempts per label = 50 captures. Vary conditions across attempts:

| Attempt | Background | Lighting | Distance |
| --- | --- | --- | --- |
| 1 | plain wall | daylight | ~60 cm |
| 2 | cluttered room | indoor lamp | ~60 cm |
| 3 | plain wall | indoor lamp | ~120 cm |
| 4 | cluttered room | daylight | ~120 cm |
| 5 | your usual setup | your usual setup | your usual setup |

Record every attempt in order. One line per capture, exactly as captured:

```
1 EAT1 plain daylight 60
2 EAT1 cluttered lamp 120
```

Save as `annotations.txt`. This file holds labels and conditions only — no
images, no names.

## 6. During capture

- Both hands inside the frame guide for the whole window; the status line counts
  3.7 s down.
- Tap **Recognise sign** once per attempt.
- If the app says *"not clear enough to read"*, that is a **failed capture**.
  Record it and move on — do not silently retry, because the retry rate is part
  of what is being measured.
- Note glare, hand leaving frame, or the phone being moved.

## 7. Retrieve and analyse

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
```

Then, on the machine with the repo:

```bash
python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

The script refuses to run if the two files disagree on capture count, rather
than guessing the pairing.

## 8. Fill in

| Metric | Value | Source |
| --- | --- | --- |
| top-1 accuracy | | `accuracy_report.py` |
| top-5 accuracy | | `accuracy_report.py` |
| rejection rate | | `accuracy_report.py` |
| accuracy at accepted coverage | | `accuracy_report.py` |
| capture-to-caption latency p50/p95 | | `adb logcat` + result card |
| failed captures | | outcome breakdown |
| per-sign worst class | | per-sign table |

## 9. Failure categories

Every failed attempt gets exactly one label. They imply different fixes, which
is the point of separating them:

| Category | Looks like | Implies |
| --- | --- | --- |
| Incomplete gesture | app reported a non-result, or coverage below 15% | capture/UX |
| Low detection coverage | `hand_frames` low while the hand was visible | detector or framing |
| Orientation/handedness | left and right swapped in the prediction's direction | camera path |
| Feature mismatch | the same clip predicts correctly in Python but not on the phone | pipeline bug |
| Model error | ≥40% coverage, correct performance, wrong label | model |
| Unsupported input | the sign is not one of the 100 labels | out of vocabulary — not a failure |

## 10. Privacy

The diagnostics file contains frame counts, timings, presence masks, the
predicted label, its score, top-5 and inference time. It contains **no images,
no landmarks, no signer identity**. Keep it and any video you record off Git.