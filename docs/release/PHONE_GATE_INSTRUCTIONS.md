# Live phone recognition gate — what you must run

No physical Android device is attached to this machine (`adb devices -l` shows
only `emulator-5554`). Nothing in this file is a measurement. It is the exact
sequence that produces one.

**APK:** `dist/echogesture-v3-87efd29.apk`
SHA-256 `ce83cf4c0e080b210f3b3cb160d650177cf759f29e190b8951730dcbbdda8aa4`
Commit `87efd29`, branch `diag/live-phone-recognition-v3`, 77,716,903 bytes.

This is the **strict pre-tap capture contract** build. It differs from the
previous candidate (`echogesture-rc-a2c0a6c.apk`) in the capture path, not the
models: the window is the 3.7 s ending at your tap, there is no waiting after
the tap, and a thin or stalled buffer is refused with a message instead of being
classified anyway. See `PHONE_BUILD_IDENTITY.md` for the full build list.

---

## 1. Connect

Phone: Settings → About phone → tap **Build number** 7× → Developer options →
**USB debugging** on. Connect by USB, accept the RSA prompt.

```bash
adb devices -l
```

Requires one line ending `device` with a real model name (not `emulator-`).
Record it — it goes in the submission:

```bash
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.product.cpu.abi
```

## 2. Install

```bash
adb install -r dist/echogesture-v3-87efd29.apk
```

If it fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, a differently-signed
build is installed:

```bash
adb uninstall org.echogesture.finalapp
```

Then reinstall. Grant camera on first launch, or:

```bash
adb shell pm grant org.echogesture.finalapp android.permission.CAMERA
adb shell am start -n org.echogesture.finalapp/.MainActivity
```

## 3. Enable diagnostics

In the app: **Settings → Capture diagnostics → on**. This deletes any previous
log and starts fresh. It writes no images, no landmarks and no identifiers, and
the app has no internet permission.

## 4. Pick 10 signs

Draw from the 100 in `models/landmark_labels.json`. Use the exact spelling.
Include at least one known-hard class so the result is informative:
**BACKPACK1, KNIGHT1, MYSELF, SPECIAL1, SQUEEZE** are the ones that failed on
their validation clip but pass on training clips.

| # | Label | Demonstration source |
| --: | --- | --- |
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |
| 7 | | |
| 8 | | |
| 9 | | |
| 10 | | |

Perform each from an authoritative demonstration. Improvising a sign from a
written gloss measures nothing.

## 5. Five attempts per sign, varying conditions

| Attempt | Background | Lighting | Distance |
| --- | --- | --- | --- |
| 1 | plain wall | daylight | ~60 cm |
| 2 | cluttered room | indoor lamp | ~60 cm |
| 3 | plain wall | indoor lamp | ~120 cm |
| 4 | cluttered room | daylight | ~120 cm |
| 5 | your usual setup | your usual setup | your usual setup |

**Keep every failed capture.** A capture that reports "not clear enough to read"
is a data point, not something to retry away. The retry rate is part of what is
being measured.

Three refusals are now possible and mean different things — record which one you
saw, because they diagnose different problems:

| Message | Meaning |
| --- | --- |
| "Not enough of the sign was recorded yet…" | The buffer was cold. Wait a second after opening the camera. |
| "The camera stopped sending frames during that sign…" | The analyser stalled mid-sign. Note it; this is a device finding. |
| "That sign was just read…" | You tapped twice inside 1.2 s. Space the attempts out. |

Keep an annotations file, one line per capture, in capture order:

```
1 EAT1 plain daylight 60
2 EAT1 cluttered lamp 120
```

## 6. Retrieve and analyse

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

The script refuses to run if the two files disagree on capture count rather than
guessing the pairing. It prints top-1, top-5, rejection rate, accuracy at
accepted coverage, a confusion matrix and per-sign recall.

For per-class and margin detail on a corpus, use
`evaluation/device/per_class_check.py --split <csv> --clips <dir>`.

## 7. What your result will decide

The off-phone evidence is already complete and points one way:

- Android and Python agree on the identical file (EAT1 both).
- 97 of 100 genuine training clips across the previously-failing classes are
  correct.
- No class fails on every clip.
- Rejection lifts 73.6% micro top-1 to **94.1% accuracy among the 64.2% it
  accepts**.

So the decision table is:

| Phone result | Verdict | Next action |
| --- | --- | --- |
| High coverage, accurate | **FROZEN MODEL SUFFICIENT** | Ship it |
| Low `hand_frames`/`frames` | Capture problem | Camera/framing/lighting work, not the model |
| High coverage, consistently wrong signs | **PHONE-DOMAIN FINE-TUNING JUSTIFIED** | Phased 4 adaptation plan, signers held out |
| Only some signers fail | Signer-domain shift | Same as above, but per-signer reporting first |

I cannot choose between these without your data. Guessing would be the one
thing that makes this submission dishonest.

## 8. Do not

- Do not report a paused sign image on a screen as a live gesture.
- Do not claim phone accuracy from emulator results.
- Do not retry a failed capture silently.
- Do not upload video containing bystanders without consent.
- Keep `diagnostics.txt` and any recordings out of Git.