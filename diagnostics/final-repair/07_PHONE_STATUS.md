# 07 — Phone status

## BLOCKED — no physical Android device was attached

```console
$ adb devices -l
List of devices attached
emulator-5554    device product:sdk_gphone64_arm64 model:sdk_gphone64_arm64 device:emu64a

$ system_profiler SPUSBDataType | grep -iE "Samsung|Galaxy|Android"
(no output)

$ adb reconnect offline
(no wireless target)
```

No USB device, no wireless ADB target, one emulator.

## What that means for the Phase 8 acceptance table

The brief asks for 15 attempts across five signs, three each, with expected
label, predicted label, correct/wrong/rejected, confidence, margin, hand
coverage, duration, auto-stop behaviour, inference time and analysed FPS.

**None of those rows exist.** They are not omitted for brevity; they were never
produced. An emulator whose camera is a synthetic colour pattern cannot supply
them.

## What WAS verified on the device that exists

| Check | Result |
| --- | --- |
| Fresh install, clean uninstall first | pass |
| Launch without crash | pass |
| Nine-model load (footer shows `995cc8d0…`) | pass |
| Deny permission → blocking card | pass |
| Grant permission → card clears, camera opens | pass |
| Grant externally → tap card → recovers, no crash | pass |
| 30 labelled clips through the real Android pipeline | 26/30 = 86.7% |
| 345 labelled clips through Python | 302/345 = 87.5% |
| JVM unit tests | 117 pass |
| Instrumentation tests | 19 pass, 0 fail |

The clip-replay result is a real recognition measurement: it decodes actual
video, runs the actual MediaPipe detector, builds the actual feature tensor and
runs the actual nine-member ensemble on the device. What it does **not** do is
take a frame from a phone camera.

## Untested and must be labelled so

| Item | Status |
| --- | --- |
| Real CameraX analysis rate | NOT MEASURED |
| Live hand-detection rate | NOT MEASURED |
| Path C — phone films a screen | NOT TESTED |
| Path D — person signs into the phone | NOT TESTED |
| Auto-stop | NOT IMPLEMENTED on this branch |
| Camera switching under load | NOT TESTED |
| Low light on a real sensor | NOT TESTED |
| Consecutive live recordings | NOT TESTED |
| Background / resume during a live capture | NOT TESTED |

## Honest reading of the user's report

The user reports "most signs still predicted incorrectly" on their phone. Against
that:

- the model scores 87.5% on 345 unseen-signer clips,
- the Android pipeline scores 86.7% on 30 clips on-device,
- the acceptance gate accepts 86% at 96% accuracy.

Those three are consistent with each other and none of them predicts "most signs
wrong". Something between "a person signs into this phone" and "the model
receives clean features" is therefore still unexplained.

The leading candidate is documented in `04_LIVE_RISK.md`: if the analyser runs
near 10 fps, a median 2.2 s sign yields ~20 frames, which both takes the
unresampled path **and** leaves only 3–6 of 24 model steps carrying a real
observation. That would degrade a working model into an unreliable one without
any single line being "wrong".

**That is a hypothesis, not a finding.** It needs the analyser-rate readout
from one real session to confirm or kill.

## What to do first with a phone in hand

1. Install `EchoGesture-Final-Fixed.apk`.
2. Settings → enable capture diagnostics.
3. Perform one known sign.
4. `adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt`

The `profile` line gives `fps=`, `interval_p50=`, `analyse_p50=`, `convert_p50=`,
`detect_p50=`, `delivered=`, `dropped=`. That single line decides whether
`04_LIVE_RISK.md` is the answer.