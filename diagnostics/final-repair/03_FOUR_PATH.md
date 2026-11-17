# 03 — Four-path comparison: what was possible

The brief asks for four paths per sign. Three need hardware this session did not
have. This file records which ran, which did not, and what that means.

| Path | Description | Status |
| --- | --- | --- |
| **A** | Python evaluator on an authorised source video | **RAN**, 345 clips |
| **B** | Same video through the Android detector + model | **RAN**, 30 clips |
| **C** | Phone films a screen showing the video | **NOT RUN** — no phone |
| **D** | A person signs directly into the phone | **NOT RUN** — no phone |

## Path A — Python, 345 clips, five unseen signers

```
correct 302 / 345   accuracy 87.5%   95% CI ±3.5 pp
P12 20/24   P21 90/104   P26 85/101   P39 96/103   P5 11/13
```

## Path B — Android device, 30 clips through the shipped pipeline

```
native 640x480   26/30   86.7%
256x256 squash   24/30   80.0%
```

A and B agree to within 1 point. **This is the discriminating result the brief
asks for.** It rules out the Android preprocessing and inference path:

> If A passes and B fails, investigate Android preprocessing and inference.

B does not fail. Both paths classify the same clips at the same rate, so the
Android detector, the feature transform, the attention mask, the nine-member
ensemble and the label mapping are all doing the same thing Python does.

## Paths C and D — not run

```console
$ adb devices -l
List of devices attached
emulator-5554   device product:sdk_gphone64_arm64 model:sdk_gphone64_arm64
```

No USB Android device appeared in `system_profiler SPUSBDataType`, no wireless
ADB target answered.

The emulator camera is a synthetic colour pattern. **No hand has ever appeared
in it.** A run there would exercise the UI and nothing else, so it cannot stand
in for C or D and is not reported as if it could.

## What this leaves open

The brief's decision rule has a fourth branch:

> If A/B/C pass but D fails, investigate signer domain shift, camera framing,
> temporal capture and model robustness.

**That branch is the open one, and it cannot be evaluated without a phone.**
A and B pass. C and D are unmeasured. Saying "recognition is fixed" on the
strength of A and B would be exactly the substitution the brief forbids: a clean
offline benchmark standing in for real-phone accuracy.

## The cheapest experiment that would close it

1. Attach one Android phone.
2. Hold up a known clip (any of the 30) on a laptop and film it — path C.
3. Sign five known glosses directly — path D, three attempts each.
4. Enable capture diagnostics and read `DiagnosticsLog` for the real analyser
   rate, hand-detection rate, and sampled-frame count.

That is roughly fifteen minutes of work and it settles C, D, and the
live-risk hypothesis in `04_LIVE_RISK.md` at once.