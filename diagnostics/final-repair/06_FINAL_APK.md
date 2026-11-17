# 06 — Final APK

| Field | Value |
| --- | --- |
| Path | `~/Desktop/EchoGesture-Final-Fixed/EchoGesture-Final-Fixed.apk` |
| SHA-256 | `39295e6ae1c61f648d16f359323c38fe1a343c943dd727b4b7015efd298f35b6` |
| Size | 83,253,286 bytes |
| Package | `org.echogesture.finalapp` |
| Version | `1.0` (versionCode 1) |
| minSdk / targetSdk | 24 / 34 |
| Signature | debug cert, SHA-256 `2e7e40e24ccff12d99457a4267a377cb5b9ae14c00750ba4b406e8df6be37e2c` |
| Branch | `release/samsung-final-app-test` |

## Permissions in the built APK

```
android.permission.CAMERA
org.echogesture.finalapp.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
```

No `INTERNET`. Verified with `aapt2 dump badging`.

## Model integrity

Nine members inside the APK. Concatenated in meta order and hashed:

```
995cc8d0b6f6ac25...   (matches ModelContract.ENSEMBLE_SHA256)
```

Per-member, all matching `ModelContract.MEMBER_SHA256`:

| Member | SHA-256 prefix |
| --- | --- |
| tcn21 | `af031d89` |
| tcnm61 | `b0a97aa4` |
| tcnm62 | `1d247397` |
| rob71 | `105218a8` |
| z_s0 | `387749f8` |
| z_s1 | `02a2dc70` |
| z_s2 | `3b97a2d4` |
| z_s7 | `2f13d0eb` |
| z_s13 | `3d3a61e0` |

Detector `hand_landmarker.task` and labels are the files the running app loads;
the app displays the ensemble digest on the Camera tab as an independent check.

## Install / launch verification

On `eg34` (sdk_gphone64_arm64, Android 14 / API 34):

| Check | Result |
| --- | --- |
| `adb install` (clean uninstall first) | Success |
| Process alive after launch | yes |
| Crash buffer | empty |
| Camera tab reached | yes |
| Nine-model footer | `9-model ensemble · FP16 TFLite · SHA-256 995cc8d0b6f6…` |
| Deny permission → card | shown, correct |
| Grant permission → card clears, camera opens | yes |
| Grant externally then tap card button | recovers, no crash |

## What this APK does NOT include

- **No automatic one-tap stop.** The capture flow on this branch is two-tap
  (Start sign → Finish sign). Phase 7 was gated on "only after the basic
  classification pipeline is reasonably validated", and the live pipeline could
  not be validated without a phone. The one-tap flow exists on a separate branch
  but was not merged here, because merging it unvalidated would put an untested
  capture path in front of the user on the eve of the demo.
- **No live recognition claim.** See `07_PHONE_STATUS.md`.

## Rollback

The previously verified APK is untouched:

```
~/Desktop/EchoGesture-Mobile-Test/EchoGesture-Test-Ready.apk
```

## Not published

This APK contains the trained ensemble. It was written to the operator's Desktop
only. It was **not** uploaded, attached, or committed anywhere, because the
repository is public and redistribution rights for the weights are unresolved.