# FINAL_RELEASE_STATUS

**ECHOGESTURE V5.1 FINAL STATUS: PARTIAL**

The app now works as a one-tap automatic recognizer and the misleading result UI
is fixed. Real-phone verification is **BLOCKED**: no physical device was attached.

## Fixed and verified

1. **One-tap automatic capture.** Start Recognition → the app detects the sign's
   start and end → classifies → presents a result. No Finish tap required.
   Manual Finish retained as fallback.
2. **Uncertain readings no longer masquerade as recognized signs.**
3. **EchoGesture-Final-Auto.apk built, verified, installed, launched.**

## Not verified

**Nothing about a real phone.** Every number here is a labelled-clip replay on
the emulator. The phrase "verified on a real phone" does not apply to any part of
this release.

## APK

```
~/Desktop/EchoGesture-Final-Auto/EchoGesture-Final-Auto.apk
SHA-256: 556af6138b8b68ddbfe5906fd3b1e100fea7f17b71cb411a62c28e3e7178a2a8
```

| Check | Result |
| --- | --- |
| Package | `org.echogesture.finalapp` |
| Signature | debug, `2e7e40e2…` |
| Permissions | CAMERA only (no INTERNET) |
| Models | 9 members, ensemble `995cc8d0b6f6ac25` |
| Installs / launches | yes, no crash |
| One-tap UI transition | yes, verified on device |

## Test totals

- JVM: **146 pass, 0 fail**
- Instrumentation: **36 pass, 0 fail**
- Batch video scanner, 30 labelled clips: **26/30 correct (86.7%)**,
  **95.7% precision on accepted readings**

## What to do before the demo

The honest demo is: tap Start Recognition once, perform a sign you have already
confirmed works, and let the app stop by itself. If it says
"Sign not recognised reliably", that is the app working as designed — not a
failure to hide.

Not recommended: claiming live-phone accuracy has been measured. It has not.
