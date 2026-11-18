# FINAL_RELEASE_STATUS

**ECHOGESTURE V5.2 — PARTIAL**

One-tap auto-recognition is implemented, tested and in the APK. Real-phone
verification is **BLOCKED**: no physical device was attached.

## What is done and verified

| Item | Evidence |
| --- | --- |
| Single-tap Start | "Start Recognition" → "Ready — perform one sign", seen on device |
| Automatic onset | 15 JVM tests incl. static-pose and motion onset |
| Automatic completion | `a completed sign auto-stops after the quiet period` |
| No Finish required | `oneTapProducesAnOutcomeWithNoSecondTap` |
| Nine-model classification | ensemble `995cc8d0…`, 26/30 = 86.7% on labelled clips |
| Uncertain handled safely | declined reading shows "Sign not recognised reliably" |
| Original-video scanner | 30 clips, per-clip CSV with verdicts |
| Android/Python parity | features 0.000000, mask 0, logits 5e-6, top-1 6/6 |
| Automatic completion memory bound | 600-frame cap added |

## What is not

| Item | Status |
| --- | --- |
| Physical-phone test | **BLOCKED** — no device |
| Live sign recognition | **UNVERIFIED** |
| The user's reported "most signs wrong" | **NOT ESTABLISHED** |

## APK

```
~/Desktop/EchoGesture-Final-Auto/EchoGesture-Final-Auto.apk
SHA-256  2e4dbb89a28cd08e2c405c7ee8301b5076869ebe57f74d1263365e556770d3f5
```

Package `org.echogesture.finalapp`, CAMERA only, 9 models, ensemble
`995cc8d0b6f6ac25`. Installs and launches clean.

## Source

Branch `release/samsung-final-app-test`, commit `812d835d93c5`.
Remote SHA verified equal.

## The honest demo

Tap Start Recognition once, perform a sign, let the app stop by itself. If it
says "Sign not recognised reliably", that is the app working — it declines rather
than guesses.

Do not claim live-phone accuracy has been measured. It has not.
