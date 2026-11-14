# Phone test report

**Status: NOT TESTED — no physical Android device was attached.**

This file exists so the gap is visible rather than quietly omitted. It is a
worksheet someone can execute on a phone, not a report of results.

---

## What was actually checked

| Check | Environment | Result |
| --- | --- | --- |
| APK installs | Android 14 emulator (`sdk_gphone64_arm64`) | **PASS** — `Success` |
| App opens | same | **PASS** — process alive, no crash, no ANR |
| Model identity renders | same | **PASS** — footer reads `9-model ensemble · FP16 TFLite · SHA-256 995cc8d0…` |
| Permissions | built APK manifest | **PASS** — `CAMERA` only; `INTERNET` absent |
| Nine assets embedded | APK contents | **PASS** — 9 × `.tflite` + detector + labels + meta |
| Live camera in an emulator | same | **NOT representative** — virtual camera, not a signer |

**An emulator is not a phone.** It has no real camera optics, no rolling
shutter, no thermal behaviour, and no signer. Nothing below was measured.

---

## What is missing, precisely

| Measurement | Status |
| --- | --- |
| Top-1 accuracy on live signs | **NOT TESTED** |
| Top-5 accuracy on live signs | **NOT TESTED** |
| Acceptance / rejection rate | **NOT TESTED** |
| Per-class accuracy | **NOT TESTED** |
| Per-signer accuracy | **NOT TESTED** |
| On-device inference latency | **NOT TESTED** |
| End-to-end time (tap → caption) | **NOT TESTED** |
| Power draw | **NOT TESTED** |
| Thermal drift over 10 min | **NOT TESTED** |

The only accuracy figures in this project are **corpus** figures measured on a
laptop. They must not be presented as phone accuracy.

---

## How to run the test

### Setup

```bash
adb devices -l                 # must show a real phone, NOT emulator-5554
adb install -r dist/echogesture-v4.3-masked.apk
```

Confirm on the camera screen that the footer reads
`9-model ensemble · FP16 TFLite · SHA-256 995cc8d0…`. If it does not, you are
running the wrong build — stop and install the correct APK.

### Protocol

Ten glosses from the shipped vocabulary, **five attempts each, 50 minimum**.
One continuous session per signer. Do not discard attempts that fail — failures
are data.

Choose ten that together cover the stated conditions:

| # | Condition to cover | Example gloss |
| --: | --- | --- |
| 1 | one-handed, plain background | |
| 2 | one-handed, cluttered background | |
| 3 | two-handed | |
| 4 | slow execution | |
| 5 | fast execution | |
| 6 | normal lighting, close | |
| 7 | normal lighting, far | |
| 8 | moderate low light | |
| 9 | signer B (different person) | |
| 10 | signer B, different distance | |

Conditions 9–10 require a second consenting signer. If none is available, record
`single-signer` in the report — do not skip the rows.

### Worksheet

Fill one row per attempt in `phone_attempts.csv`. Columns:

```csv
attempt_id,gloss_truth,prediction,top5,confidence,accepted,hand_frame_ratio,record_duration_s,inference_ms,end_to_end_ms,device_model,android_version,lighting,distance,background,handedness,signer,correct,notes
```

Every attempt gets a row. `correct` is derived, never inferred from confidence.

### Metrics to compute

```
all_attempt_top1   = correct / attempts              (no exclusions)
top5               = truth appears in top5 / attempts
accepted_accuracy  = correct among accepted
coverage           = accepted / attempts
rejection_rate     = rejected / attempts
per_class          = grouped by gloss, n >= 5
per_signer         = grouped by signer where n > 0
```

Report all six. Report the denominator every time. A 90% accepted accuracy at
40% coverage is a different product from 75% at 95% coverage.

### Optional A/B

If time allows, install `echogesture-v4.2.apk` (four members, no mask) and run
the identical protocol. Report both. Only claim a difference if the same signs
were attempted in the same session.

---

## Recording results

Write them to `submission/PHONE_TEST_REPORT.md` under a "Results" heading, and
keep `submission/phone_attempts.csv` alongside it. State:

- exact APK SHA-256
- device model, Android version, SoC if known
- number of signers and their labels (S1, S2 — never names)
- every metric with its denominator
- what failed and why, where discernible

If an attempt is rejected by the app, that is a valid outcome. Record the
rejection reason shown on screen.

---

## Honest framing for the submission

If this stays untested, say exactly this:

> Recognition accuracy was measured on the ASL Citizen corpus signer-disjoint
> (77.02% top-1 on 1,249 test clips from 11 unseen signers, laptop CPU). The
> application was verified to install, launch and load the nine-member model on
> an Android 14 emulator. **No physical-device accuracy, latency, power or
> thermal measurement exists.**