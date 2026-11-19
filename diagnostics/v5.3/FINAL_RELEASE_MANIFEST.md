# FINAL_RELEASE_MANIFEST

## APK — unchanged from V5.2, and that is correct

| Field | Value |
| --- | --- |
| Path | `~/Desktop/EchoGesture-V5.3/EchoGesture-V5.3.apk` |
| SHA-256 | `2e4dbb89a28cd08e2c405c7ee8301b5076869ebe57f74d1263365e556770d3f5` |
| Size | 82,842,069 bytes |
| Package | `org.echogesture.finalapp` |
| Permissions | CAMERA only — no INTERNET |
| Ensemble | 9 members, `995cc8d0b6f6ac25…` |
| Classes | **100** |
| Detector | `hand_landmarker.task` present |

**This hash is identical to the V5.2 APK on purpose.** The only source change
this session was in `src/main.py` — the Python evaluator's vocabulary pin. No
Android file was touched, so rebuilding produced a byte-identical artifact.
Presenting it as a new build would be false.

## Baseline preserved

`~/Desktop/EchoGesture-Final-Auto/EchoGesture-Final-Auto.apk` — verified present,
hash `2e4dbb89…`, untouched.

## What changed in source

| File | Change |
| --- | --- |
| `src/main.py` | `PINNED_LABELS_SHA256` set and enforced; wrong vocabulary now refused |
| `evaluation/device/v5_accuracy/tests/test_vocabulary_identity.py` | New regression test |

## Not published

APK contains trained weights and was written to the operator's Desktop only.
Not uploaded, attached, or committed — the repository is public and
redistribution rights are unresolved.
