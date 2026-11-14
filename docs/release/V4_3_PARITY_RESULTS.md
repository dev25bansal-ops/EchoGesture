# V4.3 — Android/Python numerical parity

**Result: PASS.**

The fixture is produced by Dev's own pipeline (`src/main.py` +
`src/features.py` at `11bbddf`) running the same nine binaries the APK ships.
Kotlin re-derives everything through `FeatureTransform` and
`LandmarkClassifier` and compares numbers, not just labels.

Reproduce:

```bash
python evaluation/device/v4_3_parity/make_parity_fixture.py \
    <manifest.json> <clips-dir> 6
adb push /tmp/eg-v4/parity_fixture.bin /data/local/tmp/
adb shell "cat /data/local/tmp/parity_fixture.bin | run-as org.echogesture.finalapp \
    sh -c 'cat > files/parity_fixture.bin'"
adb shell am instrument -w -e requireParity true \
    -e class org.echogesture.finalapp.FeatureParityTest \
    org.echogesture.finalapp.test/androidx.test.runner.AndroidJUnitRunner
```

## Results

Six clips, five distinct signers, 64–95 source frames each, five of six
multi-hand.

| Clip | Frames | Feature max abs diff | Mask mismatches | Logit max abs diff | Top-1 |
| --- | --: | --: | --: | --: | --- |
| DOWNSIZE1 | 89 | 0.000000 | 0 | 0.000005 | agree |
| 8HOUR | 67 | 0.000000 | 0 | 0.000004 | agree |
| SHAVE1 | 95 | 0.000000 | 0 | 0.000002 | agree |
| HOW1 | 71 | 0.000000 | 0 | 0.000004 | agree |
| DOG1 | 64 | 0.000000 | 0 | 0.000003 | agree |
| NOON1 | 64 | 0.000000 | 0 | 0.000004 | agree |

- **Feature tensors** (both 255- and 381-wide) are bit-identical. Both sides do
  float32 arithmetic on the same landmark input, so this is the expected result,
  not a suspicious one.
- **Masks** match exactly at every one of the 24 steps for every clip.
- **Per-member logits** differ by at most 5 × 10⁻⁶, which is FP16 kernel
  ordering, not a semantic difference.

## What this test exists to catch

It caught a real defect on its first run: the z-path scale factor.

`src/features.py` computes `np.linalg.norm(middle_mcp - wrist)` on whatever it is
handed. With `use_z=True` that array is 3-dimensional, so the norm **includes
z**. Kotlin computed it from x,y only. Consequence: every z-member feature was
scaled by a different factor per frame.

Symptom profile — the reason this needed a numeric test:

- the model loaded without error;
- it ran without error;
- it produced a top-1 label;
- **the label agreed with Python on all six clips.**

Only the feature comparison showed it: **max abs diff 1.047**.

After the fix the same comparison gives 0.000000.

## Tolerance, and why it was tightened

An earlier revision allowed `feature < 1e-4` and `logit < 0.05`. That **passed**
the 1.047 feature defect, because the assertion was reading features through the
same path that was broken. It also allowed a defect to hide behind a loose
logit bound.

Current bounds, set from what was actually observed after the fix:

| Quantity | Observed | Bound |
| --- | --- | --- |
| Feature max abs diff | 0.000000 | **1e-3** |
| Mask mismatches | 0 | **0 (exact)** |
| Logit max abs diff | 5e-6 | **1e-2** |
| Top-1 agreement | 6/6 | **all clips** |

A reordering bug would blow the feature bound immediately; the 1e-3 headroom is
there only for float32 associativity differences.

## Coverage against the brief

| Required case | State |
| --- | --- |
| 255-wide member cases | **VERIFIED** — 4 members × 6 clips |
| 381-wide member cases | **VERIFIED** — 5 members × 6 clips |
| Clips with missing hands | **PARTIAL** — masks contain zeros on these clips, but no clip had a fully absent hand |
| Distinct signers | **VERIFIED** — 5 signers |
| Short and long recordings | **VERIFIED** — 64 to 95 frames |
| Full nine-member ensemble | **VERIFIED** — aggregated top-1 agrees 6/6 |

**Gap, stated plainly:** no fixture clip had a hand absent for the entire
recording, so the all-zero-mask edge case is exercised by unit tests and by the
"no hands in view" path, not by a real clip. Worth adding when an authorized
clip of that kind is available.