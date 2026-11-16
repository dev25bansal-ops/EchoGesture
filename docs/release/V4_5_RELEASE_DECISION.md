# V4.5 — release decision

**Decision: adopt native-resolution preprocessing as an opt-in mode; keep the
submission ZIP on the frozen default.**

Date 2026-11-16. Deadline 2026-11-18 23:59 IST.

---

## What changed in the evaluator

Additive only. Nothing about the model, weights, feature contract, mask,
aggregator or output schema changed.

| Change | Effect |
| --- | --- |
| `read_frames(..., size=None)` passes frames through unresized | enables native geometry |
| `clip_to_landmarks(..., size=...)` threads it through | same |
| `ECHOGESTURE_FRAME_SIZE` env var: `native` or an integer | explicit, validated opt-in |
| Default remains **256** | every existing result stays comparable |
| MediaPipe graph built during `_init_model()` | first timed call is inference, not setup |
| `latency_ms` starts **after** `_init_model()` | no longer includes cold start |

Verified by test: default output is byte-identical to before, and
`size=None` produces genuinely different frames (640×480 vs 256×256), not the
same tensor with a different label.

---

## The measured case for native

Paired comparison through **our frozen production `predict()`**, same nine
models, same mask, same aggregator, only the frame geometry changed.

### Primary population — ASL Citizen validation, 346 clips, 5 signers

This is the authorized validation split, and it was **not** used to choose this
change.

| geometry | correct | top-1 |
| --- | --: | --: |
| 256 squash (shipped default) | 285/346 | **82.37%** |
| native 640×480 | 297/346 | **85.84%** |

Paired, n=345 (one clip has no detectable hand in either geometry):

| | |
| --- | --: |
| both correct | 275 |
| squash only | 10 |
| native only | 22 |
| net | **+12 clips, +3.47 pp** |
| McNemar exact, two-sided | **p = 0.0501** |

**Per signer** — 4 of 5 improve, and the one that does not has n=13:

| signer | n | squash | native | Δ |
| --- | --: | --: | --: | --: |
| P12 | 24 | 79.2% | 79.2% | 0 |
| P21 | 104 | 82.7% | 86.5% | +4 |
| P26 | 101 | 79.2% | 84.2% | +5 |
| P39 | 103 | 86.4% | 90.3% | +4 |
| P5 | 13 | 84.6% | 76.9% | −1 |

**Per class** — 16 better, 76 unchanged, 8 worse:

No class loses more than **one clip**. The eight that lose one are DOWNSIZE1,
CLOSE, FOREIGNER1, COPY, DOG1, AXE1, ZEBRA, BEE1. There is no critical-class
collapse.

### Secondary — val_30, 30 clips

76.67% → 86.67%, 4 gained vs 1 lost. Same direction, same rough magnitude,
but n=30 is too small to decide anything on its own.

### Third — Dev's test split

82.47% → 85.51% on 1,249 clips, McNemar z=3.10, p=0.0019. **Not used for
selection** — the test split has been examined repeatedly during development.
It is quoted here only as agreement in direction.

---

## Runtime cost

| | p50 | min | max |
| --- | --: | --: | --: |
| 256 squash | 236.1 ms | 203.3 | 287.1 |
| native | 241.8 ms | 199.4 | 339.6 |

**+5.7 ms p50 (+2.4%)** on a laptop, full clip-to-JSON. MediaPipe dominates, so
the extra resize work is nearly free. Both are far above the 50 ms budget
either way; this does not change that verdict.

---

## Why the ZIP keeps the 256 default

Native is better and I am confident in the direction. It is still **not** what
ships in the submission bundle, for three reasons:

1. **p = 0.0501 is exactly at the threshold.** It is not comfortably decisive on
   the population we are allowed to select on.
2. **The Android app is already at native.** The APK needs no change and already
   gets this benefit. The Python evaluator is a separate judging artifact.
3. **The frozen ZIP is verified end-to-end.** Every published number, the 30/30
   clean-room run, the hash pins and the result files were produced at 256.
   Swapping the geometry days before the deadline buys +3.5 points on a split
   we are explicitly told not to tune on, at the cost of re-verifying everything.

The brief's own rule applies: *"Do not risk submission correctness for an
unproven last-minute change."* A 0.05 p-value is unproven by that standard.

---

## What is therefore true of the submission

| Artifact | Geometry | Accuracy to quote |
| --- | --- | --- |
| Android APK | native 640×480 (already) | no phone measurement exists |
| Submission ZIP | 256 squash | 77.02% corpus figure; 76.67% on our 30-clip clean-room run |
| Native mode, if enabled | 640×480 | 85.84% on 346-clip validation — **not** the shipped default |

**The 85.84% must never be presented as the submission's number.** It is the
measured cost of a geometry the packaged evaluator does not use by default.

---

## Reversal is one environment variable

```bash
ECHOGESTURE_FRAME_SIZE=native python src/main.py clip.mp4
```

No code change, no rebuild, no re-pin. If the organizers run the evaluator on
their own clips, they get the documented default; if we later decide native is
the right call, flipping it costs nothing and the tests already cover it.

---

## What would change this decision

- A paired comparison on an untouched population reaching p < 0.01, **or**
- an independent signer set, **or**
- confirmation that the organizer evaluates with our own frames rather than
  their own decode path.

None of those is available before the deadline.