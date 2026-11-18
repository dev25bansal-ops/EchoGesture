# What `release/samsung-final-app-test` actually contains

**Written** 18 Nov 2026 by Dev, after reading the branch Abhijeet asked me to review.
**Branch** `release/samsung-final-app-test` @ `ef44e85`, 115 commits ahead of the 8 Nov
merge base `f81dd30`, 641 files changed, 259,617 insertions.

Every claim below is tagged **[verified]** (I ran it or read the file myself and quote it)
or **[reader]** (a fan-out audit found it and I have not personally reproduced it).

---

## The headline

**The two branches have not merged since 8 November. Fifty commits of mine are absent
from his, and his release ships files I fixed on 15–17 Nov.** [verified]

```
$ git merge-base HEAD origin/release/samsung-final-app-test
f81dd30  merge: bring in two-person sprint plan   Sun Nov 8 10:39:50 2026
$ git rev-list --count origin/release/samsung-final-app-test..HEAD
50
```

His own handoff says he consumed `hackathon/dev-model @ 11bbddf`. That commit is **not
an ancestor of his branch** — `git merge-base --is-ancestor 11bbddf <his head>` → NO.
Neither is `510a207` (the 85.45% measurement) nor `a30e60d` (the native cache rebuild).
So "your weights are consumed exactly as delivered" is true of the *binary files*, which
he copied in by hand, and false of the *code*: he is running 8-Nov versions of every
script.

---

## What is genuinely good, and it is a lot

| Subsystem | Scale | Verdict |
| --- | --- | --- |
| Android app `app/android_final` | 30 Kotlin sources, 13 instrumentation tests | Real app: camera, recorder, glossary, history, settings, diagnostics, profiler |
| Android tests | 128 `@Test` methods at HEAD | Substantial; more than my side has [reader] |
| Device eval harness `evaluation/device` | 254 files, 12 MB | Extensive: geometry, quantization, robustness, parity, release-gate suites |
| Release documentation | 58 files in `docs/release` | Unusually thorough number-provenance work |
| Submission packs | `submission/` + `submission_final/` | Two full packs, manifests, checksums, screenshots, logcat |
| Diagnostics | `diagnostics/final-repair/` | On-device profiler, tensor-level parity, phone procedure |
| `src/main.py` | 798 lines | Explicit frame geometry, hash pinning, name-based input binding |

He found and fixed things I missed. The BUG_REPORT records eleven findings from real
device testing, and two of them (B-07, B-08) were errors in **my** demo script that would
have been spoken to judges. That is the kind of thing a second pair of eyes exists for.

---

## The findings that change what we can submit

### 1. His branch reverts two of my fixes, in files he ships [verified]

| File | His branch | Mine |
| --- | --- | --- |
| `scripts/ml/robustness.py` | `torch.softmax(lg,1)` then `np.mean(out)` — **per-member probability averaging** | `np.mean(logits)` then one softmax — the shipped aggregator |
| `scripts/ml/export_tflite_tf.py` | `drop = 0.0` then `if drop > 0.15` — **dead gate, nothing can be refused** | `drop = float_acc - int8_acc` |

Both are exactly the defects I fixed on 15 Nov. His file predates the fix and was never
rebased. So any robustness number produced on his branch is from the wrong estimator —
which is the same error that made him withdraw the 4.40 pp PASS.

### 2. Two decks exist and the manifest names the wrong one [verified]

| Deck | Headline | Robustness | Size |
| --- | --- | --- | --- |
| `submission/presentation/` (8 slides) | **85.45%** | 5.76 pp FAIL | 16.33 MB |
| `submission/FINAL_DELIVERY/` (11 slides) | **77.02%** | 5.36 pp FAIL | 15.57 MB |

`build_deck.py` in `FINAL_DELIVERY` hardcodes `77.02%` and `5.36 pp` — the values his own
`ROBUSTNESS_RECONCILIATION.md` calls superseded. `submission/RELEASE_MANIFEST.json` marks
the corrected 8-slide deck DONE. Both packs are committed on the same commit.

### 3. `results/results.csv` on his branch describes a different model from its own meta [verified]

```
top1_accuracy,>=70,72.78,%,"macOS arm64 laptop (not target hardware)",...,
  model=landmark-ensemble(4x fp16) sha256=08417e72da5...
```

72.78%, four members. The committed `landmark_model_meta.json` declares **nine**
members at 0.7702. Neither is the deck's 85.45%. Three different model identities are
in one tree.

### 4. `camera_further_25pct` measured nothing [verified]

```
clean                 top1=73.65892714171338 top5=87.42994395516413 signer_min=53.521126760563384
camera_further_25pct  top1=73.65892714171338 top5=87.42994395516413 signer_min=53.521126760563384
```

Bit-identical to the clean row in every field. That corruption was a no-op, so it
contributed a fake "0.00 drop" to a six-condition robustness sweep.

### 5. The report that says all 26 failures are asset-blocked contains a non-asset failure [reader]

`docs/release/V4_5_TEST_TRIAGE.md:111` says every failure is `FileNotFoundError`;
the same document at `:73-76` describes `ClosureAssemblyTest.test_closure_is_complete_and_honest`
as a stale-premise assertion failure. Its own summary contradicts its own detail.

Also [reader]: the triage says 502 tests / 433 passed; the committed `test-report.json`
says 472 tests / 366 passed / 97 skipped. Two different inventories.

### 6. The root `README.md` — the first file a judge opens — is three generations stale [verified]

Last touched 13 Nov (`b33aa6f`), one day **before** his own nine-member integration
(14 Nov, `5251bf8`) and four days before he corrected the deck (17 Nov, `6f63582`).
It says:

> The default runtime is Dev's **four-model FP16 TFLite ensemble** … `WHATFOR1` at
> 0.8821 confidence.

and in the accuracy summary:

> The shipped runtime is a **4-member FP16 ensemble** and scores **909/1249 =
> 72.78% top-1 / 87.51% top-5**

Three things are wrong in that one file: the member count (four, not nine), the
accuracy (72.78%, not the 85.45% his own deck now leads with), and the confidence
figure — 0.8821 is the value in `parity_ensemble_logits.txt`, which is the **retired
four-member fixture**. `src/README.md` and `models/README.md` repeat the four-member
claim with the same stale hashes.

So the corrected work exists and is dated 17 Nov, but the repository's front page was
never brought forward. A reviewer reading top-down hits the stale claim first and the
correction never.

### 7. Model documentation describes a model that does not exist [reader]

`models/README.md` documents a four-member ensemble, every member `[1,24,255]` "z
dropped", pinned at `06f655bb`. The committed meta declares nine members, five of them
381-wide x,y,z, blob `19dc4b86`. The README is from 12 Nov and was never updated after
the nine-member integration on 15 Nov.

### 8. `parity_ensemble_logits.txt` still describes the four-member ensemble [reader]

12 lines, consumed by `EnsembleParityTest`, which asserts four members and confidence
0.8821. The runtime fixture beside it describes nine members and 0.9749. Two mutually
inconsistent parity references are committed together.

---

## The Android feature pipeline: one real formula bug that is provably inert, and one
## constant that is doing more work than it should [verified]

I compared `FeatureTransform.kt` against `src/features.py` step by step. Nine of the ten
steps match exactly — slot convention, root-centring, the shared wrist→MCP scale
(including z), the pre-centring ordering of inter-hand geometry, velocity/acceleration
differencing, the 255/381 widths and column order, stage-2 interpolation, the attention
mask, name-based tensor binding, and mean-logit-then-softmax aggregation. That is
careful work.

### The inert bug: `geom[0]` drops `dz` in the 381 path

Python measures the inter-hand distance over the full coordinate vector:

```python
# src/features.py:137-138, with xy = landmarks[..., :3 if use_z else 2]
d = (hands_xy[1][WRIST] - hands_xy[0][WRIST]) / scale
return np.array([np.linalg.norm(d), d[0], d[1]], np.float32)
```

Kotlin always uses two coordinates:

```kotlin
// FeatureTransform.kt:134-136
val dx = (hands[1][WRIST * coords] - hands[0][WRIST * coords]) / scale
val dy = (hands[1][WRIST * coords + 1] - hands[0][WRIST * coords + 1]) / scale
geom[t * 3] = sqrt(dx * dx + dy * dy)
```

and the comment at `FeatureTransform.kt:104-106` asserts *"Geometry is ALWAYS x/y only
… that is what src/features.py does"* — which is false for the z path.

**It cannot change a single prediction, and the reason is worth recording.** MediaPipe
defines landmark 0 as the depth origin, so the wrist's z is zero by construction. Across
the entire corpus (3,390 clips) the largest `|wrist z|` is **1.71e-06**, so `dz` in
`geom[0]` is bounded by **3.4e-06** in normalised units. On 1,049 both-hands frames the
relative understatement is 0.000 to three decimals. The formula is wrong; the term it
drops is zero. No action needed beyond deleting the false comment, because the next
person to touch this code will trust it.

### The constant that matters: `CAPTURE_SPAN_MS = 3700`

`HandTracker.kt:129` fixes the live capture window at 3.7 s and resamples 32 frames
across it. The reasoning documented at `:106-128` is correct and important: because
`features.py` differences without `dt`, the sample step *is* the time base, so feeding
consecutive 33 ms camera frames would have made every velocity ~3.5x too small. Fixing
that was right.

But the value is calibrated to the wrong number. It comes from "the extractor's own
docstring records a typical clip as 110 frames, about 3.7 s" — and that docstring
(`src/landmark_frontend.py:136-139`) is describing **one** clip where the first 32 frames
yielded no detections; it is an anecdote, not a corpus statistic. Measured over 3,388
real ASL Citizen clips:

| | value |
| --- | --- |
| median clip duration | **2.57 s** |
| mean | 2.82 s |
| p10 / p90 | 1.67 s / 4.26 s |
| clips at or above 3.67 s (110 frames) | **16%** |
| span that would be correct for the median clip | **2,574 ms** |

Since velocity scales with the sample step, a 3.7 s window against training's per-clip
span multiplies every velocity and acceleration by the ratio of the two sample steps,
`(3700/31) / (1000 × duration/31)`, which simplifies to `3700 / (1000 × duration)`:

| | factor |
| --- | --- |
| median clip | **1.44×** |
| p90 (short clips) | **2.22×** |
| p10 (long clips) | 0.87× |
| clips where the factor exceeds 1.25× | **66%** |
| clips where it exceeds 2× | 17% |
| clips within 20% of the training scale | **21.8%** |

So on four clips in five the phone feeds the model velocities at materially the wrong
amplitude — most often too large, because most signs are shorter than the assumed
window. This is a distribution shift in the input, not a code defect: the constant is
doing the best it can with no clip length to go on. But 21.8% agreement with the training
scale is a plausible contributor to the on-device accuracy gap, and it is the only
feature-side difference I found that is both real and large.

Two ways to narrow it, neither of which I would attempt before the deadline:

1. Move 3700 → ~2600, which matches the median but is still one number for all clips.
2. Better: derive the window from sign onset if the recorder already detects it
   (`SignRecorder.kt` has a segmenter), so the span tracks the sign rather than a global
   assumption. That is the only version that reproduces training's per-clip sampling.

**Do not change it without re-measuring.** A different span changes the input
distribution, so any on-device accuracy figure taken before the change stops applying.

---

## The numbers, with the population each belongs to

| Figure | Population | Whose | Safe to quote? |
| --: | --- | --- | --- |
| 85.45% | 1,251 test, native 640×480, nine members | mine, reproduced by his [verified] | **yes, with "native geometry"** |
| 82.47% | 1,249 test, 256 squash, nine members | mine | yes, with "256 squash" |
| 85.84% | 346-clip **val**, native | his | yes, with "validation, n=346" |
| 82.37% | 346-clip **val**, 256 squash | his | yes, with "validation" |
| 77.02% | 1,249 test, **old 8-Nov cache** | stale | **no** |
| 76.67% | 30 clips, clean ZIP extraction | his | only as a smoke test, say so |
| 72.78% | 1,249 test, **four-member** | superseded | **no** |
| 62.61% | single classifier | historical | no |

His `V4_5_CANONICAL_SUMMARY.md` gets almost all of this right. The problem is that the
documents which contradict it are still in the tree, and one of them is the deck.

---

## What I have already fixed on my side

- `results/robustness_gated.json` — his B-10 was correct; the "raw" row used the
  coords-kept diagnostic. Re-measured with a new script: **+5.76 raw FAIL / +2.39 gated
  PASS at 86%/80% coverage** (was +5.52 / +1.24). Clean column reproduced exactly.
- `LATENCY_DECOMPOSITION.md` and `landmark_frontend.py` — his B-09 found real
  inconsistency; the resolution was my re-measurement, not his "claim no per-frame
  PASS". Per-frame is 22.95 ms p95 against a 50 ms budget.

Both are committed. `docs/hackathon/REPLY_TO_ABHIJEET_2026-11-18.md` is the answer he
needs, including the one decision only he can make: his branch withholds the weights on
purpose for licensing reasons, and my branch publishes all nine in a public repo. Those
two facts cannot both stand.

---

## What I would do in the next 36 hours, in order

1. **Rewrite the root `README.md`.** It is the first file a judge opens and it is four
   days and two model generations behind. Everything else on this list is easier to
   excuse than that.
2. **Pick one deck and delete the other.** `submission/presentation/` (85.45%, corrected
   robustness). Leaving two contradicts the manifest.
3. **Decide the weights question** (A/B/C in the reply). Everything else in the
   submission is downstream of that.
4. **Rewrite `results/results.csv` on his branch** so it does not say 72.78% / four
   members while the meta says nine.
5. **Merge `hackathon/dev-model` into his branch**, or port the two fixes by hand
   (`robustness.py` aggregation, `export_tflite_tf.py` gate). Without this, any further
   robustness number he generates is from the wrong estimator.
6. Fix `models/README.md`, `parity_ensemble_logits.txt`, and the `camera_further` no-op.
   None is load-bearing for scoring, but a judge who reads the repo will find them.
