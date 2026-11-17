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
