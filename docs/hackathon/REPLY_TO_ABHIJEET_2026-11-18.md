# Reply to Abhijeet's sign-off request

Date 2026-11-18. Answers the three questions in `submission/FINAL_HANDOFF_DEV.md` §6,
plus B-09 and B-10 from `submission/TEST_APP_NOW/BUG_REPORT.md`.

Both of your escalations were correct. One of them I have now re-measured, and the
correction makes our number slightly worse but the verdict unchanged.

---

## B-10 — you are right, and it is a worse error than the label suggests

`results/robustness_gated.json` reported `drop_pct: 79.94` as the row labelled
`"raw (every clip)"`. The harness corruption both-zeroes the hand and gives **79.70**;
79.94 is the coords-kept diagnostic. You also caught that 85.45 − 79.94 = 5.51, not
5.52.

I checked whether the gated rows shared the same substitution, because the raw row
of a table built at three operating points is a calibration check for the other two.
They did. There was no committed script that produced that file at all — so I wrote
one and re-measured.

`scripts/ml/robustness_gated_recheck.py`, `.scratch/landmarks_native`, seed 1234,
1,251 clips, nine members, harness corruption:

| row | clean | corrupted | delta | coverage | verdict |
| --- | --: | --: | --: | --- | --- |
| raw (every clip) | 85.45 | 79.70 | **+5.76** | 100% | FAIL |
| app gate margin ≥ 0.15 | 92.13 | 89.74 | **+2.39** | 86% / 80% | PASS |
| gate margin ≥ 0.30 | 96.11 | 94.44 | **+1.66** | 76% / 69% | PASS |

The clean column reproduces the old file to the digit (85.45 / 92.13 / 96.11), which
is what makes this a correction rather than a different experiment. The corrupted
column moves, and the gated delta is **+2.39, not +1.24**.

So: your 5.76 is right, my +1.24 was too generous, and the gated PASS survives with a
thinner margin. Updated in `results/robustness_gated.json` (with the superseded rows
kept in a `superseded` block rather than deleted), `results/results.csv`,
`docs/hackathon/DROP_HAND_ANALYSIS.md`, `docs/hackathon/CLAIMS_AND_LIMITS.md`.

**Your 86%/80% coverage framing is the right one and I have adopted it.** A gate that
answers 80% of clips is not the same claim as an ungated 90%, and burying that would
be the dishonest version.

---

## B-09 — the inconsistency was real, but your resolution is the wrong branch

You found `results.csv` 15.34 ms PASS, `LATENCY_DECOMPOSITION.md` 135 ms, and
`landmark_frontend.py` "FAILS", and concluded they cannot all hold. Correct.

But the resolution you took — *"quote the classifier (~4.2 ms) and the end-to-end
(4,650 ms); claim no per-frame PASS"* — preserves the wrong number. The 135 ms figure
and the 4,650 ms end-to-end were both measured while a `distllm` inference server from
another project was saturating the CPU. Re-running the same scripts on an unloaded
machine gives:

| | contended (what your docs quote) | clean re-measurement |
| --- | --: | --: |
| per-frame detector p50 | 135 ms | **14.5–16.4 ms** (3 runs) |
| per-frame detector p95 | 199 ms | **22.7–25.1 ms** |
| detector, 32-frame clip | 4,300–4,400 ms | **602 ms median** |
| warm `predict()` p50 | 4,650 ms | **720 ms** |
| cold start | 159 s | **62.6 s** |

Artifacts: `results/latency_per_frame_distribution.json` (three runs, 160 frames each),
`results/latency_end_to_end.json` (records the competing processes and CPU load rather
than assuming them away). Still slightly contended at 29.8% → 46.4% CPU, which is why
the detector-isolated numbers are the clean ones.

So the per-frame criterion **passes** at 22.95 ms p95 against a 50 ms budget, and the
end-to-end is ~720 ms warm rather than ~4.7 s. I have marked
`LATENCY_DECOMPOSITION.md` with a correction header that keeps its structural finding
(MediaPipe dominates, the classifier is a rounding error) and withdraws its absolute
numbers, and I have rewritten the stale `landmark_frontend.py` docstring that still
said "FAILS".

This is the third time this session that a number measured under CPU contention has
propagated into a document. I should have caught it before you had to.

---

## Q1 — preprocessing: 256 squash or native?

**Agree with your recommendation: ship 256.** Your 3.47-point native gain at exact
McNemar p=0.0501 is not a basis for re-cutting a release with 36 hours left, and the
ZIP parity gate is the thing that would have to be re-run.

One correction to how it gets stated, though. Your row 5b (85.84% native, 346-clip val)
and my row 4 (85.51% native, 1,249 test) are the same finding measured on different
populations — and they agree to 0.33 points. That agreement is stronger evidence than
either number alone and is worth saying out loud, because it means the native-geometry
gain is not a val-split artefact.

---

## Q2 — are the 26 Python failures asset-blocked?

**Mostly yes. One of your four "absent" artifacts is not actually absent, and it
changes the answer for a different class of test.**

Your §4 lists `models/landmark_classifier_int8.tflite` as "gitignored, restricted".
It is neither. It is **tracked on `hackathon/dev-model`** — I have it at 509,720 bytes
and it is in `git ls-tree HEAD models/`. It is not in *your* branch and not in
`.gitignore`; it is simply a file you did not carry across. That is a merge artefact,
not a licence restriction.

This matters because your PIPELINE test on `package-embedded models` will now be able to
find it, which means some of your 26 failures are "not merged" rather than
"asset-blocked". The distinction is real to a reviewer.

The rest I agree with, and your evidence method is the right one: supplying
`landmark_classifier_fp16.tflite` and watching 28 tests flip from fail/error to pass
with no code change is exactly the demonstration that distinguishes a missing asset from
a defect. I have not reproduced that specific run — I am accepting it on the strength of
your recorded numbers.

Two things I would add:

1. `evaluation/device/v5_accuracy/results/val_346/summary.json` and
   `evaluation/device/v7_android/parity_chew2.npy` are *measurement outputs*, not
   assets. A test that depends on a gitignored measurement output is a test that can
   only ever pass on the machine that produced it. That is a test-design problem, not an
   asset problem, and it is worth fixing after the deadline.

2. Your `ClosureAssemblyTest` case C — "stale premise, not stale code" — I would not
   dismiss that as quickly. A test whose premise is retired is still a test nobody has
   re-derived; it should be deleted or rewritten, not annotated.

---

## Q3 — sign-off as model author

**Yes, with the limits stated exactly as you stated them.**

Your §6 wording is accurate and I am willing to be named against it: 85.45% corpus at
native geometry, 82.47% at the 256-squash default the ZIP runs, FP16 not INT8, misses
the 5 pp hand-dropout criterion at 5.76 unconditional, no on-device latency measurement.

Three additions I want in the same sentence, because leaving them out would make the
statement weaker than the truth in one direction and stronger in another:

- **The 5-pt dropout criterion is met at the app's own acceptance gate** (+2.39 at
  86%/80% coverage), and the unconditional 5.76 FAIL is the honest number for the
  ungated case. Both belong together; neither alone is the whole story.
- **Per-frame detector latency passes** at 22.95 ms p95 on desktop against a 50 ms
  budget — the reverse of what your docs currently say (B-09).
- **The 4.48 pp pixel robustness PASS is 512-letterbox only** and does not transfer to
  the app's 640×480 native path. Your reconciliation file already says this; it needs to
  survive into whatever the judge reads.

What I am **not** willing to have said: that we meet the robustness criterion, that any
pixel robustness result is a nine-model native result, or that any on-device latency was
measured. None of those is true.

---

## Two merge problems that are bigger than any of the above

These are the things I would fix before anything else, and neither is in your handoff.

**1. Your branch cannot load its own model set — and that is deliberate on your
side, which makes it a decision rather than a bug.** `models/README.md` on
`release/samsung-final-app-test` says the weights are "not committed to this
repository … no confirmed redistribution permission" and marks all nine as
`**no** (restricted)`. `models/landmark_model_meta.json` nevertheless declares nine
members. The consequence, reproduced:

```
$ python -c "import sys; sys.path.insert(0,'src'); import main; main._init_model()"
FileNotFoundError: Ensemble member listed in landmark_model_meta.json is missing:
                   landmark_ens_tcn21_fp16.tflite
```

I was wrong to call this a merge accident in my first draft — your README states the
policy explicitly, so a clean clone of your branch cannot run `src/main.py` by design.
The APK is self-contained so the demo is unaffected, but the organizer-facing Python
entry point is the thing the evaluator runs.

**That policy and `hackathon/dev-model` cannot both hold.** Our branch tracks all nine
`.tflite` files in a **public** repository, which your own `RELEASE_MANIFEST.json`
records as a finding. So you are shipping a release whose Python side is unusable by
design precisely because the bytes are on my branch in public. Somebody has to decide
which of those two facts is the one we submit against:

- **A — respect the restriction.** Remove the nine weights, plus the detector, from
  `hackathon/dev-model`'s working tree and from history. The public repo then carries
  no model bytes; the ZIP/APK carry them to the organizer under whatever terms the
  form provides. This is the only option consistent with `NOTICE`, and it is a history
  rewrite on a public repo.
- **B — accept publication.** Leave them tracked and stop saying redistribution is
  uncleared. That is a legal position I am not qualified to take for you, and it
  contradicts `models/README.md` and `NOTICE`.
- **C — ship the ZIP only**, treat the Python source as documentation, and say so in
  the form so the evaluator is not surprised.

I recommend **C now, A after the deadline.** C is honest, requires no destructive
operation on a public repo before the form is sent, and does not claim a runnable
Python path we are not actually publishing. A is the correct end state; it is not a
36-hour-deadline operation.

**2. One of my own tests, and I want you to know before you cite it.** Our
`evaluate.py` scores a single `.tflite`, not the nine-member ensemble, and writes its
result over the tracked `results/eval_summary_test.json` unconditionally. I found this
yesterday and have not fixed it yet. If you re-derive any figure from that file, check
which model produced it.

---

## What I need from you

1. **Decide A, B or C on the weights** (previous section). My recommendation is C now,
   A after the deadline. This is the only item that needs a decision from you rather
   than a fix from me.
2. Drop the "no per-frame PASS" line from `LATENCY_RECONCILIATION.md` and take the clean
   re-measurement instead (B-09, above).
3. Re-quote the gated robustness row as **+2.39**, not +1.24 (B-10, above).
4. Your `ClosureAssemblyTest` — decide delete or rewrite rather than annotate.

If you pick C, tell me and I will update `NOTICE`, `models/README.md` and the
`RELEASE_MANIFEST.json` finding to match, so the three documents agree with the form.

Everything else in your handoff I accept as written.
