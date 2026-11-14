# Shared branch — what I integrated and what needs Dev

Branch **`shared/v4.4-dev-integration`**, cut from the validated V4.4 release
tip (`8dace91`) and pushed. Dev's `hackathon/dev-model` was **not** modified,
not merged, and not force-pushed.

| | |
| --- | --- |
| Branch | `shared/v4.4-dev-integration` |
| Base | `8dace91` (V4.4 release, app + evaluator verified) |
| Dev source | `hackathon/dev-model` @ `11bbddf` |
| Commits added | 3 |

---

## 1. What came across from Dev

| Path | What it is |
| --- | --- |
| `scripts/ml/` (18 files) | Training, export, evaluation and robustness harnesses |
| `docs/hackathon/` (7 new) | Handoff, app-migration note, 14 Nov update, Android build env, HF dataset, pod setup |
| `results/EXPERIMENTS.md` | The rejected-experiments record — valuable, kept |
| `results/QUANTIZATION.md` | INT8 result |
| `results/{eval_summary_val,eval_summary_test,robustness,robustness_pixel}.json` | Raw measurements, kept as provenance |
| `scripts/ml/export_tflite_tf.py` | One-line fix: pass the checkpoint's own `backbone` |

## 2. What did NOT come across, and why

| Path | Reason |
| --- | --- |
| `models/*.tflite`, `models/*.task` (11 files, 16.06 MB) | **Restricted.** Dev's branch tracks all nine members. Redistribution is not cleared; they stay in private storage per `NOTICE`. `git checkout` was scoped to source directories only, so these were never touched. |
| `src/main.py` | V4.4's production evaluator already implements Dev's two-input contract, is hash-pinned, and has instrumentation Dev's 238-line reference lacks. Porting Dev's version would be a regression. |
| `src/features.py` | Byte-identical between branches already. Nothing to merge. |
| `results/results.csv` | Conflicting robustness verdicts. Left for explicit reconciliation — see `RECONCILING_DEV_NUMBERS.md`. |
| `models/label100.json` | Redundant with `landmark_labels.json`; not needed by either runtime. |

## 3. Verification performed on the shared branch

| Check | Result |
| --- | --- |
| Remote HEAD == local HEAD | **YES** — `ce793cffa3ec12ef78e1f477d39daa3c6e63e5a0` |
| All files from all 3 commits present remotely | **34 / 34**, 0 missing |
| Restricted binaries added by this branch | **0** |
| `src/main.py`, `src/features.py`, `LandmarkClassifier.kt` unchanged from V4.4 | **YES**, all three byte-identical |
| Evaluator loads nine members | **YES** — `use_z` = `[F,F,F,F,T,T,T,T,T]` |
| Ordered ensemble hash | **`995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`**, identical to Dev's files |
| JVM tests | **110 passed, 0 failed** |
| Python suite | **424/493 pass**, 3 failed + 23 errors — **identical to the untouched V4.4 tip**, all blocked on gitignored artifacts |
| Release bundle | **builds and verifies**, all nine member hashes correct, `status: structure_verified` |
| Numerical parity | fixture source hashes unchanged, so the recorded 0.000000 / 0-exact / 5e-6 / 9-of-9 result still holds |
| Android instrumentation | 11/11 at V4.4; **not re-run** on this branch |

## 4. The one real conflict: robustness

Dev reports **PASS (4.40 pp)**. V4.4 measured **FAIL (10.00 pp)**. Same nine
models, same branch base.

**Both are correct.** Dev corrupts the 21 landmarks *after* detection; V4.4
corrupts the video *before* MediaPipe sees it. A landmark-level pass says
nothing about pixel-level behaviour, and the deployed app sees pixels.

Dev's own `results/robustness_pixel.json` independently agrees with the FAIL —
every pixel condition breaches the 5-point budget, worst 35 pp at
`downscale_35pct`. Full detail in `RECONCILING_DEV_NUMBERS.md`.

**Ask of Dev:** keep both rows, add a `corruption_level` column, and never quote
4.40 as the answer to a camera-robustness question.

## 5. Remaining tasks

### For Dev

1. **Stop tracking weights.** `hackathon/dev-model` has 11 `.tflite`/`.task`
   files in Git (16.06 MB), added at `ccfd810` and earlier. There is no
   `.gitignore` rule for them on that branch. Removing them in a new commit
   does **not** remove them from history — that needs a history rewrite, which
   is the owner's decision. At minimum, add the ignore rule so it stops growing.
2. **Confirm the robustness split** and update `results/results.csv` with a
   `corruption_level` column so the two verdicts stop looking contradictory.
3. **Re-run `scripts/ml/robustness.py` against the nine shipped members** if the
   4.40 figure is to be quoted. Its own commit message says it was re-measured —
   worth confirming which checkpoint set produced it.
4. **Review `scripts/ml/`** on the shared branch — brought in verbatim and
   unmodified, 18 files.

### For Abhijeet

1. **Physical phone test.** Still the largest unverified surface. Nothing in
   this branch has run on a real device.
2. **Build the APK from the shared branch** and record its SHA-256 with the
   commit, per `BUILD_INSTRUCTIONS_SHARED.md` §5.
3. **Android instrumentation** re-run on this branch (11/11 at V4.4).
4. **Decide on `results/results.csv`** — merge with a level column, or keep
   per-branch files.
5. **Rights decision** on the weights already in public history.

### Blocked on humans, unchanged

Physical phone, rights clearance, demo video, exact `CollegeName_TeamName`,
Google Form approval.

## 6. What is proven and what is not

**Proven**
- The shared branch builds, loads all nine models, and verifies the bundle.
- Dev's model files and mine hash to the same ordered value.
- The integration introduced no test regressions — the failure set is identical
  to the untouched V4.4 tip.
- V4.4's app and evaluator are byte-identical on this branch.

**Not proven**
- Anything on a physical phone.
- That landmark-level robustness implies camera robustness. It does not, and
  the pixel evidence points the other way.