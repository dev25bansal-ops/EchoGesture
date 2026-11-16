# Dev handoff — final Track 02 state

Abhijeet → Dev. Date 2026-11-16. Deadline 2026-11-18 23:59 IST.

This is the sign-off request. Three things need a yes, a no, or a correction.
Everything else is information.

---

## 1. What is frozen

| | |
| --- | --- |
| Source branch | `release/samsung-track2-geometry-review` |
| Base | `hackathon/abhijeet-integration` @ `eef80e0` |
| Your model commit consumed | `hackathon/dev-model` @ `11bbddf` — **15 commits behind your current tip `73bff0d`**, see §3 |
| Ensemble SHA-256 | `995cc8d0…b4bb` (nine members, ordered) |
| Detector SHA-256 | `fbc2a300…cde1` |
| Preprocessing | **256 squash** is the shipped default. Native 640×480 is opt-in via `ECHOGESTURE_FRAME_SIZE=native`. |
| APK | `echogesture-p1.apk`, `150ffeeb…cb7a`, 83,350,522 B |
| Submission ZIP | `EchoGesture_Submission.zip`, `7691190b…6cca`, 13,379,265 B |

**No retraining, no architecture change, no new sweep.** Your weights are consumed
exactly as delivered.

---

## 2. What I changed, and why you should check it

I made three edits to shipped code. Each is small and each needs your read,
because two of them touch the model boundary you own.

### 2.1 `src/main.py` — explicit frame geometry

`ECHOGESTURE_FRAME_SIZE` selects `native` (no resize) or an integer edge length.
Unset behaves exactly as before at 256.

The reason it exists: the app requests 640×480 from CameraX and does not resize.
The Python path squashed 4:3 to 1:1. On the 346-clip validation split that
squash costs **3.47 points** (82.37% → 85.84%, n=345 paired, exact McNemar
**p=0.0501**). I left the default alone — a p of 0.05 at threshold, plus the app
already being native, is not enough to re-cut a release two days before the
deadline. It is available, documented, and off by default.

**Your call:** if you think the default should move to native, say so and I will
cut it — but it needs a re-run of the ZIP parity gate, which is a few hours.

### 2.2 `src/main.py` — latency accounting

`start_time` moved to *after* model init, and `_init_model` now warms MediaPipe
with one throwaway frame. Previously the first clip absorbed ~320 ms of warm-up
and inflated the reported clip time. My own early run showed a warm p50 of 236 ms,
but your later LATENCY_DECOMPOSITION work (4,650 ms warm `predict()`, detector
dominated) supersedes it — see the note below.

Neither number meets the 50 ms criterion. I have not tried to close that gap;
it is recorded as FAIL everywhere it appears.

### 2.3 `src/landmark_frontend.py` — `_fit()` passthrough

`read_frames` / `clip_to_landmarks` take `size: int | None`. `None` means no
resize. Default argument is `FRAME_SIZE`, so existing callers are unaffected.

**Verified byte-identical:** default output vs explicit `size=256` compares equal
on features and presence, tensor-for-tensor, not just by shape. That test is
`evaluation/device/test_geometry_contract.py`, 9 cases.

---

## 3. Numbers you should not quote as yours

| Figure | Why it is not a Dev number |
| --- | --- |
| 77.02% top-1 | Cached landmark provenance; the cache's presence mask disagrees with fresh extraction on 29% of test clips |
| 76.67% | 30-clip clean-extraction run of the packaged ZIP — a smoke test, not a benchmark |
| 85.84% | My 346-clip validation run, native geometry, opt-in. Not your 1,251-clip test split |
| 82.47% / 85.51% | Your fresh-extraction test-split figures, recorded but **not re-run by me** |

Your reported figures and mine agree in **direction** in every case where they
overlap. That is all I am claiming. `docs/release/V4_5_CANONICAL_SUMMARY.md` holds
the full seven-row table with each population labelled.

### Metadata discrepancy you should know about

Your branch tip has `models/landmark_model_meta.json` at
`ensemble_test_top1: 0.8545`. The APK and ZIP I shipped carry the older
`0.7702` in `assets/landmark_model_meta.json`.

The field is **never read by the app** — `ModelContract.kt` uses that file for
member order and widths only — so nothing displays the stale value. But someone
unzipping the APK and comparing it against your repository would see the
mismatch. I left it alone rather than adopt a number I had not reproduced and
rebuild the APK two days out. It is recorded in §"Known metadata discrepancy" of
`V4_5_CANONICAL_SUMMARY.md`.

**If you want it aligned**, the fix is: rebuild the APK from your sidecar,
re-run the 30-clip clean-extraction gate, update the three claim sites together.
Tell me and I will do it — that is a few hours, not a one-line edit.

### And one correction to something you may have seen

**The 4.40 pp landmark robustness PASS is withdrawn.** It averaged per-member
softmax probabilities. Shipped code averages raw logits then applies one softmax.
I published that 4.40 myself and am correcting it publicly.

Two subsequent figures need reconciling on your side before either is quoted:

- **5.36 pp** was my measurement on the *old 256-squash cache*. Superseded.
- **5.76 pp** is the harness corruption on the native cache (both coordinates
  *and* presence zeroed) — this is the number our docs now quote, and it fails.
- **5.52 pp** in your `results/robustness_gated.json` is labelled "raw (every
  clip)" but is computed from the **coords-kept diagnostic** (79.94%, delta
  actually 5.51). `DROP_HAND_ANALYSIS.md` states both variants; the gated file
  picked the diagnostic. The gate itself (margin ≥ 0.15) is real and verified in
  `CameraScreen.kt:958`, but it accepts on 86% / 80% coverage and must always be
  quoted with that.

Your **4.48 pp pixel PASS stands** — but only for 512 letterbox. The Android app
is 640×480 native, and native pixel robustness has not been measured. Do not let
those two collapse into one claim.

---

## 4. Test state, reported honestly

| Suite | Result |
| --- | --- |
| Android JVM | **110 / 110** |
| Android instrumentation | **16 pass, 0 fail, 1 skip** |
| Python `evaluation/device` | **433 / 502 pass, 3 fail, 23 error, 43 skip** |
| Clean-extraction `predict()` | 30 / 30 clips, 0 errors |

The 26 Python failures are **not** green and I have not suppressed them. Every
one is a `FileNotFoundError` on an artifact that is gitignored or retired:

- `models/landmark_classifier_fp16.tflite` (retired, on no branch)
- `models/landmark_classifier_int8.tflite` (restricted)
- `evaluation/device/v5_accuracy/results/val_346/summary.json`
- `evaluation/device/v7_android/parity_chew2.npy`

**Evidence they are asset-blocked, not broken:** dropping the retired
single-classifier weight into `models/` moves the run from 433 to **461 passed**
and clears 28 of them, with no code change. Full per-test classification in
`docs/release/V4_5_TEST_TRIAGE.md`.

The shipped nine-model release has its own suite, `release_official`, at **15/15**.

**This is the item I most want a second opinion on.** If you believe any of those
26 is a real defect rather than a missing file, tell me which and I will chase it.

---

## 5. Privacy — something you need to know about the repo

I found raw ASL Citizen participant IDs (of the `P` plus digits form) committed to this
**public** repository. Some in files I wrote this cycle; most pre-date it, back
to the initial commit `c47c5c9`.

Fixed in the working tree:
- `evaluation/asl_citizen_predictions.csv` — `participant` column removed, all 1,705 rows kept
- `evaluation/asl_citizen_{ptq,qat}_results.json` — `per_signer` keys pseudonymised
- Five V4.5 documents — per-signer tables pseudonymised to `S28C1`-style labels

**Not fixed, and I am not fixing it unilaterally:** the same IDs remain in the
**history** of eleven pre-existing files. Editing a file today does not remove it
from `git log`, from existing clones, or from GitHub's caches.

A proper cleanup is a coordinated history rewrite across all branches plus a
re-clone on your side. That is destructive shared-history surgery two days before
a deadline, so it is the owner's decision, not mine. Nothing has been
force-pushed. Full inventory in `docs/release/V4_5_PARTICIPANT_ID_AUDIT.md`.

---

## 6. Three questions

1. **Preprocessing.** Ship at 256 squash (current default, no re-verification
   needed), or move the default to native and accept a ZIP re-run? My
   recommendation: **ship at 256**, document native as available.
2. **The 26 Python failures.** Do you agree they are asset-blocked? If not,
   name the ones you think are real.
3. **Sign-off.** Are you willing to be listed as the model author on a release
   whose ensemble scores 85.45% corpus at native geometry (82.47% at the
   256-squash default the ZIP runs), ships **FP16 not INT8**, misses the 5 pp
   hand-dropout robustness criterion at 5.76 pp unconditional, and has **no
   on-device latency measurement at all**?

Answer 3 matters most. If any of those limits makes you uncomfortable being
named, say so now rather than after the form is sent.

---

## 7. Reproducing everything

```bash
git clone https://github.com/dev25bansal-ops/EchoGesture.git
cd EchoGesture && git checkout release/samsung-track2-geometry-review
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Geometry comparison:
```bash
./.venv/bin/python evaluation/device/geometry_paired.py --clips <DIR> --out out.json
./.venv/bin/python evaluation/device/geometry_latency.py
```

Contract test (no clips needed):
```bash
./.venv/bin/python -m pytest evaluation/device/test_geometry_contract.py -v
```

Clean-extraction check:
```bash
unzip -q submission/EchoGesture_Submission.zip -d /tmp/clean && cd /tmp/clean
/path/to/.venv/bin/python src/main.py <clip.mp4>
```

Android:
```bash
cd app/android_final && ./gradlew testDebugUnitTest
```

Build instructions, including the toolchain paths and the emulator setup, are in
`docs/release/BUILD_INSTRUCTIONS_SHARED.md`.

---

## 8. What is still blocked on a human

Not mine to clear, and none of it is code:

- Registration confirmation — **unverified**, no receipt in the repo
- Exact `CollegeName_TeamName` string — gates the deck filename
- Demo video — not recorded
- Model redistribution rights — no confirmed permission
- Organizer clarification on INT8 wording — drafted, unsent
- Google Form — requires explicit authorisation before submission

I will not report this as submitted without an actual receipt.