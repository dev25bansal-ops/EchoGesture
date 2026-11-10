# EchoGesture submission-safety audit (A15, 8 November 2026)

**Verdict: NOT READY FOR SUBMISSION. Do not label this release compliant.**
Organizer vocabulary and licence review are both unresolved. This audit
covers safety only (privacy, licensing, secrets, path/data handling); it
does not check accuracy, latency, power, thermal, or INT8 compliance — see
`docs/release/FINAL_EVIDENCE.md` (all NOT_MEASURED) and
`evaluation/device/CANDIDATE_INTEGRATION.md`.

Audited in an isolated copy at `$TMP/echogesture-.../A15` (no `.git`
history, no `.gitignore` — original checkout verified separately). The
original repo has no `LICENSE` file either. Lab:
`evaluation/device/privacy_lab/` (`audit.py`, `reproduce_defects.py`,
`test_privacy_lab.py`, `fixes.py`).

---

## 1. Verified checklist

| # | Check | Result | Evidence |
|---|---|---|---|
| C1 | Shipped code is offline (`src/`, `app/`) | PASS (with note) | Zero network imports in `src/` or `app/` after excluding the `queue.Queue` attribute false-positives. Only network use is research-only (`evaluation/eval_asl_citizen.py`, `evaluation/device/fetch_smoke_clip.py` → `urllib.request` against `download.microsoft.com`), never imported by `src/main.py`, `app/offline_demo.py`, or the bundle builder. |
| C2 | No hardcoded secrets | PASS | Full-repo scan for `api[_-]?key / client_secret / password / PRIVATE KEY / Bearer …` finds nothing. `secretary` in `models/label100.json` is the ASL gloss, not a credential (naïve `secret` grep false-positives on it; scanner excludes it). |
| C3 | Demo temp-clip deletion | PASS on-disk; wording overclaims slightly | `app/offline_demo.py` `_run()` → `NamedTemporaryFile(delete=False)` then `finally: path.unlink(missing_ok=True)` — verified deleted on success **and** on predictor exception (repro D5). Residual: SIGKILL/machine crash orphans the file, and frames are held in RAM (deque + `list(captured)`) though never written. Proposed wording: "no footage is kept on disk after each capture; frames are in memory only during inference". See D4. |
| C4 | Research clips outside Git | PASS in behaviour; BLOCKER is the lab copy | No `*.mp4/*.avi/*.mov/*.npz/*.npy/*.pt` tracked; `git ls-files` shows only `evaluation/asl_citizen_predictions.csv` and `models/*.tflite`. `fetch_*.py` write to operator-chosen `--output` dirs. **The isolated audit copy lacks the original's `.gitignore`** (original excludes `*.mp4`, `.scratch/`, `*.npz`); re-verify the original before submission — a stray `git add -A` without it would add clips. |
| C5 | Research-derived results separation | FAIL (minor) | `evaluation/asl_citizen_predictions.csv` (1,705 rows, `participant` IDs `signer_e0f4874f`-style, 77 KB) **is** tracked. Per-participant rows are personal-adjacent metadata: confirm ASL Citizen research-licence terms allow redistribution, or strip/hash the participant column (fix D7). The `build_bundle.py` `--extra` allowlist (`src/models/scripts/evaluation`) still permits copying research fetchers into the submission ZIP if passed as `--extra` — must be filtered or excluded by not passing them. |
| C6 | Bundle path traversal / symlink safety | PASS with one bypass | `build_bundle.safe_member` rejects absolute names, `..`, and drive letters; `verify_zip` rejects symlink members (`S_ISLNK` check before `extractall`); `build()` resolves `extra_path` and requires it under the source tree, so an external symlink target is rejected (verified). Bypass: `safe_member("data:file")` returns `True` because only `parts[0].endswith(":")` is checked — colon-bearing names pass on any platform (repro D3). Low risk on POSIX; Windows NTFS `file:stream` interpretation is the concern. Fix in `fixes.py`. |
| C7 | Manifest path handling | FAIL (low risk, deliberate capability?) | `run_manifest.read_manifest` resolves relative paths from the manifest dir but accepts `../` and absolute paths anywhere on disk (repro D2). Local operator-supplied evaluator ⇒ low risk, but containment is the correct default for a submission-time tool. Fix in `fixes.py`. |
| C8 | Corrupted-clip cleanup (robustness) | FAIL | `robustness.py:45` `shutil.rmtree(workdir)` is not in a `try/finally`; a decode/write/write exception aborts before it and leaves derivative corrupted clips in `<output-dir>/work/` (repro D1). `src/README.md` claims "temporary corrupted clips are deleted" — true on the happy path only. Fix in `fixes.py`. |
| C9 | Evidence hygiene (paths + hostname) | FAIL | `run_manifest.py:324-334` writes absolute `path.resolve()` strings for predictor/labels/model plus `platform.node()` (operator hostname) as `device_model` into `benchmark.json`. That file must not be pasted into submission notes or results without redaction. Fix in `fixes.py` (relative paths, generic device label). |
| C10 | Absolute paths in code | PASS (research-only) | `C:\Users\dev25\…` site-packages append + `E:\samsung\.scratch\…` in `evaluation/eval_wlasl_signers.py`, `eval_asl_citizen.py` and `.scratch` relatives in `scripts/ml/export_tflite_tf.py` — research scripts only, never imported by the shipped entry point. A naïve "no absolute paths" CI gate would flag them; gate should scope to `src/` + `app/` (which use `Path(__file__)`-relative roots throughout — verified). `$TMP/echogesture-.../models` in `test_model_contract.py` is an optional skip-if-absent path, not an absolute write. |
| C11 | On-device model licensing | NOT_CLEARED — do not ship yet | See §2. |

---

## 2. Licensing and attribution — what must not be overclaimed

There is **no `LICENSE`/`NOTICE` file in either copy**, yet `README.md:167`
states "Our code and harnesses: MIT". That sentence has no licence text
behind it: either add the MIT text for the repo's own code or drop the
claim. Document these contexts accurately (do not upgrade any of them):

| Asset | Context | What to do before shipping |
|---|---|---|
| Repo code/harnesses | README says MIT; no licence text exists | Add `LICENSE` (MIT) covering only repo-authored code, or remove the MIT claim. GPL/Apache alternatives are fine if added explicitly. |
| `models/hand_landmarker.task` (7.5 MB) | Google MediaPipe model blob | Apache-2.0/NOTICE attribution and a provenance note in the submission (source URL, version, SHA `fbc2a300…`). Checked in as a binary blob — carry its licence text alongside it. Verify current MediaPipe terms at submission time. |
| S3D weights `ptq_100_int8.tflite`, `s3d_conv1d_qat_int8.tflite` | README attributes MIT to Kimpinde & Olukanmi, *Algorithms* 2026. But the weights were trained on WLASL, which `README.md:171` itself notes is **C-UDA academic/non-commercial and cannot be shipped commercially** | Confirm the MIT grant actually covers redistribution of these specific weight files, carry the paper citation, and do not claim commercial shippability. If the evaluator requires a commercially shippable model, neither S3D weight is safe under that reading. |
| Dev landmark classifier `landmark_classifier_fp16/int8.tflite` + `landmark_labels.json` (ASL Citizen-derived) | ASL Citizen is **Microsoft research licence (non-commercial, no redistribution)**. A model *trained on* ASL Citizen-derived data inherits the research-only context: evaluation is what the licence contemplates; redistribution/shipping of trained weights and their vocab must be confirmed with Dev and the organizer — **do not relabel as MIT/freely licensable** | Get explicit Dev + organizer written confirmation before including either weight in the submission ZIP. `build_bundle.py` already enforces total model bytes vs. the 20 MB cap, but that is size, not licence. |
| WLASL / ASL Citizen *data* | Both non-commercial, non-redistributable | Keep clips, splits, and landmark caches outside Git and outside the ZIP (workspace). `fetch_smoke_clip.py` / `fetch_validation_batch.py` correctly default to operator output dirs outside the repo. Any validation manifest containing private clip paths must also stay outside Git (`src/README.md` already says so). |

**No approval is implied by this audit.** Green checks above mean
"verified in code", not "approved by the organizer or any licensor".

---

## 3. Defect reproductions (all in `evaluation/device/privacy_lab/reproduce_defects.py`)

| ID | Severity | Reproduction command | Expected vs actual |
|---|---|---|---|
| D1 | Medium | `python evaluation/device/privacy_lab/reproduce_defects.py` → `D1-robustness-workdir-leak reproduced: true` | `paired_evaluate` on an unreadable clip raises `ValueError`; `<output-dir>/work/` survives with derivative files. Expected: empty/absent. |
| D2 | Low | `D2-manifest-path-traversal` | `c1,../outside.mp4` resolves outside the manifest dir and is accepted. Expected: containment error. |
| D3 | Low | `D3-bundle-safe-member-colon` (`safe_member("data:file") == True`) | Colon-bearing ZIP members pass the safety check. Expected: `False`. |
| D4 | Low | `D4-temp-dir-shared` | Temp capture `echogesture-capture-*.mp4` lands in the shared system temp dir (`/var/folders/…/T` on macOS). 0600 + O_EXCL from `NamedTemporaryFile` blocks name-guessing, but the name is world-listable. Expected (release): private 0700 subdir. |
| D5 | Control | `D5-temp-deleted-on-exception` | Temp clip correctly deleted even when `predict()` raises — the on-disk deletion claim holds on the happy path and on exception. **Only SIGKILL/crash leaks** (no user-space code can clean those). |
| D6 | Low | `D6-hostname-in-benchmark` | `benchmark.json` embeds `platform.node()` + absolute artifact paths. Expected: generic label + relative paths. |
| D7 | &lt;Low | `signer_ea80f856-research-data` (audit finding, not a runtime repro) | Tracked `asl_citizen_predictions.csv` exposes participant IDs. Expected: stripped/hashed, or terms-confirmed redistribution. |

Unit tests: `python -m unittest evaluation.device.privacy_lab.test_privacy_lab -v` → 7 tests OK (pytest equivalent). `test_spike.py` is a temporary spike asserting D5; delete it once temp handling is reworked.

---

## 4. Integration readiness and handoff

**Owned by A15 (this copy only):** `evaluation/device/privacy_lab/` and this file. Nothing outside those paths was edited; the original checkout was read-only at audit time.

**Suggested fixes** for Abhijeet-owned files are in `evaluation/device/privacy_lab/fixes.py` (proposed diffs in comments, ready to apply):

1. `robustness.py` — `try/finally` around the corrupted-clip loop (closes D1).
2. `run_manifest.py` — manifest-directory containment (closes D2); generic `device_model` + relative artifact paths (closes D6).
3. `build_bundle.py` — colon rejection in `safe_member` (closes D3); consider excluding research fetchers from the `--extra` allowlist (closes C5 partially).
4. `offline_demo.py` — private 0700 temp subdir (closes D4); soften "no footage retained" wording to the on-disk truth (closes C3 wording).
5. Repo root — add `LICENSE` (MIT, repo code only) + `NOTICE` with per-asset provenance (§2); strip/hash participant IDs in `asl_citizen_predictions.csv` or confirm redistribution terms (closes C5/D7).

**Blockers (human / organizer, not code):**

- Official vocabulary, manifest, evaluator starter kit, and final submission form — none received, so vocabulary/training-source legality cannot be finally checked.
- INT8 vs FP16 requirement and whether `hand_landmarker.task` bytes count toward the 20 MB cap.
- Written permission/confirmation for shipping ASL Citizen-derived (landmark) and WLASL-derived (S3D) weights, and the correct licence texts for code + MediaPipe blob.
- Registration, target device, and demo permission (macOS camera denied; no Android SDK/device).

Until those are resolved, the correct label for this release is
**not release-certified**, matching `docs/release/FINAL_EVIDENCE.md`.

**Files created (this copy):**

- `evaluation/device/privacy_lab/audit.py`
- `evaluation/device/privacy_lab/reproduce_defects.py`
- `evaluation/device/privacy_lab/fixes.py`
- `evaluation/device/privacy_lab/test_privacy_lab.py`
- `evaluation/device/privacy_lab/test_spike.py`
- `evaluation/device/privacy_lab/README.md`
- `docs/release/COMPLIANCE_AUDIT.md` (this file)

**Commands run (audited copy `$TMP/echogesture-.../A15`, Python 3.11 venv):**

```
python evaluation/device/privacy_lab/audit.py --root . --out evaluation/device/privacy_lab/findings.json
python evaluation/device/privacy_lab/reproduce_defects.py --out evaluation/device/privacy_lab/reproductions.json
python -m unittest evaluation.device.privacy_lab.test_privacy_lab -v   # 7 OK
python -m unittest evaluation.device.privacy_lab.test_spike -v         # 1 OK
```
