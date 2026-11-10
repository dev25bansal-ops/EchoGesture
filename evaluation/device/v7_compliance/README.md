# EchoGesture V7 Compliance — Privacy, Licensing & Security Audit (A15)

**Verdict: NOT READY FOR SUBMISSION. Track-2 licence review and Organizer
vocabulary/starter-kit are still unresolved. No submission ZIP ships until the
blockers below are cleared.**

Work confined to `evaluation/device/v7_compliance/`. Read-only scan of the
audited tree; no network calls; no installs. Python 3.11 stdlib only
(`python3 -I`). Verified against **live code** in `/tmp/echogesture-v7-agents/A15`
on 9 Nov 2026. Prior `evaluation/device/privacy_lab/` findings were re-checked
live — two of them are stale (marked CLOSED below) and must not be cited as open.

## Verified checklist (live code)

| # | Check | Result | Evidence |
|---|---|---|---|
| V1 | Shipped `src/`+`app/` offline (no Firebase/cloud/urllib/requests/socket) | **PASS** | 0 network sinks in `src/`, `app/`. `src/main.py` + `app/offline_demo.py` run TFLite/MediaPipe locally. The only `https` text in shipped code is the `hand_landmarker.task` download hint in `src/landmark_frontend.py` error message (not a call). Research-only fetchers (`evaluation/eval_asl_citizen.py`, `evaluation/device/fetch_smoke_clip.py` → `urllib.request` to `download.microsoft.com`) are never imported by `src/main.py`, `app/offline_demo.py`, or `scripts/release/build_bundle.py`. Verify: `grep -rnE 'urllib|requests|socket|firebase' src/ app/` |
| V2 | No hardcoded secrets | **PASS** | `api[_-]?key/client_secret/Bearer/PRIVATE KEY` scan over repo (excl. ASL gloss `secretary`, excl. this lab's own regex sources) finds nothing. |
| V3 | Demo temp-clip deletion | **PASS** (on-disk); nuance noted | `app/offline_demo.py` `PredictionWorker._run()`: `write_temp_clip()` → `predict()` → `finally: path.unlink(missing_ok=True)`. Temp dirs are `TemporaryDirectory(prefix="echogesture-capture-")` = `mkdtemp` mode 0700 — a private subdir, not loose files in shared `/tmp`. Deleted on success **and** on predictor exception. Residual: SIGKILL/crash orphans the clip; frames sit in RAM (deque + `list(captured)`) during inference, never explicitly scrubbed. Public wording "No footage is retained" is on-disk-true; recommend: "no footage kept on disk; frames in memory only during inference". |
| V3b | demo_lab output persistence | **FAIL** (low) | `app/demo_lab/__init__.py:180` defaults outdir to `/tmp/echogesture-demo-lab` (`gettempdir`, no cleanup); HUD PNG renders a video frame, so video-derived output persists in shared `/tmp`. Operator-run local demo — document it, or switch default to a private `TemporaryDirectory`. |
| V4 | Evaluator retains no video | **PASS** | `src/main.py` `predict()` opens `video_path` in place (`cv2.VideoCapture`, no `VideoWriter`, no copy, no write). Nothing to delete. |
| V5 | Bundle ZIP contains no clips/research code | **PASS** (actual zips) | Both `provisional.zip`s contain 13 members: `src/`, `models/`, one export script, `results.csv`, `Dockerfile`, `release.json`. No `.mp4`, no `asl_citizen_predictions.csv`, no `fetch_*`/`eval_*`. Residual: `build_bundle --extra` allowlist still includes `evaluation/` roots, so a future mis-passed `--extra` could copy research fetchers into a new ZIP — filter or trim. |
| V6 | Robustness corrupted-clip cleanup | **PASS** — prior D1 **CLOSED** | `evaluation/device/robustness.py:paired_evaluate` wraps the clip loop in `try/finally: shutil.rmtree(workdir, ignore_errors=True)`. `privacy_lab/findings.json` D1 is stale. |
| V7 | Manifest path containment | **FAIL** (low, still open) | `run_manifest.read_manifest` resolves relative paths from the manifest dir but accepts `../` and absolute paths anywhere. Low (local operator-supplied evaluator); add a parents-check or document as deliberate operator capability. |
| V8 | Evidence hostname/path leak | **FAIL** (still open, = prior D6) | `run_manifest.py` writes `platform.node()` as `device_model` plus absolute `path.resolve()` artifact paths into `benchmark.json`. Redact (generic device label, repo-relative paths) before any evidence leaves the machine. |
| V9 | Bundle `safe_member` traversal/symlink | **PASS** — prior D3 **CLOSED** | `safe_member` now rejects `":" in parts[0]` (plus absolute, `..`, drive letters, backslashes). `safe_member("data:file") == False`, `("ok/file") == True`. `verify_zip` rejects symlink members. |
| V10 | LICENSE file vs. README MIT claim | **FAIL** | No `LICENSE`/`NOTICE` in repo root; `README.md` "Our code and harnesses: MIT" has no licence text behind it. Add MIT text for repo-authored code only, or drop the claim. |
| V11 | Model/asset licensing & attribution | **NOT_CLEARED** | See §2 below. Do not ship any weight until Organizer + licence-holder confirmation. |
| V12 | Participant metadata in tracked CSV | **FAIL** (needs terms check) | `evaluation/asl_citizen_predictions.csv` (1,706 lines; `participant` column `P15`-style) is in the tree. Confirm ASL Citizen terms allow redistributing per-participant result rows, else strip/hash the column before submission. |
| V13 | Absolute local paths in research scripts | **FAIL** (research-only) | `C:\Users\dev25\…` site-packages appends (`eval_*.py`), `/Users/…/.venv` paths in README/qa scripts. Shipped `src/`+`app/` are `Path(__file__)`-relative (PASS for evaluator). Keep out of submission ZIP; scope any "no absolute paths" gate to `src/`+`app/`. Prior `privacy_lab` JSONs also embed their own `/private/tmp/…` roots — regenerate or exclude from submission. |

## 2. Licensing — must not be overclaimed

| Asset | Context | Required before shipping |
|---|---|---|
| Repo code/harnesses | README says MIT; no licence text | Add `LICENSE` (MIT) covering only repo-authored code, or remove the MIT claim. |
| `models/hand_landmarker.task` (7.5 MB) | Google MediaPipe blob | Apache-2.0 attribution + provenance note (source URL, version, SHA) alongside the blob; verify current MediaPipe terms at submission time. |
| Dev landmark classifier `landmark_classifier_fp16/int8.tflite` + `landmark_labels.json` | ASL Citizen-derived; Microsoft research licence = non-commercial, no redistribution | Trained weights inherit the research-only context: evaluation only; do **not** relabel MIT/freely licensable or ship commercially. Get explicit Dev + Organizer written confirmation before including either weight in the ZIP. |
| S3D weights `ptq_100_int8.tflite`, `s3d_conv1d_qat_int8.tflite` | README cites MIT (Kimpinde & Olukanmi, *Algorithms* 2026), but weights were trained on WLASL C-UDA (academic/non-commercial, no commercial shipping) | Confirm the MIT grant covers redistribution of these weight files; carry the paper citation; do not claim commercial shippability. |
| WLASL / ASL Citizen *data* | Non-commercial, non-redistributable | Keep clips, splits, caches outside Git and outside the ZIP. `fetch_*.py` correctly default to operator output dirs outside the repo. Any validation manifest with private clip paths stays outside Git. |

No approval is implied by this audit. Green checks mean "verified in live code",
not "approved by the Organizer or any licensor".

## 3. Reproduction / verification

```bash
V=.venv/bin/python  # any Python 3.11; only stdlib is needed by this lab
$V -I evaluation/device/v7_compliance/v7_audit.py --root . \
  --out evaluation/device/v7_compliance/findings.json
$V -I evaluation/device/v7_compliance/test_v7_audit.py -v
```

`v7_audit.py` exits 0 when the sweep completes (findings may include FAILs);
exit 2 only if the audit itself errors. `test_v7_audit.py` pins the live-code
claims (V1–V9, D1/D3 CLOSED, D2/D6 OPEN, licensing NOT_CLEARED) so a later fix
can be verified by test. Result at audit time: **14 checks, 7 pass, 6 fail,
1 not_cleared; 12/12 unit tests OK.**

Two prior `privacy_lab` claims were *disproved* on live code and are flagged
stale here: D1 (robustness workdir leak — now `try/finally`) and D3 (bundle
colon bypass — now `":" in parts[0]` rejected). That lab's `NamedTemporaryFile`
description also no longer matches `offline_demo.py` (now `TemporaryDirectory`
+ `write_temp_clip`). Its stored `findings.json`/`reproductions.json` predate
the fixes and embed absolute `/private/tmp/…` roots.

## 4. Handoff & blockers

**Files owned here:** `evaluation/device/v7_compliance/{v7_audit.py, test_v7_audit.py, findings.json, README.md}`. Nothing outside this directory was edited.

**Concrete defects needing owner fixes (proposed diffs in `../privacy_lab/fixes.py`, re-verify with this lab):**

1. `run_manifest.py` — manifest containment (V7/D2) + generic `device_model` / relative artifact paths (V8/D6).
2. `app/demo_lab/__init__.py` — private outdir default or documented retention (V3b).
3. Repo root — add `LICENSE` + `NOTICE` (V10) and per-asset provenance for `hand_landmarker.task` + weights (V11).
4. `evaluation/asl_citizen_predictions.csv` — strip/hash `participant` or confirm redistribution terms (V12).
5. `build_bundle --extra` allowlist — exclude research fetchers from `--extra` (V5 residual).

**Blockers (human/Organizer, not code):**

- Official vocabulary, manifest, evaluator starter kit, and submission form — not received; vocabulary/training-source legality cannot be finally checked.
- Written Organizer + licence-holder permission for ASL Citizen-derived (landmark) and WLASL-derived (S3D) weights.
- Whether `hand_landmarker.task` bytes count toward the 20 MB model cap; INT8 vs FP16 requirement.
- Registration / target device / demo permission (no Android SDK/device in this environment).
