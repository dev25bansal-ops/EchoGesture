# EchoGesture — final compliance-gate report (A04)

Scope: privacy, sanitization, licensing, and bundle-extraction safety for the
Track 2 bundle. This report documents the reusable checker, its verified
results on the live tree, and the licensing facts that remain legally
unresolved. It is not legal advice and it grants nothing.

**Verdict: NOT release-certified.** The safety checks are green; the licensing
gates remain open (see "Unresolved legal gates" below).

---

## 1. What the checker is

`scripts/release/compliance_final.py` is a single-module, stdlib-only,
read-only, `python3 -I`-safe sweep callable by the lead during bundle
verification:

```
python3 -I scripts/release/compliance_final.py --root . [--zip path/to/bundle.zip] --out /tmp/findings.json
```

* Exit 0 on a completed sweep (FAILs and NOT_CLEARED included); exit 2 only on
  an audit error; stdout is JSON; `--out` writes the same JSON atomically.
* It composes the existing hardened primitives rather than reimplementing
  them: `build_bundle.safe_member` for ZIP member safety, and
  `evaluation/device/v8_privacy/sanitize_evidence` semantics for participant
  ID / hostname / absolute-path redaction.
* It never recurses into `.git`, `.venv`, `__pycache__`, `.scratch`, `tmp`, or
  downloaded research-media subtrees, and it never reads text files over 5 MB.
* It performs no network access and never writes into the audited tree.

Eleven checks (one finding each): `C01` offline inference, `C02` secrets,
`C03` evidence hygiene (participant IDs / usernames / hostnames / absolute
paths), `C04` manifest containment, `C05` ZIP extraction safety, `C06`
temporary media, `C07` screenshot policy, `C08` raw footage, `C09` restricted
datasets, `C10` LICENSE/NOTICE presence, `C11` model licensing.

---

## 2. Live-tree results (2026-11-10, branch `hackathon/abhijeet-integration`)

`python3 -I scripts/release/compliance_final.py --root .` -> exit 0,
`checks=11 pass=6 fail=4 not_cleared=1`:

| Check | Result | Meaning |
|---|---|---|
| C01 offline-inference | PASS | No network/cloud imports in `src/`+`app/`; the only URL literal is the MediaPipe download-hint error string in `src/landmark_frontend.py:62`. |
| C02 secrets | PASS | No credential patterns in `src/`, `app/`, `scripts/`, or label JSONs. Excludes the ASL gloss `secretary` and the scanner's own regex source. |
| C03 evidence-hygiene | FAIL | Raw participant IDs (`signer_e0f4874f`-style) remain in three tracked ASL Citizen files; see §4. No hostname or absolute-path leaks in `results/results.csv`. |
| C04 manifest-paths | PASS | `run_manifest.py` resolves clips inside an allowed input root and writes a generic (non-hostname) device model. |
| C05 zip-safety | PASS | The four committed provisional zips have no traversal/absolute/symlink/hardlink/external-attr/duplicate members and stay under the size caps (max member 20 MB, total 100 MB). |
| C06 temp-media | PASS | `app/offline_demo.py` writes one temp clip via `mkstemp` in a private `TemporaryDirectory` (0700) and unlinks it in `finally` (success and predictor exception). |
| C07 screenshot-policy | FAIL | `app/demo_lab/__init__.py` defaults HUD PNG + JSON to persistent `tempfile.gettempdir()/echogesture-demo-lab` with no cleanup. |
| C08 raw-footage | PASS | No tracked raw video/array caches or screenshots; on-disk research media (`v5_accuracy/*.mp4`, `v7_android/*.npz`) is untracked and skipped from scans. |
| C09 restricted-data | FAIL | `evaluation/asl_citizen_predictions.csv` still carries a `participant` column; `asl_citizen_{ptq,qat}_results.json` embed `per_signer` dicts. |
| C10 license-notice | FAIL | No `LICENSE`/`NOTICE` at repo root (templates exist only under `evaluation/device/v8_privacy/`). |
| C11 model-licensing | NOT_CLEARED | MediaPipe detector is Apache-2.0 (re-verify at submission); the S3D and ASL Citizen-derived weights are not redistributable without written permission. |

The checker can also be pointed at an explicit bundle:

```
python3 -I scripts/release/compliance_final.py --root . --zip scripts/release/v8_release/provisional.zip
```

This re-runs the extraction-safety checks on that ZIP plus the committed ones.

---

## 3. Fixes applied (A2, 2026-11-10)

The four fixable gates are now closed by code; only the rights gate remains.

1. **C03/C09 — participant-ID hygiene.** New
   `scripts/release/sanitize_export.py` (stdlib only, `-I`-safe) reads the
   tracked originals and writes publish-safe copies to
   `evaluation/device/compliance_final/publish/`:
   * `asl_citizen_predictions.csv`: the `participant` column is dropped by
     default, or replaced by a deterministic keyed `signer_pseudonym`
     (`--key`) when local linkage is needed;
   * `asl_citizen_{ptq,qat}_results.json`: `per_signer` keys become
     `signer_redacted_<n>` (default) or keyed `signer_<hash8>` (with `--key`);
   * it refuses (raises) if any emitted value still carries a raw `P<digits>`
     token, hostname suffix, or absolute path, and never writes to the source.

   The originals are unchanged on disk (local research inputs, NOTICE section 3).
   The checker's default C03/C09 now audits the publish surface; `--raw`
   audits the originals and still reports the raw IDs as local-only.

2. **C07 — demo media retention.** `app/demo_lab/__init__.py` no longer
   defaults to `tempfile.gettempdir()/echogesture-demo-lab`. Without
   `--outdir` it writes into a private per-run `TemporaryDirectory` that is
   removed when the command exits; explicit `--outdir` still persists (opt-in).
   `run_video`, `--video`, and `--check` keep their existing behavior.

3. **C10 — LICENSE/NOTICE.** Root `LICENSE` (MIT, repo-authored code only) and
   root `NOTICE` (per-artifact provenance table, dependency licenses from
   installed metadata, and a permission-request draft). C10 passes.

4. **C11 / `--strict`.** `compliance_final.py --strict` exits 1 when any
   mandatory gate is FAIL / NOT_CLEARED / NOT_MEASURED; default stays
   report-only exit 0. While C11 is unresolved it prints
   `NOT legally cleared for distribution` on stderr. C11 clears only when a
   NOTICE row carries the machine-checkable `AUTHORIZATION RECORDED` marker;
   no such marker exists for the restricted weights today.

---

## 4. Tests

`evaluation/device/compliance_final/test_compliance_final.py` — 49 focused,
deterministic tests against a synthetic temp directory (no repo artifacts, no
models, no network), plus `test_demo_lab_retention.py` — 4 retention tests.
Verified with `$REPO/.venv/bin/python -I`:

```
python -I evaluation/device/compliance_final/test_compliance_final.py
# Ran 49 tests — OK
python -I evaluation/device/compliance_final/test_demo_lab_retention.py
# Ran 4 tests — OK
python -I evaluation/device/test_offline_demo.py
# Ran 17 tests — OK   (existing demo API unbroken)
```

Covered: offline import detection, secret detection excluding the `secretary`
gloss, participant-ID vs. latency-percentile disambiguation (`signer_e0f4874f` matches,
`p50`/`p95`/`p99` do not), publish-surface vs. raw-originals evidence audits,
manifest containment, ZIP traversal / absolute / duplicate / symlink / size
rejection, temp-media and screenshot policy, LICENSE/NOTICE provenance and
C11 authorization-marker transitions, strict-gate exit codes (nonzero on the
current unresolved state, zero on a fully-authorized fixture, nonzero on a
FAIL, report-only exit 0 by default), sanitizer determinism, source-immutability,
hostname/absolute-path cleaning, and the demo-lab ephemeral-default contract.

---

## 5. Current gate state (live tree)

`python -I scripts/release/compliance_final.py --root .` -> exit 0,
`checks=11 pass=10 fail=0 not_cleared=1` (C11). `--strict` exits 1 with
`NOT legally cleared for distribution: C11-model-licensing unresolved`.

---
   carry provenance (source, version, SHA-256) in the submission.
5. Repo root `LICENSE`/`NOTICE` still missing (templates below).

---

## 5. Verified local licenses (audited, source-cited)

Installed packages in `.venv` (`importlib.metadata` + `METADATA`/`LICENSE`
files), as of this run:

| Package | Version | License | Local evidence |
|---|---|---|---|
| mediapipe | 1.2.2 | Apache-2.0 | `mediapipe-1.2.2.dist-info/licenses/LICENSE` |
| ai-edge-litert | 2.3.0 | Apache-2.0 | `ai_edge_litert-2.3.0.dist-info/licenses/LICENSE` |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | `numpy-2.4.6.dist-info/{LICENSE.txt, licenses/…}` |
| opencv-python | 5.0.0.93 | Apache-2.0 (wrapper is MIT) | `cv2/LICENSE.txt`, `cv2/LICENSE-3RD-PARTY.txt` |

Model artifacts in `models/`:

| Asset | Provenance / terms (from existing local docs) | Status |
|---|---|---|
| `hand_landmarker.task` (7.5 MB) | Google MediaPipe; source URL `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`; SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`. Apache-2.0 per MediaPipe terms. | NOT_CLEARED — re-verify current MediaPipe terms at submission; carry attribution. |
| `landmark_classifier_fp16.tflite` (SHA `a012f2b5…`), `landmark_classifier_int8.tflite` (SHA `9d24640a…`), `landmark_labels.json` | ASL Citizen-derived (Microsoft Research licence: non-commercial, no redistribution). Trained weights inherit the research-only context. | RESTRICTED — explicit Dev + organizer written confirmation required before shipping either weight. |
| `s3d_conv1d_qat_int8.tflite` (SHA `107756f3…`), `ptq_100_int8.tflite` (SHA `067c743c…`), `label100.json` | S3D-Conv1D (Kimpinde & Olukanmi, *Algorithms* 2026; MIT attributed in README) but trained on WLASL (C-UDA academic/non-commercial). | NOT_CLEARED — confirm the MIT grant covers these specific weight files; WLASL restriction reattaches commercially. |

---

## 6. License / NOTICE draft material (accurate only; grants nothing)

`LICENSE` (MIT, repo-authored code and harnesses only — never model assets or
datasets):

```
MIT License

Copyright (c) 2026 EchoGesture contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software") in
repo-authored code and harnesses under this repository (including src/, app/,
evaluation/device/ harnesses, and scripts/release/), to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`NOTICE` — draft attribution blocks only (do not claim any right beyond what
each licensor grants):

* Repo code: MIT (see LICENSE; does not extend to model assets or datasets).
* `hand_landmarker.task`: Google MediaPipe, Apache-2.0; source URL and SHA-256
  as in §5; carry the MediaPipe-required attribution; re-verify terms at
  submission.
* S3D weights: cite Kimpinde & Olukanmi, *Algorithms* 2026; record that the
  MIT claim is source-attributed and must be confirmed; WLASL training data is
  C-UDA non-commercial — do not claim commercial shippability.
* Landmark classifier (`landmark_classifier_*` + `landmark_labels.json`):
  trained on ASL Citizen (Microsoft Research licence, non-commercial, no
  redistribution); not MIT, not freely shippable.
* WLASL / ASL Citizen data: non-redistributable; clips, splits, caches, and
  per-participant rows stay outside Git and outside the bundle.
* Python deps: numpy (BSD-3-Clause), opencv-python (Apache-2.0),
  ai-edge-litert (Apache-2.0), mediapipe (Apache-2.0) — attribution by SPDX id.

---

## 7. Unresolved legal gates (human/organizer — not code)

These block release certification; the checker reports each as FAIL or
NOT_CLEARED rather than overclaiming:

1. **ASL Citizen-derived landmark weights** — no written permission to
   redistribute/ship `landmark_classifier_fp16.tflite`,
   `landmark_classifier_int8.tflite`, or `landmark_labels.json`.
2. **WLASL-derived S3D weights** — confirm the MIT grant from Kimpinde &
   Olukanmi actually covers redistribution of `s3d_conv1d_qat_int8.tflite`
   and `ptq_100_int8.tflite`; WLASL is C-UDA non-commercial.
3. **MediaPipe `hand_landmarker.task`** — Apache-2.0 context is locally
   documented, but the current MediaPipe terms and attribution must be
   re-verified at submission time and recorded beside the blob.
4. **Per-participant metadata** — `asl_citizen_predictions.csv` participant
   column and the `per_signer` JSONs must be stripped/pseudonymised or have
   confirmed redistribution terms.
5. **Official vocabulary / manifest / evaluator starter kit** — not yet
   received, so final vocabulary and training-source legality cannot be
   checked.

### Permission-request draft (concise)

> Subject: Redistribution confirmation for EchoGesture Track 2 submission
>
> We are an on-device sign-language-recognition submission ("EchoGesture").
> Our bundle would include, for evaluation only: (1) a landmark classification
> model trained on ASL Citizen, and (2) INT8 weights derived from the
> S3D-Conv1D architecture trained on WLASL. Before we include either model
> file in the submission archive, please confirm in writing whether we may
> redistribute the following specific files for competition evaluation:
>
> * models/landmark_classifier_fp16.tflite and
>   models/landmark_classifier_int8.tflite (+ models/landmark_labels.json)
> * models/s3d_conv1d_qat_int8.tflite and models/ptq_100_int8.tflite
>
> If permission is not granted for redistribution, we will instead keep these
> assets private/local-only and reference them by SHA-256 in our submission.

### Private/local-only fallback (apply if permission is denied)

If no written permission is obtained, the correct release posture is a
**local-only / private evaluation release**: do not ship the ASL Citizen- or
WLASL-derived weights in the public bundle; document them by SHA-256 only.
The MediaPipe detector (Apache-2.0) may still ship with attribution; the
repo-authored code remains MIT.

---

## 8. Files this agent created / changed (owned paths only)

Created:
* `scripts/release/sanitize_export.py` — publish-safe evidence exporter.
* `LICENSE` — root MIT, repo-authored code only.
* `NOTICE` — per-artifact provenance + permission-request draft.
* `evaluation/device/compliance_final/publish/` — sanitised exports
  (`asl_citizen_predictions.csv`, `asl_citizen_ptq_results.json`,
  `asl_citizen_qat_results.json`).
* `evaluation/device/compliance_final/test_demo_lab_retention.py` — 4 tests.
* `docs/release/COMPLIANCE_FINAL.md` (this report).

Changed:
* `scripts/release/compliance_final.py` — publish-surface C03/C09, real C07
  detection, per-asset C10, authorization-marker C11, `--strict`, `--raw`.
* `evaluation/device/compliance_final/test_compliance_final.py` — 49 tests.
* `app/demo_lab/__init__.py` — ephemeral default output only (retention fix).

Not touched: `app/offline_demo.py`, `src/`, `models/`, `results/results.csv`,
`scripts/release/build_bundle.py`, `scripts/release/release_build.py`,
`app/android_final/`, `app/v7_demo/reliable.py`, and the three tracked
`evaluation/asl_citizen_*` originals (byte-for-byte unchanged; no restricted
asset was published, removed, or moved).