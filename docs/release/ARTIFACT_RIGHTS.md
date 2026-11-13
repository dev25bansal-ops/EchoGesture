# Artifact rights register

Scope: every artifact this submission ships, links, or relies on. This is a
project-internal compliance record, not legal advice. It records what the
repository can and cannot prove, and marks "no documentary evidence in repo"
wherever that is the honest answer.

Source documents consulted: `NOTICE`, `LICENSE`, `models/README.md`,
`README.md` §Licensing, `docs/release/COMPLIANCE_AUDIT.md`,
`docs/release/COMPLIANCE_FINAL.md`, `submission_final/RELEASE_MANIFEST.json`,
and the installed package metadata under `.venv/lib/python3.11/site-packages/`.

Compiled 2026-11-11. Artifacts and hashes are as they exist in this checkout.

## Summary

| Artifact | Origin | Licence/permission evidence | Allowed internal use | Allowed redistribution | Status |
| --- | --- | --- | --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` (939,152 B, `112279bc…`) | Trained by Dev on ASL Citizen-derived features | `NOTICE` §2: trained on ASL Citizen-derived data; **no written permission on file** | Yes (research/evaluation, the use ASL Citizen's licence contemplates) | **No** — no documentary evidence in repo | BLOCKED for publication |
| `landmark_ens_tcnm61_fp16.tflite` (832,888 B, `01a261d8…`) | Same lineage, mirror-augmented | Same | Yes | **No** — no documentary evidence in repo | BLOCKED for publication |
| `landmark_ens_tcnm62_fp16.tflite` (832,888 B, `27058a85…`) | Same lineage, mirror-augmented | Same | Yes | **No** — no documentary evidence in repo | BLOCKED for publication |
| `landmark_ens_rob71_fp16.tflite` (939,152 B, `08417e72…`) | Same lineage, stronger augmentation | Same | Yes | **No** — no documentary evidence in repo | BLOCKED for publication |
| `models/hand_landmarker.task` (7,819,105 B, `fbc2a300…`) | Published by Google MediaPipe at `storage.googleapis.com/mediapipe-models/.../float16/1/hand_landmarker.task` | Apache-2.0 claimed in `NOTICE` §2 from MediaPipe terms; `mediapipe` 1.2.2 wheel carries `Apache License 2.0` text in `dist-info/licenses/LICENSE`. The **detector binary's own** licence file is not vendored in this repo. | Yes | Apache-2.0 permits redistribution **with attribution + licence text**; that attribution is not currently vendored alongside the file | PERMITTED with attribution (attribution not yet shipped) |
| APK `app/android_final/app/build/outputs/apk/debug/app-debug.apk` (77,710,351 B, `530f5ec4…`, commit `cc00f6a`) | Built from this repo; embeds all four ensemble members (`assets/landmark_ens_*.tflite`), `hand_landmarker.task`, and `landmark_labels.json` | Composite: MIT (repo code) + Apache-2.0 (MediaPipe JNI, detector) + unresolved (ensemble weights, labels) | Yes | **No** as built — it contains the four unresolved weight files and their vocabulary | BLOCKED for publication |
| Release ZIP `evaluation/device/release_final2/echogesture_final2_ensemble.zip` (9,007,144 B, `34a0e667…`) | Built by `scripts/release/build_bundle.py` from this checkout; `official_submission: false` | Composite: MIT (repo code + docs) + Apache-2.0 (detector) + unresolved (four ensemble members, `landmark_labels.json`) | Yes | **No** — same four unresolved members and their vocabulary; no documentary evidence of permission | BLOCKED for publication |
| ASL Citizen dataset (clips, split CSVs, landmark caches) | Microsoft Research | `NOTICE` §3: "Microsoft Research licence, non-commercial, no redistribution"; `README.md` §Licensing: "non-commercial … may not be redistributed". The upstream licence text is **not** vendored in the repo. | Yes, research/benchmarking only | **No** — not redistributable | BLOCKED; kept out of Git and out of the ZIP |
| WLASL dataset (S3D baseline lineage) | WLASL authors | `NOTICE` §3 / `README.md`: C-UDA, academic / non-commercial; licence text **not** vendored | Yes, research | **No** | BLOCKED; S3D weights also unshipped by the ensemble build |
| Training code (`scripts/ml/train.py`, `ensemble.py`, `evaluate.py`, landmark extraction) | Dev-authored | MIT per `LICENSE` (repo-authored code). **Not present on `hackathon/abhijeet-integration`** — it exists only on `origin/hackathon/dev-model`. | Yes | MIT permits redistribution **when present**; not in this branch's tree | PERMITTED (MIT) — absent from this branch |
| Runtime + packaging code (`src/`, `scripts/release/`, `evaluation/device/`) | Repo-authored | MIT per `LICENSE` | Yes | Yes, under MIT | PERMITTED (MIT) |

## Detail and evidence

### 1. The four TFLite ensemble members

`NOTICE` §2 records each as "trained on ASL Citizen-derived data" with
**"NO confirmed redistribution permission … LOCAL/PRIVATE only pending written
authorization"**. `models/README.md` repeats `**no** (restricted)` for each.
`docs/release/COMPLIANCE_FINAL.md` records the same as `NOT_CLEARED`.

There is **no documentary evidence in this repo** — no signed permission, no
written grant, no organizer clause — that these weight files may be published
or included in a publicly distributed submission. The permission-request draft
in `NOTICE` §2a is unsent; the "record the reply" instruction there is
unfulfilled.

**Correction (2026-11-12).** An earlier version of this section said the weights
lived only inside the submission ZIP and that the manifest recorded
`"model_weights_published": false`. That was wrong: the repository is PUBLIC and
`hackathon/dev-model` exposes all four ensemble members at byte-identical SHAs,
plus the detector, the legacy INT8 classifier, and a 1,705-row CSV containing 11
raw participant IDs. The manifest has been corrected.

So the weights are *not* merely inside the ZIP — they are publicly clonable from
the repository today. See `docs/release/PUBLIC_WEIGHT_EXPOSURE.md` for the full
inventory and the remediation options. Whether handing restricted weights to
competition organizers counts as "redistribution" is still a question for the
rights holder, but that is now a second problem, not the only one.

### 2. `hand_landmarker.task`

Apache-2.0 is claimed in `NOTICE` §2 and repeated in `models/README.md:25`. The
supporting evidence available in-repo is the MediaPipe Python package's own
bundled `LICENSE` (Apache License 2.0) in `mediapipe-1.2.2.dist-info/licenses/`.
That establishes the library's licence, not conclusively the `.task` model
binary's. `NOTICE` itself flags "Re-verify current MediaPipe terms and carry the
upstream attribution + source URL at submission."

Apache-2.0 requires the licence text and NOTICE attribution to travel with a
redistributed binary. **No `NOTICE`/`LICENSE` copy is currently staged inside
the release ZIP or the APK** — the ZIP contains `Dockerfile`, `release.json`,
`models/`, `results/`, `scripts/`, `src/` only. This is a fixable gap
(add the Apache-2.0 text + MediaPipe attribution to the bundle), not an
unresolved-rights problem.

### 3. APK and release ZIP

Both are composites. The repo's own code is MIT. Everything else inherits the
status of its parts: the four ensemble members and `landmark_labels.json` are
unresolved (section 1); the detector is Apache-2.0 (section 2). The APK also
bundles `libmediapipe_tasks_vision_jni.so` in three ABIs, which is Apache-2.0
like the Python wheel. Because the unresolved weights are inside both
archives, **neither archive may be publicly redistributed** on the evidence
currently in the repo.

### 4. Datasets

ASL Citizen and WLASL are recorded in `NOTICE` §3 and `README.md` §Licensing as
non-commercial and non-redistributable. Neither upstream licence text is
vendored. Raw clips and landmark caches are correctly kept outside Git and
outside the ZIP (`build_bundle.py` stages an explicit allowlist rooted at
`src/ models/ scripts/ evaluation/`, and the clip data directories are not on
that list). Per-participant rows remain a separate open item flagged in
`COMPLIANCE_AUDIT.md` C5.

### 5. Training code

`LICENSE` is MIT over repo-authored code, which is sufficient documentary
evidence that the training code *may* be redistributed under MIT. Two caveats
recorded as fact: (a) `scripts/ml/train.py` and the full training pipeline are
**not on this branch** — only `scripts/ml/export_tflite_tf.py` is; they exist on
`origin/hackathon/dev-model`; (b) the MIT grant covers the code, not the
weights it produced.

## What is deliberately not claimed

- No artifact is labelled "MIT" except repo-authored code.
- No redistribution right is asserted for any weight file or dataset.
- No organizer approval or target-device certification is implied.
- Where the repo lacks evidence, this document says so rather than inferring a
  permission from the file being present.

## Open actions (owner: Abhijeet, unless noted)

1. Send the `NOTICE` §2a permission request; record any written reply, its
   date and its exact scope here and in `NOTICE`.
2. Vendor the Apache-2.0 licence text and MediaPipe attribution into the release
   bundle and APK, or confirm the organizer accepts a linked NOTICE.
3. Decide repo visibility for `hackathon/abhijeet-public`: a public link exposes
   the unresolved weights (`COMPLIANCE_FINAL.md` marks this BLOCKED).
4. Resolve the tracked per-participant CSV (`COMPLIANCE_AUDIT.md` C5).