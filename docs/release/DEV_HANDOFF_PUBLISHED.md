# Dev handoff — Abhijeet → Dev

**Branch:** `hackathon/abhijeet-public` @ `6f41174`
**PR:** https://github.com/dev25bansal-ops/EchoGesture/pull/1 (draft, base `hackathon/two-person-sprint`)
**Status:** pushed and verified; review requested from `dev25bansal-ops`.

---

## What was published

- `src/main.py` — `predict()`: meta-driven loader, MediaPipe landmarks + 255-feature
  transform, softmax over FP16 logits. Opt-in `ECHOGESTURE_TRACKER=parallel`.
  Invalid tracker config now raises instead of silently falling back.
- `scripts/release/` — `build_bundle.py` (stages the full runtime closure and
  verifies by clean extraction), `compliance_final.py` (`--strict` gate),
  `sanitize_export.py`, `generate_results.py`.
- `app/` — offline tap-to-capture HUD; `android_final/` Kotlin source.
- `evaluation/device/` — benchmarks, robustness, parallel tracker, tests.
- `models/README.md`, `LICENSE`, `NOTICE`, `docs/TEAM_WORKFLOW.md`.
- `results/results.csv` bound to the FP16 SHA (honest FAILs retained).

## What is NOT published (deliberate)

All model weights: `landmark_classifier_fp16.tflite`, `landmark_classifier_int8.tflite`,
`hand_landmarker.task`, and the S3D weights carry **no confirmed redistribution
permission** (see NOTICE). `.gitignore` blocks them. `models/README.md` documents the
contract and the exact SHA-256 to verify after supplying them locally.

## Stable interfaces — please confirm unchanged

| | |
| --- | --- |
| Input | `[1, 24, 255]` float32 |
| Output | `[1, 100]` float32 logits (softmax in `src/main.py`) |
| Vocabulary | 100 ordered glosses, `models/landmark_labels.json` |
| Feature semantics | **Dev-owned** |
| `predict(video_path) -> dict` | **Abhijeet-owned** |

## Model currently pinned

`models/landmark_classifier_fp16.tflite`, sha256
`a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`, 832,888 B.

## Ensemble (72.78%) — NOT integrated

Your tip ships a 4-model ensemble with a `members` list in `landmark_model_meta.json`.
This branch still runs the single classifier (62.61% test). To adopt it I need:

1. Exact SHA-256 for each of `landmark_ens_{rob71,tcn21,tcnm61,tcnm62}_fp16.tflite`.
2. The training/export commit SHA.
3. Confirmation the `members` list is final and ordered.
4. Confirmation `src/features.py` at your tip is the training version
   (yours adds `use_z`; `FEATURE_DIM` is still 255, so the default contract holds).
5. Total deployment size and measured latency.
6. Redistribution authorization, or confirmation it stays local-only.

**Do not report 62.61% and 72.78% as comparable** — different model, different run.

## INT8

`landmark_classifier_int8.tflite` (`9d24640a…`, 509,720 B) is measured but not selected:
73.41% vs 78.61% FP16 (−5.20 pp). Needs calibration provenance before it can replace FP16.

## Please review

- Feature/label parity against the pinned FP16 SHA.
- That `src/features.py` / `src/landmark_frontend.py` match the training versions.
- Files I did **not** modify semantically: `src/features.py`, `src/landmark_frontend.py`,
  `models/*`, `scripts/ml/`.

## Do not overwrite

`src/main.py`, `app/`, `scripts/release/`, `evaluation/device/`, `results/` are mine.
I have not touched your training pipeline, artifacts, or branch.