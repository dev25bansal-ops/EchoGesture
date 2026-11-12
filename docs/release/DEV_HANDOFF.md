# Dev handoff — Abhijeet → Dev

**Branch:** `hackathon/abhijeet-integration` @ `d2bf115` (5 commits ahead of `origin/main`)
**Status:** prepared locally; **not yet pushed** — no GitHub credential on this machine.

---

## What I changed

- `src/main.py` — `predict()` runtime: reads the model from `landmark_model_meta.json`,
  runs `detect_hands` + `build_features`, softmax over FP16 logits. Optional
  `ECHOGESTURE_TRACKER=parallel`; default path unchanged.
- `scripts/release/build_bundle.py` — now stages `src/features.py`,
  `src/landmark_frontend.py`, `models/hand_landmarker.task`,
  `models/landmark_model_meta.json` (previously omitted → broken extract).
- `app/`, `evaluation/device/`, `results/`, `docs/`, `LICENSE`, `NOTICE`,
  `models/README.md`, `.gitignore`.

## What is experimental (do not treat as settled)

- `ECHOGESTURE_TRACKER=parallel` — opt-in only; ~366 ms vs ~470–680 ms default on one
  clip. The 50 ms clip-p95 target is **not** met.

## Dev-owned files — please review, I did not change their semantics

```
src/features.py
src/landmark_frontend.py
models/*.tflite, models/*.task     (not committed — see licensing)
scripts/ml/
models/landmark_model_meta.json    (committed: interface only)
models/landmark_labels.json        (committed: interface only)
```

## Stable interfaces

| | |
| --- | --- |
| Input | `[1, 24, 255]` float32 |
| Output | `[1, 100]` float32 logits (softmax in `src/main.py`) |
| Labels | 100 ordered glosses, `models/landmark_labels.json` (sha `e78f5028…`) |
| Feature semantics | **Dev-owned** — presence masks, root/scale norm, hand slots |
| `predict(video_path) -> dict` | **Abhijeet-owned** public contract |

## Model artifact currently pinned

`models/landmark_classifier_fp16.tflite`, sha256
`a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`,
832,888 B, FP16 weights / FP32 IO. **Not in the repo** (no redistribution permission).

## Open item that needs your call

Your branch tip (`origin/hackathon/dev-model` @ `ace8a15`) has moved past this:
it ships a **4-model FP16 ensemble** (`landmark_ens_{rob71,tcn21,tcnm61,tcnm62}_fp16.tflite`)
driven by a `members` list in `landmark_model_meta.json`, and reports **72.78% test
top-1 (above target)**.

This branch still runs the **single** classifier (62.61% test). To adopt the ensemble I
need from you:

1. Confirmation the meta `members` list is final and stable.
2. The member SHAs and that `src/features.py` at your tip is the training version
   (your `features.py` differs from mine — additive `use_z`, `FEATURE_DIM` still 255).
3. A decision on whether ensemble logit-averaging is wanted in `src/main.py`
   (I would add it behind the same meta-driven loader).

I have **not** merged or modified your branch.

## Remaining INT8 work

INT8 artifact exists (`9d24640a…`, 509,720 B) but is not selected: 73.41% vs 78.61%
FP16 (−5.20 pp). Needs calibration/provenance before it can replace FP16.

## What to review before merge

- Feature/label parity against the pinned FP16 SHA.
- That `src/features.py` + `src/landmark_frontend.py` match the training versions.
- Nothing here edits your training pipeline or artifacts.