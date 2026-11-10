# PR: integrate EchoGesture inference, evaluation, Android source and release tooling

**Base:** `hackathon/two-person-sprint`
**Head:** `hackathon/abhijeet-integration`
**Status:** draft — submission is not ready (no target device, no organizer kit, model
weights not redistributable).

## Summary

Integrates the two-hand landmark pipeline as the default `predict()` runtime, adds a
verified release bundle and evaluation harness, and publishes the Android and offline
demo sources. Source is public; trained weights are not (see **Security and licensing**).

## Abhijeet-owned contributions

- `src/main.py` — `predict(video_path) -> dict`, FP16 classifier + softmax/top-5,
  optional `ECHOGESTURE_TRACKER=parallel`, strict env validation.
- `scripts/release/` — `build_bundle.py` (full runtime closure + clean-extract verify),
  `compliance_final.py` (`--strict` gate), `sanitize_export.py`, `generate_results.py`.
- `app/` — offline tap-to-capture HUD, `android_final/` Kotlin source.
- `evaluation/device/` — benchmarks, robustness, parallel tracker, test suites.
- `results/`, `docs/`, `models/README.md`, `LICENSE`, `NOTICE`, `.gitignore`.

## Dev-owned dependencies

`src/features.py`, `src/landmark_frontend.py`, `models/*` (weights), `scripts/ml/`.
Feature semantics, label order and model artifacts remain Dev-owned.

## Executed tests (this session)

- Top-level: `unittest discover -s evaluation/device` → **281 tests, OK (skipped=1)**
- 22 per-directory suites → **350 tests, 0 failures, 2 skips**
- Public API: default and `ECHOGESTURE_TRACKER=parallel` both return `WHATFOR1` (0.8415)
- Release ZIP clean-extracts and predicts `WHATFOR1`

## Measured performance (this machine, one clip)

| Path | End-to-end | Notes |
| --- | --- | --- |
| Default (singleton IMAGE) | ~470–680 ms | inherent per-clip cost |
| `ECHOGESTURE_TRACKER=parallel` (opt-in) | ~366 ms | **experimental**, default off |

Not a p95 claim. The 30-clip p95 is 327.6 ms and **fails** the <50 ms target.

## Model qualification

- FP16 validation: **272/346 = 78.61%** top-1 (5 signers) — checkpoint selected on val.
- INT8 validation: **254/346 = 73.41%** (−5.20 pp, McNemar p=0.0079) — not selected.
- Held-out test: **782/1249 = 62.61%** top-1, 11 unseen signers — **below the 70% target**.

## Security and licensing

- No secrets in the diff. Absolute host paths replaced with placeholders.
- Participant IDs pseudonymised to `signer_<hash>` in committed evidence.
- **Weights are NOT redistributable** (ASL Citizen / WLASL). `.tflite`/`.task` are
  gitignored; `models/README.md` documents the contract and local setup. `NOTICE`
  records per-artifact provenance.

## Known limitations

- No target device → power/thermal `NOT_MEASURED`.
- Organizer manifest/vocabulary absent → official bundle path blocked.
- Android sources present but **no Gradle/SDK/device** here → no APK or device result.
- Robustness: 4 of 5 corruption conditions exceed the 5 pt drop budget.

## Reproduction

```bash
.venv/bin/python -I -m unittest discover -s evaluation/device -p 'test_*.py'
.venv/bin/python -I src/main.py "<clip>.mp4"
.venv/bin/python -I scripts/release/build_bundle.py verify \
  evaluation/device/release_final/echogesture_release.zip --clip "<clip>.mp4"
```

## Review requested

Dev — confirm the feature contract `[1,24,255] -> [1,100]`, the 100 ordered labels,
and that `src/features.py` / `src/landmark_frontend.py` match the pinned artifact.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)