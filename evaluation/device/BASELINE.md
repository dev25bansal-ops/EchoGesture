# ABH-01/02 baseline handoff — 8 November 2026

Owner: Abhijeet. Branch: `hackathon/abhijeet-integration`.

## Reproduction

From the repository root, with the ignored `.venv/` Python 3.11 environment:

```bash
.venv/bin/python -m pip install -r src/requirements.txt
.venv/bin/python -m unittest discover -s evaluation/device -p 'test_*.py' -v
.venv/bin/python evaluation/device/run_manifest.py /absolute/path/to/manifest.csv --output-dir /absolute/path/to/output
```

The manifest has `clip_id,path` and optional `ground_truth` and `signer_id`.
Supply labels for all clips or none. The runner uses the real predictor,
preserves exact Unicode IDs and aborts on any failed or invalid clip in strict
mode. Profiling mode reports failures explicitly. See `src/README.md`.

## Actual local smoke measurement

A synthetic, uniformly coloured 24-frame MP4 was created outside Git with
OpenCV solely to exercise the runtime and interface. It contains no sign and
must not be used for accuracy or real-video latency claims.

- Python 3.11.18, macOS 26.3 arm64, LiteRT XNNPACK CPU.
- QAT model SHA-256:
  `107756f3a5ec118dec7855ed5ada34dcf85dccaedbef7d80a953d8cc1294cc0a`.
- Current `src/main.py` SHA-256:
  `e872f1f04391af5e0286b7599987e87796d89edf2e9fbd485e95b5fe67f126d0`.
  This is uncommitted work, not a release commit.
- Three calls to the same file, with different manifest IDs: cold call
  **7.99 ms**; two warm calls **4.12 ms and 4.49 ms**. Warm p95 = **4.49 ms**.
  Sample size is too small for a performance claim; these are harness checks.
- Each output predicted `CANDY` with confidence 0.0113 and returned the
  requested exact clip ID. No ground truth or accuracy was reported.
- The independent cold CLI smoke run returned the same prediction in 44.28 ms.
  The original **19** evaluator tests passed then; the expanded integration
  suite now contains **39** passing tests.

The real CLI `python src/main.py <clip>` also ran successfully after the
optional manifest-ID integration change. `predict(video_path)` remains the
one-argument evaluator contract; `clip_id` is an optional keyword override for
manifest IDs that differ from the filename.

## Scope and next dependencies

The pixel model remains the baseline. Dev's landmark candidate has since been
integrated and verified separately in `CANDIDATE_INTEGRATION.md`; the original
`src/main.py` SHA above no longer describes that newer file. No phone
measurement, organizer manifest or official release prediction file is claimed.

Dev: hand off the model SHA, ordered labels, feature config, frame sampling,
input/output quantization and a permitted smoke fixture when ready. The runner
accepts the corresponding predictor, label map and model paths together.

Organizer inputs still needed: official vocabulary and manifest, allowed
training sources, evaluation device/runtime, what the 50 ms boundary includes,
and whether detector weights count toward the size cap. Registration and phone
reservation require human confirmation; neither is marked complete here.
