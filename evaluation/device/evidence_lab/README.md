# Evidence hardening (A12) — reproduction and candidate matrix

Status date: 8 November 2026. Owner: Agent A12 (Evidence Engineer).

This document is the reproduction/evidence companion to the hardened
`scripts/release/generate_results.py` and new `evaluation/device/evidence_lab/`
package. It fabricates nothing: every number below is either computed from
files on disk in this copy or marked NOT_MEASURED.

## 1. What changed and why

| Problem in prior generate_results.py | Hardening |
| --- | --- |
| Rows traced only to model SHA + manifest SHA | Every row now carries candidate SHA, feature-config SHA, code commit, dataset/split, clip count, device/runtime, recorded_at and method (see `evidence_lab/provenance.py`). Missing fields are listed explicitly, never silently dropped. |
| `confidence` semantics implicit | New `confidence_semantics` row: model output is **logits** (verified empirically, below); confidence = top-1 softmax over dequantized logits, uncalibrated. Reproduced exactly by `evidence_lab/confidence.py::softmax_confidence`, mirroring `src/main.py`. |
| `model_size` = classifier bytes only | Total artifact accounting via `evidence_lab/sizes.py`: classifier + detector + labels/metadata + feature code. Incomplete inventory ⇒ NOT_MEASURED, never PASS. |
| No per-signer view | `evidence_lab/per_signer.py` derives per-signer accuracy **only** when the benchmark supplies `per_signer` counts (the ASL Citizen research JSON does) or predictions with signer IDs. Otherwise emits `NOT_SIGNER_DATA_SUPPLIED` / NOT_MEASURED. |
| Stale robustness/power/thermal accepted by SHA only | All evidence kinds go through `require_bound_to()`: missing or mismatched candidate SHA raises and aborts generation (exit 2). |
| Smoke-run PASS risk | PASS for accuracy/latency requires the representative flags (≥20 real clips, complete ground truth, no failures, documented split). A 1–3 clip smoke run yields NOT_MEASURED even at 100%. |

CSV schema is unchanged (`metric,target,measured,unit,device,runtime,pass,notes`)
so `build_bundle.py` verification keeps working.

## 2. Confidence semantics, verified on the real models

Run (venv Python 3.11, ai-edge-litert):

```bash
PYTHONPATH=$TMP/echogesture-.../A12 $REPO/.venv/bin/python -c "
from ai_edge_litert.interpreter import Interpreter
import numpy as np
for name in ['landmark_classifier_fp16.tflite','landmark_classifier_int8.tflite']:
    it=Interpreter(model_path=f'$TMP/echogesture-.../A12/models/{name}')
    it.allocate_tensors()
    x=np.zeros((1,24,255),dtype=np.float32)
    it.set_tensor(it.get_input_details()[0]['index'],x); it.invoke()
    y=it.get_tensor(it.get_output_details()[0]['index'])[0].astype(np.float32)
    print(name,'min',y.min(),'max',y.max(),'sum',y.sum())
"
```

Result:

- `landmark_classifier_fp16.tflite`: min −3.02, max 4.83, sum 10.22 → **logits**
- `landmark_classifier_int8.tflite`: min −2.53, max 3.67, sum 12.76 → **logits**

Neither lies in [0,1] nor sums to 1, so `src/main.py`'s softmax is the only
valid confidence definition. The S3D INT8 models additionally dequantize via
`(q − zp) × scale` before softmax. No temperature scaling / ECE calibration
has been measured: confidence is an uncalibrated score, not P(correct).

Tensor contract (same run):

| Model | Input | Output | Quantization |
| --- | --- | --- | --- |
| landmark FP16 | [1,24,255] float32, scale 0 | [1,100] float32, scale 0 | float16 weights, float32 IO — **not INT8** |
| landmark INT8 | [1,24,255] float32, scale 0 | [1,100] float32, scale 0 | float32 IO; not selected as default |
| S3D QAT INT8 | [1,24,64,64,1] int8, scale 0.00392 | [1,100] int8, scale 0.00391 | dequantize → logits → softmax |
| S3D PTQ INT8 | same as QAT | same as QAT | same |

## 3. Total artifact accounting (measured from disk)

```bash
PYTHONPATH=$TMP/echogesture-.../A12 $REPO/.venv/bin/python -c "
import hashlib,pathlib
base=pathlib.Path('$TMP/echogesture-.../A12/models')
total=0
for p in sorted(base.iterdir()):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        print(p.name, p.stat().st_size, h)
        total+=p.stat().st_size
print('ALL models/ files total:', total)
"
```

| Artifact | Bytes | SHA-256 (prefix) |
| --- | --- | --- |
| landmark_classifier_fp16.tflite (default candidate) | 832,888 | a012f2b5… |
| hand_landmarker.task (detector, required at runtime) | 7,819,105 | fbc2a300… |
| landmark_labels.json | 1,492 | e78f5028… |
| landmark_model_meta.json | 123 | 7f189a16… |
| **New-candidate shipped total** | **8,653,608 (≈8.65 MB)** | — |
| s3d_conv1d_qat_int8.tflite (old S3D baseline) | 901,200 | 107756f3… |
| ptq_100_int8.tflite | 904,256 | 067c743c… |
| label100.json (S3D vocabulary) | 1,279 | 741dd6a6… |
| **Old-S3D shipped total** | **1,806,735 (≈1.81 MB)** | — |

Both totals are under the 20 MB cap, but only the *total* figure is the
submittable one; the old classifier-only 0.83 MB figure understated the new
candidate by ~7.8 MB. `generate_results.py --artifact detector=...` must be
passed the `.task`, labels, meta and `src/features.py`/`src/landmark_frontend.py`
for the final run, or the row is honestly NOT_MEASURED.

## 4. Honest current candidate matrix (existing real evidence only)

### 4a. Old S3D candidate (pixel, INT8) — research evidence, NOT release evidence

Evidence: `evaluation/asl_citizen_qat_results.json`,
`evaluation/asl_citizen_ptq_results.json`,
`evaluation/asl_citizen_predictions.csv` (1,705 clips, 11 signers P6/P9/P15/
P17/P18/P22/P35/P42/P47/P48/P49, ASL Citizen test split, signer-disjoint
research eval via `evaluation/eval_asl_citizen.py`). The predictions CSV
recomputes exactly to the PTQ JSON (25/1705), confirming provenance of that
file; the QAT JSON (17/1705) has no surviving per-clip file in this copy.

| Metric | QAT (107756f3…) | PTQ (067c743c…) | Status |
| --- | --- | --- | --- |
| Top-1 (n=1705) | 1.00% (17/1705) | 1.47% (25/1705) | FAIL vs ≥70% (representative research run) |
| Per-signer top-1 | 0.00–1.85% (P35 best, P42 0/160) | 0.63–2.47% (P35 best) | FAIL; signer breakdown supplied, see §5 |
| Latency p95 | NOT_MEASURED (no run_manifest benchmark.json in copy) | NOT_MEASURED | — |
| Robustness / power / thermal | NOT_MEASURED (no robustness/power/thermal JSON in copy) | NOT_MEASURED | — |
| Total size | 1.81 MB (incl. label100.json) | 1.81 MB | PASS (<20 MB) |

This candidate is **rejected on accuracy** and is kept only as a reference.
The ASL Citizen eval also changes dataset (WLASL→ASL Citizen) and vocabulary
coverage vs. any organizer set, so it cannot isolate signer effects — per
`docs/hackathon/RESEARCH.md`.

### 4b. New landmark candidate (FP16, default `ECHOGESTURE_MODEL=landmark`)

Evidence: `evaluation/device/CANDIDATE_INTEGRATION.md` single real clip
`0009701491968876486-PULL CONVINCE.mp4` (SHA-256 7976e5a1…), predicted
PULLCONVINCE @ 0.7507; stage profile 642.01 ms clip-to-JSON (612.15 ms
landmark tracking); synthetic smoke (QAT-era) 7.99 ms cold / 4.12–4.49 ms warm.
One seeded low-light pair (N=1).

| Metric | Value | Status |
| --- | --- | --- |
| Top-1 | 1/1 smoke correct | **NOT_MEASURED** (N=1, nonrepresentative) |
| Latency p95 | 642.01 ms (N=1 profiled); 485.22 ms second run | **NOT_MEASURED** as p95; sample point-in-time **exceeds 50 ms** — FAIL if it were representative |
| Robustness drop | N=1 pair, 0 pt | **NOT_MEASURED** (transform check only) |
| Avg power / thermal | no device measurement | NOT_MEASURED |
| Total size | 8.65 MB (classifier + detector + labels/meta) | PASS (<20 MB); FP16 ≠ INT8 requirement |
| Quantization | FP16, float32 IO | INT8 compliance **not established** (INT8 alt. not selected) |

No `benchmark.json` / `robustness.json` for this candidate exists in this
copy, so `generate_results.py` cannot yet produce a PASS row for it — by
design. The blockers are execution blockers (representative validation
manifest, physical device), not report-formatting ones.

## 5. Per-signer accuracy from existing data (S3D PTQ, n=1705)

Recomputed from `asl_citizen_predictions.csv` (matches PTQ JSON exactly):

| Signer | Correct/Total | Top-1 |
| --- | --- | --- |
| P35 | 4/162 | 2.47% |
| P22 | 3/148 | 2.03% |
| P47 | 3/147 | 2.04% |
| P15 | 3/161 | 1.86% |
| P9 | 3/167 | 1.80% |
| P17 | 2/145 | 1.38% |
| P49 | 2/147 | 1.36% |
| P48 | 2/157 | 1.27% |
| P18 | 1/153 | 0.65% |
| P42 | 1/160 | 0.63% |
| P6 | 1/158 | 0.63% |

(QAT per-signer from its JSON: P35 3/162 best, P42 0/160 worst.) New-candidate
per-signer accuracy: NOT_MEASURED — no signer-labelled run of the landmark
model exists in this copy.

## 6. Reproduction commands

```bash
# Tests (39 pre-existing + 15 new = 54; all must pass)
PYTHONPATH=$TMP/echogesture-.../A12 \
  $REPO/.venv/bin/python -m unittest \
  discover -s evaluation/device -p 'test_*.py' -v

# Generate a results CSV from a real benchmark.json (example; needs a real run)
python scripts/release/generate_results.py \
  --benchmark /path/to/benchmark.json \
  --model-sha a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4 \
  --artifact detector=models/hand_landmarker.task \
  --artifact labels=models/landmark_labels.json \
  --artifact meta=models/landmark_model_meta.json \
  --artifact features=src/features.py \
  --output /tmp/results.csv
```

## 7. Integration readiness and blockers

Ready to integrate: `scripts/release/generate_results.py` (backward-compatible
CLI + CSV schema), `evaluation/device/evidence_lab/*.py` (new package, no
existing-file changes), `evaluation/device/test_release.py` (existing tests
preserved, new hardening tests appended). Read-only inputs were not modified:
`run_manifest.py`, `robustness.py`, benchmark JSON, `src/main.py`,
`src/features.py`.

Blockers before any PASS can be claimed (unchanged from baseline, now sharper):

1. No representative labelled validation run of the landmark candidate
   (≥20 clips, complete ground truth, documented split, profiled, no
   failures) — accuracy/latency/robustness remain NOT_MEASURED.
2. N=1 latency evidence (642 ms / 485 ms) already exceeds the 50 ms target;
   MediaPipe over 32 frames is the documented bottleneck.
3. No physical Android device: power and thermal remain NOT_MEASURED;
   laptop numbers must not be presented as target-device evidence.
4. INT8 compliance unresolved (default FP16; organizer requirement unconfirmed).
5. Organizer vocabulary/manifest/timing-boundary/detector-size-accounting
   still pending; per-signers and totals above are research-scope only.
