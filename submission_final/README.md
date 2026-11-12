# EchoGesture — final submission package

On-device isolated sign recognition for a 100-gloss American Sign Language
vocabulary. Samsung XR Hackathon 2026, Track 2.

Everything here runs locally. There is no network call on the inference path,
and no frame, landmark or caption leaves the device.

---

## What this package contains

| Path | What it is |
| --- | --- |
| `FINAL_CHECKLIST.md` | **Start here.** Every deliverable with its state, owner and next action. |
| `RELEASE_MANIFEST.json` | Machine-readable identity: member hashes, tensor indices, metrics, disclosures. |
| `architecture/ARCHITECTURE.md` | How the pipeline works and where the time goes. |
| `evidence/PROVENANCE.md` | Where every number and artifact came from, and how to reproduce it. |
| `presentation/` | The deck (source lives in `docs/submission/`). |
| `demo/` | Recording guide (source lives in `docs/demo/`). |
| `reproduction/` | Commands to regenerate any result. |
| `organizer/` | Organizer requirements, the INT8 clarification draft, contact. |

## The model in one table

| | |
| --- | --- |
| Architecture | 4 × FP16 TFLite, mean-logit aggregation, single softmax |
| Input | `[1, 24, 255]` float32 — position, velocity, acceleration, inter-hand geometry, presence masks, canonical `[Left, Right]` slots |
| Output | `[1, 100]` logits → softmax → top-5 uppercase glosses |
| Size | 11.37 MB including the 7.82 MB hand detector |
| Accuracy | 72.78% top-1 / 87.51% top-5, 1,249 scored clips, 11 unseen signers |
| Latency | 99.8 ms p95 full clip-to-JSON at best (parallel-4) on a Mac laptop |
| Quantization | **FP16**, not INT8 |

## Running it

```bash
# 1. install
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r src/requirements.txt

# 2. supply the weights (not in this repository — see models/README.md)
#    place hand_landmarker.task and the four landmark_ens_*_fp16.tflite files in models/

# 3. predict on one clip
python3 -c "
from src.main import predict
print(predict('path/to/clip.mp4'))"
```

```bash
# verify the whole suite
python3 -I scripts/run_all_tests.py --quiet
```

The evaluator entry point is `src/main.py`:

```python
def predict(video_path: str) -> dict:
    """Return {"clip_id", "prediction", "top5", "confidence", "latency_ms"}."""
```

`prediction` is uppercase, drawn from `models/landmark_labels.json`. `latency_ms`
is the full clip-to-JSON wall time, self-reported.

## What is not in this package, and why

**Model weights.** They derive from ASL Citizen and WLASL and carry no confirmed
redistribution permission. They are gitignored and documented in `models/README.md`
with their exact expected SHA-256 values. A clone without them starts, validates
the metadata, and then raises `FileNotFoundError` naming the missing member —
it never silently runs with fewer models than declared.

**Organizer-manifest predictions.** No manifest was supplied. The predictions
here are for our own held-out split. Inventing organizer clip IDs would be
fabrication, so it was not done.

## Limitations

Stated plainly, because a submission that hides them is worse than one that meets
fewer targets honestly:

- The 50 ms latency target is **not met**. Hand tracking is 90% of runtime; the
  ensemble itself is 0.36 ms. Meeting 50 ms requires changing the feature
  pipeline, not the model.
- Crop (−10.00 pp) and occlusion (−6.67 pp) exceed the 5 pp robustness budget.
- The test split was inspected during model selection. It is held-out, not blind.
- On-device latency, power and thermal were **not measured** — no Android device
  was available. They are reported as NOT MEASURED, never estimated.
- The model recognises isolated glosses, not continuous sentences.

## Contact and provenance

Public repository: https://github.com/dev25bansal-ops/EchoGesture
Branch: `hackathon/abhijeet-public`
Organizer contact: `prism@samsung.com`

For the full audit trail — every claim tied to a file and a command — read
`evidence/PROVENANCE.md` and `FINAL_CHECKLIST.md`.