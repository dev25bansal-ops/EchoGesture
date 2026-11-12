# Executed stage profile — ensemble runtime

`predict()` with `profile=True`, warm process, 5 runs averaged, single permitted
clip (`0283886564670357-WHAT FOR.mp4`), macOS arm64, ai-edge-litert 2.3.0,
singleton IMAGE tracker.

| Stage | Mean ms | Share |
| --- | --: | --: |
| Landmark tracking (MediaPipe) | **217.26** | **90%** |
| Video open + decode | 21.78 | 9% |
| Output post-processing | 0.44 | <1% |
| Ensemble inference (4 members) | 0.41 | <1% |
| Feature preparation | 0.19 | <1% |
| Model initialization | 0.08 | <1% |
| **Total** | **~240** | |

## What this establishes

The 50 ms full-clip target cannot be met by changing the classifier: ensemble
inference is 0.41 ms, i.e. 0.2% of the clip cost. **Hand landmark tracking over
32 sampled frames is 90% of the runtime.** Any credible path below 50 ms must
reduce per-frame detector cost or the number of tracked frames — both of which
change features and therefore invalidate the trained model.

Parallel tracking (4 workers) reduces this wall-clock by running frames
concurrently (measured p50 79.3 / p95 99.8 ms), which is the practical best
achieved on this hardware.

## Rejected this session

GPU delegate was tested and **rejected**: slower and not bit-identical to the
CPU/XNNPACK path, so it cannot be used without changing predictions.
