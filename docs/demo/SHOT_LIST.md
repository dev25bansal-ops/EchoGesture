# Shot list — offline clip-path recording

**Camera and Android are blocked; do not shoot those shots.** Record the
clip-path fallback below. Every frame must be a real run of the actual
candidate. If a shot cannot be produced honestly, cut it and note the gap in
the dashboard instead of faking it.

| Shot | Required content | Evidence to show |
| --- | --- | --- |
| External signer | Third-person framing, supported isolated sign from the 100-sign vocabulary | Consent; signer held out of training |
| Receipt/dependency check | Manifest with `clip_id,path` columns; Python 3.11 environment | Real input clip path; no network |
| Prediction | Real JSON output: prediction, top-5, confidence, latency | Model file name + FP16 format; SHA from `../release/FINAL_EVIDENCE.md` |
| Retry | No-hand or low-confidence clip raising the explicit error / provisional retry state | Visible retry state; threshold labelled **provisional** |
| Profile | Stage timings: decode 27.34 / tracking 612.15 / features 0.24 / classifier 0.17 ms | **N=1** label on screen; no p95 claim |
| Architecture | Local pipeline diagram | Landmarks → shared features → classifier path; shipped bytes 8,651,993 |
| Metrics | Dashboard table | 39 tests pass; accuracy/p95/robustness/power/thermal marked NOT_MEASURED with N |
| Limitations | Scope and unmet requirements | FP16 (not INT8), 50 ms not met, camera/Android unverified, vocabulary pending organizer input |

Do not film until the release candidate is frozen and SHA-verified. Do not
edit a correct caption over an unrelated clip.
