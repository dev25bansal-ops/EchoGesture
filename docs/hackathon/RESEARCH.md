# Research decisions for the Track 2 sprint

The first experiment should test whether two-hand landmarks improve recognition
on unseen signers. Run the clip pipeline on the intended device before choosing
a frame budget. Use the organizer's vocabulary for the final comparison.

| Evidence from the repository | Decision |
| --- | --- |
| The default INT8 QAT model scored 1.00% and the PTQ model scored 1.47% on 1,705 ASL Citizen clips from held-out signers. The benchmark also changes datasets from WLASL to ASL Citizen. | Keep the pixel model as a reference. Compare new candidates on the same data and vocabulary. This benchmark cannot isolate the effect of signer identity from other dataset differences. |
| The 17 readable WLASL clips in the positive control came from training signers. | Use the positive control to check the pipeline. Use signer-disjoint validation to choose the model. |
| The dev laptop reported 117.8 ms p95 for clip-to-prediction. A separate decode experiment using 240p and 12 sampled frames reached 92.1 ms p95. | Profile decode and landmark extraction before optimizing the classifier. Test frame counts and resolution against accuracy and full-path p95 on the intended runtime. |
| The shipped QAT model is 901,200 bytes. The replacement's size, accuracy, robustness, power and thermal behavior are open measurements. | Run the exported artifact through the same validation as the float model. Record actual model bytes and measure the release candidate. |

[Baseline measurements](../../RESULTS.md) ·
[Cross-dataset evaluation](../../S3D_signer_disjoint_evaluation.md) ·
[Architecture notes](../../ARCHITECTURE_R8.md)

1. **Dev** verifies class coverage, split integrity and feature extraction on a
   small pilot. Then train a temporal CNN with both hands and compare 16 and
   24 frames on the same validation set. Record the clip and signer counts,
   per-class errors, and what fraction of clips have missing landmarks.
2. **Dev** exports INT8 and compares float and quantized accuracy using the same
   clips. Hand off the model SHA, label order, feature config and expected
   input/output shapes so Abhijeet can check integration.
3. **Abhijeet** measures the full clip path and records the device, runtime,
   sample count and p95. If extraction or decoding dominates, compare a smaller
   frame budget using a model validated at that same budget. Report live caption
   delay separately.
4. **Both** select the release candidate using validation accuracy, clip p95
   and paired clean-versus-degraded results. Freeze the candidate before the
   held-out run. Record measurements against the exact model shipped in the ZIP.

The model choice and frame counts are experiments. The official targets and
required output format are tracked in the [submission checklist](SUBMISSION.md).
