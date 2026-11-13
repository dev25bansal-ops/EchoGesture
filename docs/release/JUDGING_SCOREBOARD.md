# Judging scoreboard — Track 02

Official weights from deck p.10: **Execution & Efficiency 40%** (model
quantization 20% · edge latency 20%), **Quality & Accuracy 60%** (top-1 35% ·
robustness & HUD 25%).

**No judge scores are invented here.** This maps our measured evidence onto each
criterion so you can see where points are likely and where they are not. The
"likely credit" column is my estimate of the *evidence* available to a judge,
not a prediction of their score.

## Execution & Efficiency — 40%

| Criterion | Weight | Our evidence | Likely credit | Gap |
| --- | --: | --- | --- | --- |
| Model quantization | 20% | FP16 ensemble + FP16 detector, 11.36 MB. INT8 candidate exists with a **measured 5.2 pt loss** and float32 IO; detector cannot be INT8'd without retraining. | **Partial** — size passes every stated budget; the word "INT8" is not satisfied | Needs organizer reply on whether "quantized TFLite <20 MB" suffices |
| Edge latency | 20% | **275.8 ms p95** default; 51.4 ms p50 at 12 workers. Classifier 0.36 ms; tracking is 90%. | **Low** — target is <50 ms and the default configuration is 5.5× over | Real device measurement; parallelism as a documented mode |

**Section estimate: weak.** Size is genuinely good; quantization format and
latency are not.

## Quality & Accuracy — 60%

| Criterion | Weight | Our evidence | Likely credit | Gap |
| --- | --: | --- | --- | --- |
| Top-1 accuracy | 35% | **72.78%** test on 11 unseen signers (target ≥70%). **84.39%** validation on 346 clips. | **Good** — clears the stated 70% bar, traceable, signer-disjoint | Organizer held-out set is unknown to us |
| Robustness | 25% (shared) | Worst condition **10.00 pt** drop (crop); occlusion 6.67; low light, blur, frame drop ≤0 | **Partial** — 2 of 5 corruptions fail the 5 pt target, and the deck says so | Crop/occlusion augmentation would need retraining |
| HUD clarity | 25% (shared) | Full Android app: live preview, frame guide, countdown, result card, top-5, uncertainty wording, offline TTS, history, glossary | **Good** | Unverified on a phone |

**Section estimate: reasonable.** Accuracy and HUD are the strongest parts of
this submission.

## Where the submission actually stands

| Area | State |
| --- | --- |
| Accuracy (35%) | Measured, clears target, honestly caveated |
| HUD (part of 25%) | Built and emulator-verified |
| Robustness (part of 25%) | Measured, **fails** on 2 of 5 conditions, disclosed |
| Size (part of 20%) | **Passes** both stated budgets |
| Quantization format (part of 20%) | **FP16**, disclosed, ambiguous requirement |
| Latency (20%) | **Fails** by 5.5× in default config |

## Honest summary

Roughly **half the rubric is in good shape** — accuracy and HUD. The other half
(quantization + latency, 40% of the total) is weak on measured evidence, and no
amount of re-packaging changes the numbers: 275.8 ms is 275.8 ms and the models
are FP16.

The two levers that would move the score, in order of value per day:

1. **Measure latency on an actual phone.** If a Galaxy-class device tracks
   hands faster than this laptop, the 20% latency criterion could flip from
   "fails" to "close", and that is a measurement, not a claim. This is the
   single highest-value action and it needs hardware.
2. **Get the INT8 clarification.** If "quantized TFLite <20 MB" is accepted,
   the project already passes without touching the model. If INT8 is mandatory,
   the honest options are a 5.2 pt accuracy loss or a bounded
   quantization-aware fine-tune — and the latter needs training data plus time
   we may not have before 18 Nov.

What will **not** help: more UI polish, more slide design, or restating
estimates as measurements. Judges re-measure latency themselves (deck p.11:
"the organizers re-measure on their own device").