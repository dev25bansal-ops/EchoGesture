# EchoGesture Track 2 sprint

**Team** Abhijeet and Dev

**Dates** 7–18 November 2026, IST

**Deliverable** A fully local ASL isolated-sign recognizer with a judge-runnable
Python entry point and a phone demo that displays live captions.

[Official brief, PDF pages 9–12 and 24–27](../../XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf) ·
[Current baseline measurements](../../RESULTS.md) ·
[Registration form](https://forms.gle/oGBXsydqbhE3HS2o9)

## Start here

| Owner | First action | End-of-sprint deliverable |
| --- | --- | --- |
| [Dev](DEV.md) | Create the data manifest and run a small both-hand extraction and training pilot. | A reproducible INT8 candidate, label map, feature contract and unseen-signer results. |
| [Abhijeet](ABHIJEET.md) | Confirm registration and starter kit access, reserve an Android phone, and run the current evaluator entry point. | A working evaluator bundle, measured device results, offline caption demo, deck and submission. |
| Both | Agree on the [handoff contract](HANDOFF.md) and pin the sprint base. | Select the final model together and have Dev independently reproduce the release. |

Registration closes **8 Nov, 11:59 PM IST**. Final submission closes
**18 Nov, 11:59 PM IST**. Use **17 Nov** as the internal release deadline.

## Schedule

| IST date | Dev | Abhijeet | Checkpoint |
| --- | --- | --- | --- |
| **7–8 Nov** | Audit signer splits and class coverage. Extract a small provisional sample. | Check registration, request missing evaluator details, reserve phone, smoke-test the baseline on Python 3.11. | Shared vocabulary status, sample manifest and pinned baseline SHA. |
| **9–10 Nov** | Train a temporal CNN on both-hand features. Export the first INT8 candidate. | Build the manifest runner and camera preview with clearly marked mock captions. | Real validation score, model bundle and repeatable baseline harness. |
| **11–12 Nov** | Hand over 16- and 24-frame results. Send a smaller candidate if tracking is too slow. | Run the candidate through the real clip API and measure the full path on the intended runtime. Connect the live app to real predictions. | Candidate choice based on unseen-signer accuracy and p95. |
| **13–14 Nov** | Address the biggest validation error and test robustness changes. | Test low light, occlusion and airplane-mode captions. Rehearse with a consenting signer held out of training. | One release candidate, plus a clear fallback if continuous sign boundaries are unreliable. |
| **15–16 Nov** | Freeze weights, vocabulary and feature config. Run the untouched held-out evaluation once. | Run final clip timings, robustness and sustained device checks. Generate outputs, demo and deck. | Complete release bundle with measured results tied to the same model. |
| **17 Nov** | Reproduce from Abhijeet's ZIP in a fresh directory and review claims. | Validate all manifest IDs and access links, then submit the checked bundle. | Confirm submission receipt and save the exact release SHA. |
| **18 Nov** | Fix any reproducible release blocker. | Use the remaining time as a submission buffer. | Submission confirmed by 11:59 PM IST. |

## Decision gates

| Metric | Track target | How we use it |
| --- | --- | --- |
| Held-out top-1, 35% of rubric | At least 70% | Select on signer-disjoint validation. Evaluate the frozen candidate on the held-out split. |
| Edge latency, 20% | At most 50 ms p95 in the results table | Time the complete clip-to-JSON path. Keep live camera-to-caption timings separate. |
| Quantization, 20% | Under 20 MB INT8 in the judging rubric | Track bytes and quantization of all shipped model files separately while the organizer confirms how detector weights are counted. |
| Robustness and HUD, 25% | At most 5 percentage points of drop in the results table, plus HUD clarity | Evaluate matched clean and degraded clips. Show legible offline captions and a clear retry state. |
| Power and thermal | At most 500 mW and at most 20% drift over 10 minutes | Record workload, device and method. Mark unavailable measurements explicitly. |

The default QAT baseline recorded **1.00%** on 1,705 unseen-signer ASL
Citizen clips. The PTQ variant recorded **1.47%**. The repository reports
**117.8 ms p95** end-to-end on the dev laptop. These are comparisons for the
sprint, separate from Samsung's evaluation. Use the official starter kit or
final form to resolve the brief's differences on model size, predictions
filename and demo duration. A **2–3 minute** demo follows the track and detailed
submission slides.

## Working rhythm

Agree on one daily handoff time. Each owner records the task ID, branch and SHA,
what ran, the actual result and the next blocker. Use the
[handoff template](HANDOFF.md#daily-update-template). Keep work on owner
branches and bring reviewed changes into the shared sprint branch in small
increments. Confirm both accounts can push before relying on the shared branch.
