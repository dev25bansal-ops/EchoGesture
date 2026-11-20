# Deck audit, 2026-11-19

The presentation in the shared mirror was written on **16 Nov against a 4-model,
72.78% build**. The shipped system is very different, and the deck was
**understating the project by 12.7 accuracy points** while also claiming two
things that are no longer true.

An updated deck is at `docs/submission/EchoGesture_Submission_2026-11-19.pptx`.
The original is untouched in the mirror at
`.scratch/shared_read/docs/submission/EchoGesture_Submission.pptx`
(backup: `EchoGesture_Submission.pptx.bak`).

## What was wrong

| slide | said | true now |
|---|---|---|
| 3, 5 | 4-model ensemble | **9 models** (4 on x,y; 5 on x,y,z) |
| 6 | top-1 72.78% (909/1249) | **85.45% (1069/1251)** |
| 6 | top-5 87.51% | **96.64%** |
| 6 | val 75.36% | **89.31%** |
| 5 | 11.37 MB deployed | **15.57 MB** (8.11 models + 7.46 detector) |
| 7 | "the 50 ms target is NOT met" | **detector measured on the phone: 84–112 ms p50** |
| 8 | 30-clip run, occlusion −6.67, crop −10.00 | **1251-clip: pixel +4.48 PASS, gated +2.39 PASS, landmark +5.76 FAIL** |
| 9 | "device install NOT completed" | **runs on a real arm64 phone, zero exceptions** |
| 9 | "4 members embedded" | **9 members, 82,715,402 B, verified inside the APK** |

Slide 10's own line — "every number on these slides is measured" — was the most
damaging thing in the deck, because several numbers were measured, just against a
build that had since been superseded.

## Every replacement number and where it comes from

- `85.45%`, `96.64%`, `89.31%` — `results/results.csv` rows `top1_accuracy`,
  `top5_accuracy`, `top1_accuracy_val`, each naming the 9-model FP16 ensemble on
  the native cache.
- `8.11 MB` models / `7.46 MB` detector / `15.57 MB` total — computed by summing
  `.tflite` and `hand_landmarker.task` entries inside
  `artifacts/EchoGesture-2026-11-19.apk`.
- `82,715,402 B` and 9 members — `zipfile` on the same APK; sha256 `be2b9c8e…`.
- `84–112 ms`, `159–187 ms`, `8.7–11.6 fps`, `76–82% dropped` —
  `docs/ON_DEVICE_LATENCY_2026-11-19.md`, six samples, two installs, via the
  app's own `AnalysisProfiler`.
- `+4.48 / +2.39 / +5.76` — `results/results.csv` rows
  `accuracy_drop_robustness_pixel`, `_gated`, `_robustness`.

## What was NOT verified

The deck was edited programmatically (`python-pptx`), replacing text only.

**Update, later on 2026-11-19: it has now been rendered and seen.** PowerPoint 16.0
is installed on this machine and is scriptable over COM, so all 11 slides were
exported to PNG and inspected. Every slide renders cleanly — no overflow, no
truncated text, no overlapping boxes, correct figures on all of 3–11.

The layout is sparse: every content slide puts five or fewer lines in the top
third and leaves the lower half empty. That is the original deck's design, not
something introduced by the edits, and it is legible — but it is not a
design-forward deck, and no amount of text editing will change that.

Rendered files: `.scratch/deck_render/Slide1.PNG` … `Slide11.PNG`.

Slide 1 confirms the placeholders are prominently visible exactly as intended —
`<TEAM NAME — confirm before submission> · <COLLEGE NAME — confirm>` — so they
cannot survive a submission by accident.

## The static checks that were run before the render

- all 11 slides present, file is a valid zip (58 entries, no corruption);
- no stale string survives (`Four-model`, `4-model`, `72.78`, `87.51`,
  `NOT completed` all return zero hits);
- every body text box is 11.7 × 5.0 in with at most 5 paragraphs, so no new
  overflow; the longest new line is 100 chars, and the original deck already
  carried 90-char lines in the same boxes.

**Open the file in PowerPoint before presenting and check the four edited
slides (5, 7, 8, 9).**

## Still needs a human

Slide 1 carries `<TEAM NAME — confirm before submission>` and
`<COLLEGE NAME — confirm>`. Those are the registered strings the checklist says
are held privately. **Fill them in before submitting.**
