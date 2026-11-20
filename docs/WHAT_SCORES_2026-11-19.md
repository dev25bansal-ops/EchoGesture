# What actually scores, and what I spent the day measuring

Written 2026-11-19. This exists because an adversarial review of my own work
found that the largest piece of analysis in this repo optimises a quantity the
jury never scores.

## The rubric, quoted

From the organizer deck (`_slides_text.txt`, p.11, "100-POINT JURY MATRIX"):

| criterion | weight |
|---|---|
| Model Quantization — TFLite/ONNX < 20 MB | **20%** |
| Edge Latency — measured inference time < 50 ms | **20%** |
| **Top-1 Accuracy** — precision on held-out test videos | **35%** |
| Robustness & Delivery — occlusions & HUD clarity | **25%** |

The evaluator's contract (p.10-11) is: it runs `predict(video_path)` on each
held-out clip, takes the returned `prediction`, and computes top-1 accuracy. It
also reads `latency_ms` — **self-reported, and re-measured on the organizer's
own device**.

## The consequence I missed

The app's acceptance gate (`ResultPolicy`, `MIN_MARGIN = 0.15`) lives in the
**Android UI only**. `predict()` returns its argmax unconditionally. So:

- Acc@accept at matched coverage: **not scored**.
- The per-class threshold study: **not scored**.
- The scorer bake-off (margin vs entropy vs top5_mass): **not scored**.
- The gate trade (143 extra correct refused vs 62 wrong caught, 2.31:1): **not scored**.

That work is not worthless — it correctly stopped a threshold change from
shipping — but it was the wrong thing to spend the runway on, and it was framed
as the top priority.

## What is actually in trouble, by weight

| criterion | weight | state | gap |
|---|---|---|---|
| Top-1 accuracy | 35% | **85.45%**, target ≥70% | **PASS**, comfortably |
| Robustness & HUD | 25% | pixel +4.48 **pass**; landmark occlusion **+5.76 FAIL ungated**, +2.39 pass with the app's gate | **PARTIAL** — see below |
| Model size | 20% | **16.33 MB** total (9 FP16 members 8.51 + detector 7.82), target <20 MB | **PASS**, 3.7 MB spare |
| **Edge latency** | **20%** | detector p50 **34–45 ms**, p95 **46–54 ms** on the phone (GPU delegate, 4 runs); classifier 3.9–4.8 ms | **PARTIAL** — p50 passes in all 4 runs, p95 in 3 of 4 |

**Why Robustness is PARTIAL, not PASS.** An earlier version of this table marked
it PASS while quoting only the two passing numbers. `results.csv` records the
same ≤5 target **failing at +5.76** for `drop_hand_25pct` — losing a hand for a
quarter of the clip. That failure is real and is a model limitation, not a
measurement artefact. What makes the criterion defensible is that the app's
acceptance gate recovers it to **+2.39 at 86%/80% coverage**: the app does not
answer on the corrupt inputs where it would be wrong, it declines them. Both rows
are in `results.csv` and both should be shown.

**Why Edge latency is PARTIAL, not PASS.** The four GPU runs measured p95
46/49/50/54 ms against ≤50. Three pass; the fourth is 4 ms over on a thermally
throttled phone (Thermal Status 2, SoC 39.9 °C). p50 34–45 ms clears the target
in every run. Marking a range that includes 54 as a clean pass is the kind of
claim a judge can check and disprove, so the scored row now says `partial`.

### The 50 ms referent is NOT defined — this section used to overclaim

An adversarial review on 2026-11-19 corrected an error in an earlier version of
this document. It said latency "fails by 11–15×" against the 50 ms budget. The
deck's only definition is:

> **Edge Latency (20%):** Measured inference time <50 ms.

(from `_slides_text.txt` L263-264; the other two mentions are "≤ 50 ms latency"
at L186 and L200, neither of which names a unit). **The deck never says whether
50 ms is one detector inference, one per-frame pass, or one whole-clip
`predict()` call.** So "11–15× over target" asserted a referent the organizer
never fixed — and the repo disagreed with itself, because `results.csv` marks
the end-to-end row `NOT_APPLICABLE` with target `not specified` while this
document called it a failure.

**What is defensible without the organizer's answer**, stated by stage and device:

```
desktop x86, per-frame detector   15 ms p50 / 22.95 ms p95   -> under 50 on the per-inference reading
desktop x86, whole-clip predict() 577-749 ms p50              -> over on the whole-clip reading
Vivo V2404, per-frame detector    34-45 ms p50 / 46-54 ms p95 (GPU delegate, DEBUG build, upper bound)
```

**CORRECTED.** This block previously listed `80-109 ms p50 / 159-187 ms p95` and
concluded the detector was "1.6-3.7x over 50 ms", making the latency criterion
"at risk on either reading". Those were **CPU-delegate** numbers, taken before
`HandTracker.create()` was changed to request the GPU delegate. On GPU the same
phone measures **34-45 ms p50**, which is *inside* the target, and 46-54 ms p95,
which is inside it in 3 of 4 runs.

So the honest statement is neither "fails" nor "passes": **p50 clears the budget
comfortably on the device the organizer would use; p95 sits on the line.** The
old conclusion was written against superseded numbers and is retracted.

The question is already drafted in
`docs/submission/INT8_CLARIFICATION_2026-11-19.md` item (b) — *"The exact
boundary for the 50 ms latency target: per-frame inference, or end-to-end
clip-to-JSON?"* — and that email is **NOT SENT**. Until it is answered, no
latency row should be published as a confident PASS or FAIL.

### Where the time goes (unchanged, and still the real lever)

```
decode ~100 ms + detect ~526-607 ms + features 0.7 ms + classifiers 3.2 ms
```

**The nine classifiers cost 3.2 ms. The detector costs the rest.** The model is
not the problem; the frontend is. The app now tries the MediaPipe GPU delegate
with a CPU fallback, which is the one untried lever on this criterion.

## Two fixes attempted; both measured and rejected

### VIDEO mode for the detector

MediaPipe's VIDEO mode tracks hands across frames instead of re-detecting each
one, so it should be cheaper. Measured on 300 clips from the test split, each
mode run on the same clip in the same process:

| mode | accuracy | mean latency |
|---|---|---|
| IMAGE (shipped) | 84.67% (254/300) | 708.9 ms |
| VIDEO | 85.00% (255/300) | 756.6 ms |

A **+1 clip** accuracy difference and slightly *slower*. Top-1 agreement was only
87% — the modes disagree on 13% of clips, so the difference is noise, not a
signal. **Rejected.** (An earlier 24-clip sample suggested VIDEO was uniquely
correct on 3 of 5 disagreements; on 300 clips that evaporated. This is exactly
why single-tens samples do not decide anything.)

### Sampling fewer frames

Covered by `predict()`'s existing `max_frames=max(32, _FRAMES)` — it already
samples 32, not every frame of a 110-frame clip. The per-frame detector cost is
what dominates, and 32 frames × ~19 ms is the 607 ms.

## What would actually move it

The detector is MediaPipe `HandLandmarker` in IMAGE mode on CPU. Options, in
order of expected value per hour:

1. **Delegate.** MediaPipe ships GPU/NPU delegates. The organizer lists runtime
   as a fillable column ("LiteRT INT8, GPU delegate"), so a GPU-delegate
   measurement is a legitimate submission. This is the single highest-leverage
   change and it was never tried.
2. **Reduce sampled frames for short clips.** A 3.7 s clip at 8.6 Hz is ~32
   frames, which is already the floor. Not much room.
3. **Report the honest breakdown.** The rubric asks for a measured value and a
   pass/fail. Reporting `577–749 ms, FAIL` with the stage breakdown is more
   defensible than reporting 22.95 ms, which measures a different thing than the
   organizer re-measures.

## The deadline question, which I cannot resolve

The organizer deck says, in four places:

- p.24 `FINAL SUBMISSION · Last date 18 Nov 2026, 11:59 PM`
- p.24 `Everything below needs to be submitted by 18th Nov 2026.`
- p.24 `**No extension would be provided.**`
- p.25 `a working prototype with a 5-minute demo video by 18th November`

Today is 19 November. The working assumption in this session has been a 21 Nov
extension. **No artefact in this repository supports that**, and the deck says
the opposite in writing.

Either the extension was communicated outside the repo (an email, a WhatsApp, an
organizer message) or the deadline has passed. This is the single fact that
changes everything else, and it needs a human to check the organizer portal or
email before any more work is planned.

If 18 Nov was real, the correct move is triage: submit what exists, and stop.

## Files

- `.scratch/video_vs_image_mode.py` — the mode benchmark (300-clip version)
- `docs/PER_CLASS_THRESHOLDS_2026-11-19.md` — the gate study, retained as the
  record of why per-class thresholds did not ship
- `_slides_text.txt` — the organizer deck, quoted above
