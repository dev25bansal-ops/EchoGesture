# Organizer clarification request — Track 02

Status: **DRAFT — NOT SENT.** Sending requires your explicit authorization, and
should come from the registered team email so it can be matched to the entry.

Every question below is one the official deck does not answer. Each is phrased
so a one-line reply is enough.

---

**Subject:** Track 02 (Sign Language on Edge) — clarification before final
submission

Dear Samsung XR Hackathon organizers,

We are preparing our Track 02 submission and want to make sure we deliver
exactly what the evaluator expects. Six points in the supplied deck
(`XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1`) are ambiguous or have
no supplied input:

**1. Fixed vocabulary.**
Slide 11 states `prediction` must be "a word from the fixed vocabulary list,
uppercase", and that `clip_id` must "match the test manifest exactly". Track 04
slides 21–22 reference a student kit containing `manifest.csv` and
`supported_classes.json`, but Track 02 slides 8–12 contain no equivalent and no
vocabulary list was supplied. Could you share the Track 02 vocabulary file and
test manifest, or confirm that teams may define their own vocabulary?

**2. Quantization requirement.**
Slide 9 specifies "≤ 20 MB quantized model (INT8)". Slide 11's scoring line
reads "Compilation to TFLite/ONNX <20MB" without naming INT8, and slide 11's
results-sheet template gives the size target as `<=25 MB`. Could you confirm:

  (a) is INT8 specifically required, or is any quantized TFLite/ONNX accepted?
  (b) does the size budget include the hand-detection model, or only the
      classification model?

**3. Predictions file name.**
Slide 11 requires a single `predictions.jsonl`, one JSON object per line.
Slide 12's bundle diagram shows `results/predictions.json`. Should we submit
both, or is one preferred?

**4. Demo video length.**
Slide 10 says "2–3 min", slide 24 says "5-min demo video", slide 25 says "max 3
minutes". We plan for ≤3 minutes — please confirm.

**5. Registration and deadline.**
Slide 25 lists registration closing 8 Nov 2026 and final submission 18 Nov 2026.
Could you confirm that (a) 18 Nov 2026 11:59 PM is in the organizer's timezone —
which timezone specifically? and (b) teams that completed registration will
receive the Google Form link directly?

**6. Held-out evaluation data.**
Slide 11/12 say the model is evaluated on a held-out dataset. Will teams receive
the evaluation clips and manifest in advance, or is the held-out set applied
after submission? This determines whether we can report accuracy on the official
distribution at all.

Thank you — a short reply to any of these unblocks our final packaging.

---

## What each answer changes on our side

| Question | If answered one way | If answered the other |
| --- | --- | --- |
| 1 vocabulary/manifest | We re-run `predict()` over the official clips and regenerate `predictions.jsonl`; ~1 hour of compute | We submit our 100-gloss predictions and state the vocabulary gap explicitly |
| 2 INT8 | We must replace the FP16 ensemble, accepting a measured 5.2 pt top-1 loss (73.4% vs 78.6% single-model) or quantizing the ensemble | We keep the FP16 ensemble at 84.4% top-1 and document the size compliance |
| 3 file name | We submit only the named file | We submit both; already generated from one source |
| 4 video | We re-cut the demo | No change |
| 5 timezone | We have days, not hours | Could block delivery entirely — check first |

## Do not claim

Do not tell the organizers that we are INT8-compliant, that we have the official
vocabulary, or that we have met the 50 ms latency target. None of those are true
yet.