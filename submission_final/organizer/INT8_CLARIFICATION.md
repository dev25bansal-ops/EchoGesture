# INT8 requirement — clarification pack

Prepared 2026-11-11 by Agent A. **The email below has NOT been sent.**

Source: `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf` at the
repository root (1,168,491 B, 28 pages). Text below was extracted independently
from the PDF with `pypdf` 7.0.1, not copied from a repo transcription. Page
numbers are physical PDF pages.

---

## 1. Verbatim organizer text — what the deck actually says

### Page 9 — "About the Problem", Track 02, CHALLENGE ABSTRACT

> `•  Fully on-device, exocentric pipeline: live smartphone camera video → translated text or audio.`
>
> `•  Recognizes conversational sign gestures in real time without draining battery or thermal throttling.`
>
> `•  Budgets: ≤ 20 MB quantized model (INT8), ≤ 50 ms latency, zero cloud streaming.`
>
> `•  Datasets: How2Sign (80+ h ASL) & WLASL (2,000 words); starter baselines provided.`

### Page 10 — "Challenge Blueprint", Track 02, KEY CONSTRAINTS & REQUIRED DELIVERABLES

KEY CONSTRAINTS & ARCHITECTURE:

> `•  Pose-invariant 3D hand features; quantized TFLite / ONNX Runtime model`

REQUIRED DELIVERABLES:

> `•  GitHub repo: predict() video → text, reproducible setup`
>
> `•  predictions.jsonl (one JSON per clip) + results.csv`
>
> `•  Quantized model + conversion script (< 20 MB INT8)`
>
> `•  Demo video (2–3 min) or deployed app URL`

### Page 11 — "Submission Guidelines", Track 02, 100-POINT JURY MATRIX

Under `1. Execution & Efficiency` (40%):

> `•  Model Quantization (20%):`
> `Compilation to TFLite/ONNX`
> `<20MB.`
>
> `•  Edge Latency (20%): Measured`
> `inference time <50ms.`

**This is the whole scored quantization criterion. The string "INT8" does not
appear on page 11.** The only "INT8" on page 11 is in a *format example* for the
`runtime` column of the participant results sheet:

> `Teams fill in measured, device (for example, "Galaxy S25, Snapdragon 8 Elite"), and runtime (for example, "LiteRT INT8, GPU delegate").`

That is an example value for a free-text field, not a requirement.

### Occurrences of "INT8" in all 28 pages

| Page | Context | Strength |
| --- | --- | --- |
| 9 | `Budgets: ≤ 20 MB quantized model (INT8)` | Requirement (abstract) |
| 10 | `Quantized model + conversion script (< 20 MB INT8)` | Requirement (deliverables) |
| 11 | `runtime (for example, "LiteRT INT8, GPU delegate")` | Example value only |

No other page mentions INT8. Pages 13–23 are Tracks 3 and 4; pages 24–28 are
general rules and do not mention INT8.

---

## 2. What is genuinely contradictory and what is not

| Strength | Page | Wording | Reading |
| --- | --- | --- | --- |
| Strongest | 9 | `Budgets: ≤ 20 MB quantized model (INT8)` | INT8 named in the budget |
| Strongest | 10 | `Quantized model + conversion script (< 20 MB INT8)` | INT8 named in the deliverable name |
| Weaker | 11 | `Compilation to TFLite/ONNX <20MB` | Format + size only; INT8 absent |
| Not a requirement | 11 | `"LiteRT INT8, GPU delegate"` | Placeholder example text |

The scored 20% sits on p.11, and p.11 does **not** say INT8. But the p.9 budget
and the p.10 deliverable name do. Two reasonable readings:

- **Reading A (favourable):** "quantized" means *any* reduced-precision exported
  artifact; the scored test is TFLite/ONNX under 20 MB. FP16 passes.
- **Reading B (strict):** the p.9/p.10 parenthetical is normative and INT8
  weights are mandatory. FP16 fails the 20% quantization criterion.

The deck does not resolve this. **Do not claim INT8 compliance.** Our artifact
is FP16 (`models/landmark_model_meta.json` → `"quantization": "float16"`), and
the shipped weights total 3,544,080 B across four members.

---

## 3. Ready-to-send email draft — DO NOT SEND

**To:** prism@samsung.com
**Subject:** Track 2 (EchoGesture) — clarification requested: is INT8 mandatory, or does FP16 TFLite under 20 MB qualify?

> Hello,
>
> We are a registered team in Track 2 (EchoGesture — sign language prediction
> on an edge device) of the Samsung XR Hackathon 2026, and we are preparing our
> final submission for 18 November. We would like one point confirmed in
> writing before we submit, because the consolidated slides give two different
> statements about the model budget.
>
> The deck says the following:
>
> - Page 9 (Challenge Abstract): "Budgets: ≤ 20 MB quantized model (INT8),
>   ≤ 50 ms latency, zero cloud streaming."
> - Page 10 (Required Deliverables): "Quantized model + conversion script
>   (< 20 MB INT8)."
> - Page 11 (100-Point Jury Matrix, Model Quantization 20%): "Compilation to
>   TFLite/ONNX <20MB."
>
> Page 11 is the scored criterion and does not name INT8, while pages 9 and 10
> do.
>
> Our current submission is a quantized (float16) TFLite ensemble of four
> members. The deployed weights are 3,544,080 bytes, and the total including
> the hand-landmark detector is 11.37 MB — under the 20 MB budget either way.
> We are using the CPU/XNNPACK delegate; no cloud path is involved.
>
> Could you please confirm which of the following is correct?
>
> 1. An FP16 TFLite model under 20 MB satisfies the quantization criterion, or
> 2. INT8 weights are mandatory and an FP16 artifact would lose the
>    quantization score.
>
> If INT8 is mandatory, we would also be grateful to know the deadline for
> resubmitting an INT8 artifact, since we would need to re-export and
> re-benchmark our model.
>
> While we have your attention, we would also appreciate confirmation on four
> items the deck leaves open:
>
> a) The Track 2 fixed vocabulary list and the test manifest — are these
>    available, and where can we download them?
> b) The exact timing boundary for the 50 ms latency target: is it per-frame
>    inference or end-to-end clip-to-JSON?
> c) The submission package layout: page 11 says results/predictions.jsonl
>    while the page 12 diagram says results/predictions.json. Which filename
>    should we use?
> d) The deadline timezone: the slides state 18 Nov 2026, 11:59 PM without a
>    timezone. Is that IST?
>
> Thank you for your help. We will submit by the published deadline.
>
> Best regards,
> <NAME>, EchoGesture team — Track 2
> <COLLEGE NAME> · <TEAM NAME>
> <EMAIL / PHONE>

**Send from:** the address used for team registration, so the organizer can match
the request to the registered team. Save the reply under
`docs/submission/organizer/` (create that directory; it does not exist yet —
`submission_final/organizer/` is empty and unused).

---

## 4. If the organizer answers

### Answer: FP16 under 20 MB is acceptable
- Record the reply (PDF/screenshot + plain text) in `docs/submission/organizer/`.
- Cite it in `COMPLIANCE_FINAL.md` on the INT8 row and in the submission note.
- No model change. Keep reporting the artifact as FP16 honestly.
- Keep the conversion script (`scripts/ml/export_tflite_tf.py`) as shipped.

### Answer: INT8 weights are mandatory
This is a real model change and **Dev owns it**; Abhijeet must not ship a
different model without Dev's sign-off (`docs/hackathon/HANDOFF.md` — Dev owns
`models/`, `scripts/ml/`, features and the model interface).

Concrete fallback, in order of preference:

1. **Post-training static INT8 quantize the shipped ensemble** with the
   representative 24×255 feature dataset used for calibration. Four members is
   roughly 0.9 MiB each at FP16, so INT8 should land near ~1.8 MiB total —
   comfortably under 20 MB. Requires: a 3.11 environment with `tensorflow` (not
   in today's `src/requirements.txt`), the existing export script, and a
   per-member accuracy re-check on the same 1,249-clip scored set.
2. **QAT (`TFLITE_QUANT_AWARE`) re-training** only if PTQ loses more than the
   acceptable accuracy margin. This is a training run, not a packaging step.
3. **Fall back to the existing `models/landmark_classifier_int8.tflite`**
   (509,720 B) only if Dev accepts the single-model accuracy regression — it is
   the single model that scored 62.61%, versus 72.78% for the FP16 ensemble. This
   would trade ~10 accuracy points for the 20% quantization score. It is a
   judgement call, and Dev must make it.

**Acceptance gate for any INT8 candidate (all must hold):**
- `quantization` in the meta sidecar reads `int8`, not `float16`.
- Top-1 on the 1,249-clip scored set is re-measured, not assumed.
- Full clip-to-JSON p95 is re-measured.
- Ensemble aggregation parity is preserved (mean logits → one softmax), and
  `app/android_final` parity tests still pass if the Android path is kept.
- `results/results.csv` is regenerated with the new model SHA and a new
  `model_size` row.
- The APK is rebuilt only if Dev and Abhijeet agree it is in scope — the current
  75 MB APK embeds four FP16 members.

**What must NOT happen:** silently swapping in `landmark_classifier_int8.tflite`
and continuing to report 72.78%, or labelling an FP16 artifact as INT8 in
`results.csv` or the deck. Both would be a false measurement claim.

### Answer: no reply before the deadline
Ship FP16, state it explicitly on the deck (slide 10 already does) and in the
submission note, and record that clarification was requested with no reply. A
truthfully labelled FP16 artifact is a defensible submission; a mislabelled one
is not.

---

## 5. Current state (unchanged by this document)

| Fact | Value | Source |
| --- | --- | --- |
| Shipped members | 4 × FP16 TFLite | `models/landmark_model_meta.json` |
| Member bytes | 3,544,080 B | `stat` on `models/landmark_ens_*_fp16.tflite` |
| Detector bytes | 7,819,105 B | `models/hand_landmarker.task` |
| Total accounted | 11,363,185 B (10.84 MiB; 11.36 MB decimal; repo convention 11.37 MB) | `results/results.csv` |
| INT8 artifact present but not shipped | `models/landmark_classifier_int8.tflite`, 509,720 B, single model | `models/` |
| Scored set | 909/1249 = 72.78% top-1, 87.51% top-5, 11 signers | `results/results.csv` |
| Email sent | **No** | this file |