# Organizer email — one message, five questions

Send **once**, from the address used for team registration, to
`prism@samsung.com` (deck p.27). Do not send these separately; one thread keeps
the answer traceable.

Nothing here blocks the build — send it today regardless of where the other
submission work stands.

---

## Subject

```
Track 2 (EchoGesture) — 5 clarifications before submission: FP16 vs INT8, vocabulary, manifest, filename, deadline timezone
```

## Body

> Hello,
>
> We are a registered Track 2 team (EchoGesture: on-device sign recognition on an
> edge device). We have five questions about the 28-page consolidated deck
> ("XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf"). We would rather
> ask than guess, because at least one point affects a 20-point scoring
> criterion.
>
> **1. FP16 versus INT8.** The deck is not consistent on this.
>
> - p.9: `Budgets: ≤ 20 MB quantized model (INT8), ≤ 50 ms latency`
> - p.10: `Quantized model + conversion script (< 20 MB INT8)`
> - p.11 (the 100-point jury matrix): `Model Quantization (20%): Compilation to
>   TFLite/ONNX <20MB.` — INT8 is not mentioned.
> - p.11 results template: `model_size … <=25 MB`
>
> Our shipped artifact is a **float16** TFLite ensemble, 11.36 MB including the
> hand detector — comfortably under both the 20 MB and 25 MB readings. Since the
> scored criterion on p.11 names only TFLite/ONNX under 20 MB, does an FP16
> artifact under 20 MB satisfy the model-quantization criterion, or are INT8
> weights mandatory? A one-line answer is enough, and it determines whether we
> submit our current model or need a different conversion.
>
> **2. The fixed vocabulary.** p.11 says `prediction is a word from the fixed
> vocabulary list, uppercase`, but we could not find the list itself in the deck
> or in any starter material. Could you share the official vocabulary file for
> Track 2? If no fixed list exists, please confirm we may define and publish our
> own 100-gloss list along with the submission.
>
> **3. The test manifest.** p.11 says `clip_id must match the test manifest
> exactly` and that the evaluator `checks that every clip ID is present`. We have
> not received a manifest. Could you send it, or confirm that teams generate
> predictions over their own held-out clips? We will not invent clip IDs.
>
> **4. Predictions filename.** p.10 and p.11 refer to `predictions.jsonl` (one
> JSON object per line), while the package diagram on p.12 shows
> `results/predictions.json`. Which filename does the automated evaluator read?
> We currently emit both, with identical JSON-Lines content, but would like to
> match the expected layout exactly.
>
> **5. Deadline timezone.** p.25 states the final submission is due 18 November
> 2026, 11:59 PM. The deck does not state a timezone anywhere in its 28 pages.
> Should we assume IST (UTC+5:30)?
>
> One further question, if it is within your scope: our model weights are derived
> from ASL Citizen and WLASL, whose terms are academic and non-commercial. Does
> delivering model artifacts to the evaluators for judging constitute an accepted
> use under the competition rules, or should our repository remain source-only
> with weights supplied separately?
>
> Thank you — we appreciate the time.
>
> [NAME], Team [CollegeName_TeamName], Track 2

---

## What each answer changes

| Question | If the answer is… | Then we… |
| --- | --- | --- |
| INT8 | FP16 accepted | nothing changes; submit as-is |
| INT8 | INT8 mandatory | Dev must approve a conversion; nothing ships until then |
| vocabulary | official list supplied | regenerate labels, recheck the six no-hand clips |
| manifest | supplied | build the official bundle and generate real predictions |
| filename | `.jsonl` or `.json` | drop the other from the package |
| timezone | not IST | move the internal deadline accordingly |
| rights | delivery accepted | can hand weights to evaluators |
| rights | not accepted | source-only submission, stated plainly |

## What not to do

- Do not claim INT8 compliance while waiting for the answer.
- Do not invent a manifest or vocabulary.
- Do not send a second email chasing this before the deadline approaches.