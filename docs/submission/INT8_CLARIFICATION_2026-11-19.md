# INT8 clarification — ready to send

Updated 2026-11-19. Supersedes the 11 Nov draft in
`.scratch/shared_read/docs/submission/INT8_CLARIFICATION.md`, which quotes model
figures that were true then and are not now.

**Status: NOT SENT. Sending is a human action.**

## What changed from the 11 Nov draft, and why

| | old draft | now |
|---|---|---|
| ensemble | "four members" | **nine members** (4 on x,y; 5 on x,y,z) |
| deployed weights | 3,544,080 B | **8,508,832 B** (8.11 MB) |
| total with detector | 11.37 MB | **15.57 MB** (8.11 + 7.46) |
| deadline | "submit for 18 November" | **21 November** (extended) |
| latency framing | not mentioned | measured on device, now quotable |

Sending the old text would have described a model that no longer exists. The
analysis itself — that p.11 is the scored criterion and does not name INT8,
while p.9 and p.10 do — is unchanged and still correct.

---

## Email

**To:** prism@samsung.com
**Subject:** Track 2 (EchoGesture) — clarification: is INT8 mandatory, or does FP16 TFLite under 20 MB qualify?

> Hello,
>
> We are a registered team in Track 2 (EchoGesture — sign language prediction on
> an edge device) of the Samsung XR Hackathon 2026. We would like one point
> confirmed in writing before we submit, because the consolidated slides give two
> different statements about the model budget.
>
> The deck says:
>
> - Page 9 (Challenge Abstract): "Budgets: ≤ 20 MB quantized model (INT8),
>   ≤ 50 ms latency, zero cloud streaming."
> - Page 10 (Required Deliverables): "Quantized model + conversion script
>   (< 20 MB INT8)."
> - Page 11 (100-Point Jury Matrix, Model Quantization 20%): "Compilation to
>   TFLite/ONNX <20MB."
>
> Page 11 is the scored criterion and does not name INT8; pages 9 and 10 do.
>
> Our submission is a quantized (**float16**) TFLite ensemble of **nine**
> members. The deployed model weights total **8,508,832 bytes**, and the total
> including the MediaPipe hand-landmark detector is **15.57 MB** — under the
> 20 MB budget either way. Inference runs on the CPU/XNNPACK delegate; there is
> no cloud path anywhere in the submission.
>
> Could you please confirm which is correct?
>
> 1. An FP16 TFLite model under 20 MB satisfies the quantization criterion, or
> 2. INT8 weights are mandatory, and an FP16 artifact would lose the
>    quantization score.
>
> If INT8 is mandatory, we would be grateful to know the deadline for
> resubmitting, since we would need to re-export and re-benchmark.
>
> We would also appreciate confirmation on four items the deck leaves open:
>
> a) The Track 2 fixed vocabulary list and test manifest — are these available,
>    and where can we download them?
> b) The exact boundary for the 50 ms latency target: per-frame inference, or
>    end-to-end clip-to-JSON? We have per-frame detector measurements on a real
>    arm64 device and want to report the right quantity.
> c) The submission package layout: page 11 names `results/predictions.jsonl`
>    while the page 12 diagram names `results/predictions.json`. Which is
>    expected? We can supply either or both.
> d) The deadline timezone: the slides state 11:59 PM without a timezone. Is
>    that IST?
>
> Thank you. We will submit by the published deadline.
>
> Best regards,
> `<NAME>`, EchoGesture team — Track 2
> `<COLLEGE NAME> · <TEAM NAME>`
> `<EMAIL / PHONE>`

**Send from the registered address** so the organizer can match the request to
your team. Replace the three `<...>` placeholders before sending — the same
strings that still need filling on slide 1 of the deck.

---

## Why this is worth sending

The quantization criterion is **20% of the scored rubric**. Two readings are
defensible from the deck, and the difference between them is a fifth of the
score. A written answer settles it; silence leaves it to whichever judge reads
the deck that day.

It also asks (b) and (c), which are worth more than they look:
- We have per-frame detector numbers but no clip-to-JSON number on device. If
  the target is end-to-end, that changes what we should measure.
- p.11 and p.12 disagree on the deliverable filename. We currently ship **both**
  `predictions.json` and `predictions.jsonl`, so either answer is satisfied —
  but confirming it costs nothing.

## What NOT to do

Do **not** claim INT8 compliance anywhere in the submission. The shipped artifact
is FP16 (`models/landmark_model_meta.json` → `"quantization": "float16"`). If the
answer comes back "INT8 mandatory", that is a real model change and needs time —
which is the argument for sending this today rather than on the 21st.
