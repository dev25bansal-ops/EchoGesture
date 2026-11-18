# RECOGNITION_ROOT_CAUSE

## The defect that was proven

**A declined guess was presented as a recognized sign.**

`CameraScreen.renderResult()` computed `ResultPolicy.isAccepted(...)` correctly,
then assigned `captionText.text = result.prediction` **unconditionally**. A
reading with a 0.02 margin — the policy's own test case — was rendered under the
heading "Recognised sign", spoken aloud when speech was enabled, and written to
recognized-sign history.

This is exactly what the user described: "it tells me the wrong word". The model
was often right to be unsure; the UI was wrong to say so confidently.

**Evidence it was the shipped behaviour, not a mistake in reading the code:**
`ResultCardRenderTest.lowMarginResultIsShownButLabelledUncertain` asserted it:

```kotlin
assertEquals("an uncertain result must still show its prediction",
             "WHATFOR1", caption.text.toString())
```

The test had to be changed, because it pinned the defect.

## What the evidence says about the classifier itself

It is **not** broken, and three measurements say so:

1. **Tensor-level parity is exact.** Feature tensor differences 0.000000, mask
   mismatches 0, worst logit difference 0.000005, top-1 agreement 6/6 against a
   Python fixture spanning all five signers.
2. **87.5% on 345 authorised clips** across five unseen signers, with no signer
   below 83%.
3. **95.7% precision on what the app actually presents**: of 30 labelled clips,
   22 were correct and accepted, 1 was accepted but wrong, and 7 were honestly
   declined.

The 43 errors on the 345-clip set are **41 unique confusion pairs** — a
decision-boundary pattern, not a label-mapping defect. The one structural case
(8HOUR↔MAPLE) is bidirectional and instance-specific, and **both platforms make
the same mistake on the same clip**.

## What was NOT the cause

| Hypothesis | Verdict |
| --- | --- |
| Wrong feature transform | DISPROVED — bit-identical to Python |
| Wrong attention mask | DISPROVED — 0 mismatches |
| Wrong label order | DISPROVED — 41 unique pairs, no offset |
| Wrong ensemble aggregation | DISPROVED — mean logits then one softmax, matches |
| Android inference defect | DISPROVED — logits agree to 5e-6 |
| Low analyser frame rate | NOT SUPPORTED — measured 29.4 fps, 0 dropped, limit=CAMERA |
| Handedness swap needed | DISPROVED — both platforms assign Left→0 |

## The one thing still unknown

**No physical phone was attached at any point.** Every number above is a
labelled-clip replay or a source/bytecode analysis run on `eg34`, an emulator
whose camera is a synthetic pattern that never contains a hand.

So this is unresolved: whether a person signing into a real phone produces
landmarks that land in the same region as these clips. Nothing in this report
claims otherwise, and the phrase "live recognition verified" does not appear.

The declining behaviour is the mitigation: an out-of-distribution live sign
should produce a low margin and be declined with "Sign not recognised reliably"
rather than named wrongly.
