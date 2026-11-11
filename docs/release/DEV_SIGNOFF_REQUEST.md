# Dev sign-off request — final review before submission

**To:** Dev (`dev25bansal-ops`)
**From:** Abhijeet
**PR:** https://github.com/dev25bansal-ops/EchoGesture/pull/1 (draft, base `hackathon/two-person-sprint`)
**Head:** `hackathon/abhijeet-public` @ `26b29905a072d96df31a30dbabc7dc8f322314d6`
**Deadline:** internal 17 Nov 2026, competition 18 Nov 2026 23:59 IST

Nothing below asks for retraining. This is a freeze-and-confirm review.

---

## 1. Confirm the four member hashes

Computed from the working tree; please confirm these are the intended shipped
members and not a stale export.

| Member | SHA-256 |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a` |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63` |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3` |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21` |

Total 3,544,080 B. Metadata `landmark_model_meta.json` declares this exact
`members` list and order.

## 2. Confirm member order

The runtime iterates `members` in metadata order and averages logits
element-wise before a single softmax. Because logits are averaged (not stacked
per-class offsets), order should be mathematically irrelevant — please confirm
that is true for these four architectures, or tell me if any member is
class-remapped.

## 3. Confirm feature version

`[1, 24, 255]` float32. 255 = position + velocity + acceleration per hand,
inter-hand geometry, presence masks, canonical `[Left, Right]` slots.
`features_sha256 = baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84`.

## 4. Confirm `STRICT_RESAMPLE=False`

The shipped feature extractor uses `STRICT_RESAMPLE=False`. Please confirm this
matches training. If training used `True`, every shipped metric is invalid.

## 5. Confirm the test reproduction

| Population | Result |
| --- | --- |
| Test, scored | 909/1249 = **72.78%** top-1 |
| Test, top-5 | 1093/1249 = **87.51%** |
| Test, all eligible | 72.43% (1,255 denominator) |
| Unscored eligible | **6** clips, no hand detected |

Please confirm 909/1249 reproduces on your machine from the shipped members.

## 6. Explain 83.77% validation vs 75.36% recomputed

This is the largest unexplained discrepancy in the repo.

- `RESULTS.md` quotes **83.77%** validation.
- The committed `predictions_val.jsonl` recomputes to **260/345 = 75.36%**.
- `eval_summary_val.json` says **76.52%**.

Three different numbers for one population. No artifact in the repo supports
83.77%. Which is the number that should appear in the submission, and what
produced the other two? If 83.77% came from a different split, a different
model, or a per-class average rather than per-clip, say which.

**We are currently shipping 72.78% test and do not cite 83.77% anywhere.** If it
is legitimate I'd like to cite it; if not, it should be struck from `RESULTS.md`
so it cannot leak into a submission.

## 7. Explain the 920 vs 909 robustness baseline

The robustness harness reports a "clean" baseline of **920/1,249** correct, but
the shipped predictions give **909/1,249** — an 11-clip gap on what should be the
same condition. Is the robustness clean set a different 1,249 clips, a re-decode
with different corruption seeding, or frames sampled differently from the
evaluation path? This changes the reported percentage-point drops
(crop −10.00, occlusion −6.67).

## 8. Confirm the six no-hand failures

Six eligible clips produce no prediction at all. Confirm this is the expected
count and that they are detector failures rather than decode/feature failures.
They are disclosed in the submission rather than dropped.

## 9. Confirm quantization and size evidence

Shipped artifact is **FP16 weights, float32 IO**, 11.37 MB with detector.
The organizer slides say "INT8 ≤20 MB" on p.9/p.10 but the scored jury matrix on
p.11 says only "TFLite/ONNX <20MB". I have drafted a clarification email; I am
**not** substituting your lower-accuracy INT8 artifact without your approval.

If INT8 turns out to be mandatory, what is the accuracy of your INT8 candidate
and would you approve shipping it?

## 10. Confirm licensing and redistribution rights

The weights derive from ASL Citizen / WLASL. They are gitignored and **not**
published. Please state whether redistribution is permitted, and under what
terms — or confirm it is not, so the submission says so plainly.

## 11. Independently run the extracted release

Please extract `evaluation/device/release_final2/echogesture_final2_ensemble.zip`
into a clean directory, run it on one MP4, and confirm the prediction matches
this branch. This is the one check only an independent machine can make.

## 12. Review PR #1

Nine of these are answered by reading the diff. A review — even a "LGTM with
these caveats" — is what unblocks the merge gate.

---

## What I need back, minimum

1. Items 6 and 7 explained (the metric discrepancies)
2. Item 4 confirmed (`STRICT_RESAMPLE=False`)
3. Items 1 and 11 confirmed (hashes, independent extraction)
4. Item 10 stated (licensing, either way)
5. A review on PR #1

Items 2, 3, 5, 8, 9, 12 are confirmations; if any is wrong, say so immediately
rather than at the deadline.