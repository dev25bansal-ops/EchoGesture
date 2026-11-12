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

**Resolved by audit — please confirm or correct.** The numbers are now accounted
for, and none of them is 83.77% from a shipped artifact:

| Number | What it is | Artifact |
| --- | --- | --- |
| 83.77% | 289/345. Claimed in `RESULTS.md`. **No artifact in the repo produces it.** | none |
| 76.52% | `eval_summary_val.json`, blob `9ae0f3ba` | frozen at commit `226bd41` (DEV-03, single-model era) |
| 75.36% | 260/345, recompute of the committed predictions | `results/predictions_val.jsonl`, blob `b770ec8`, frozen at `226bd41` |

The 345 denominator is identical across all three, so this is **not** a
denominator difference. The committed val predictions are byte-identical at
every later commit including the tip of `hackathon/dev-model`, and their latency
profile (p50 0.546 ms) is single-model, not ensemble (1.205 ms). The committed
val evidence is a stale single-model file that was never regenerated for the
ensemble.

So: 83.77% has no supporting artifact; 76.52% and 75.36% describe the same
single-model run. **We ship 72.78% test and cite 75.36% as the validation
figure, never 83.77%.** Please either produce an artifact for 83.77% or confirm
it should be struck from `RESULTS.md`.

## 7. Explain the 920 vs 909 robustness baseline

**Resolved by audit — please confirm.** 920/1249 = 73.66%, which matches
`clean.top1` in `results/robustness.json` exactly. That file was bundled at
commit `4594bfa` but is absent from the current tree. Its harness
(`scripts/ml/robustness.py`) runs **PyTorch checkpoints over cached landmarks**,
not the FP16 TFLite members — a different inference path that would plausibly
differ by a few clips.

Separately: the robustness figures we actually publish (crop −10.00, occlusion
−6.67 pp) come from the **newer video-level ensemble harness**
(`robustness_ensemble/RESULTS.md`, clean 23/30 = 76.67%), not from
`robustness.json`. Please confirm the 11-clip gap is the PyTorch-vs-TFLite path
difference and not a bug in the shipped predictions.

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