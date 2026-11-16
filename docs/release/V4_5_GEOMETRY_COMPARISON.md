# V4.5 — geometry comparison

All numbers produced through the frozen production `predict()`. Only the frame
geometry differs between rows. Same nine members, same hash pin, same mask, same
mean-logit aggregator.

Reproduce:

```bash
python evaluation/device/geometry_paired.py --clips <DIR> --out out.json
python evaluation/device/geometry_latency.py
```

---

## Source geometry — checked, not assumed

Every ASL Citizen clip we hold is **640×480 (4:3)**:

- val_30: 30/30 clips
- val_346: 40/40 sampled from 346

So "native" means **no resize at all**, and the shipped 256 path is a *squash*
from 4:3 to 1:1 — a **1.33× horizontal distortion** applied to every hand.

That is not the same as letterboxing, and it is not the same as preserving
aspect ratio. Dev's `app640x480` column in `geometry_pair.json` is
*letterboxed*, which is a third thing again and scores 86.31% — higher than the
true native path's 85.51%. Letterbox and native must not be conflated.

---

## Accuracy — validation split, the population we may select on

**346 clips, 5 signers.** This split did not choose this change.

| geometry | correct | top-1 | errors |
| --- | --: | --: | --: |
| 256 squash | 285/346 | **82.37%** | 1 |
| native | 297/346 | **85.84%** | 1 |

Paired (n=345, the one clip with no hand in either geometry excluded):

| | count |
| --- | --: |
| both correct | 275 |
| squash only correct | 10 |
| native only correct | 22 |
| **net** | **+12** |
| McNemar exact two-sided | **0.0501** |

The single error is `5989539775823847-INTUITIVE` — the evaluator raises
"No hand detected in any frame" under both geometries. That is a detector
failure on the clip, not a geometry effect, and it is identical in both arms.

#
> **Signer labels.** Signers appear as `S…` pseudonyms produced by the repository's
> own `hash_signer_id` scheme (HMAC-SHA256 with a fixed project salt). The raw
> ASL Citizen participant IDs are deliberately not published and the
> participant→pseudonym mapping is not recorded in this repository.

## Per signer

| signer | n | squash | native | Δ pp |
| --- | --: | --: | --: | --: |
| S28C1 | 24 | 79.2% | 79.2% | 0.0 |
| S4ABE | 104 | 82.7% | 86.5% | +3.8 |
| S1BC1 | 101 | 79.2% | 84.2% | +5.0 |
| S476B | 103 | 86.4% | 90.3% | +3.9 |
| SCA7F | 13 | 84.6% | 76.9% | −7.7 |

Four of five improve. SCA7F has 13 clips, so its one-clip drop is 7.7 points — not
evidence of a signer-specific regression.

### Per class

| | classes |
| --- | --: |
| native better | 16 |
| unchanged | 76 |
| native worse | 8 |

Every regression is exactly **one clip**: DOWNSIZE1, CLOSE, FOREIGNER1, COPY,
DOG1, AXE1, ZEBRA, BEE1 (4→3 or 6→5 each).

---

## Accuracy — smaller population, for reference only

**val_30, 30 clips:** 76.67% → 86.67%. 4 gained, 1 lost. Direction agrees, but
n=30 cannot carry a decision.

---

## Dev's test-split figures — agreement only, never selection

| geometry | test top-1 |
| --- | --: |
| 256 squash (fresh extraction) | 82.47% |
| 512 letterbox | 85.67% |
| 640×480 letterboxed to square | 86.31% |
| **native 640×480** | **85.51%** |

Same 1,249 clips. **I did not re-run these**; reproducing 1,249 clips twice is a
multi-hour job and the validation split already answers the release question.
They are recorded as Dev's measurement agreeing in direction, nothing more.

Also note: the **cached** landmarks give 76.94% at the same geometry, because
presence differs from fresh extraction on 29% of test clips. Every row above is
fresh extraction. The cache discrepancy is a separate defect documented in
`V4_5_DEV_UPDATE_REVIEW.md` §1.2.

---

## Runtime impact

| geometry | p50 | min | max | n |
| --- | --: | --: | --: | --: |
| 256 squash | 236.1 ms | 203.3 | 287.1 | 12 |
| native | 241.8 ms | 199.4 | 339.6 | 12 |

**+5.7 ms p50.** The resize is a small fraction of a pipeline dominated by
MediaPipe. Laptop CPU, full clip-to-JSON; not a phone measurement.

---

## Behavioural check on a misclassified clip

`15347326711184706-NOON` is the clearest single example:

| geometry | prediction | confidence |
| --- | --- | --- |
| 256 squash | `BEE1` | 0.2678 |
| native | `NOON1` | 0.3450 |

The true gloss is `NOON`. Same model, same weights, different frame geometry.

---

## Default preservation — proven, not claimed

| check | result |
| --- | --- |
| default vs explicit `size=256` | **byte-identical** features and presence |
| default vs native | differs, as required |
| default prediction on `ANYONE` clip | `ANYONE`, 0.8631 — unchanged |
| native frame shape | 640×480, not resized |

A shape check alone would not have caught a sentinel bug; the test compares
actual tensors.