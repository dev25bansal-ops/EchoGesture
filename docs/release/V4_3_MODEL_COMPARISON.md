# V4.3 — A/B/C model comparison

Three configurations, identical clips, identical features, identical
segmentation. The only variable is the model set.

| | Configuration |
| --- | --- |
| **A** | V4.2 frozen four-member ensemble (old binaries, preserved for rollback) |
| **B** | New four masked x,y members (Dev: 74.06% test) |
| **C** | New nine-member masked x,y,z ensemble (Dev: 77.02% test) |

## Results

Full recorded interval, no segmentation. Two splits, three conditions.

| Split | Condition | n | A | B | C |
| --- | --- | --: | --: | --: | --: |
| train | clean | 32 | 29/32 | 32/32 | 31/32 |
| train | 2 s lead idle | 32 | 24/32 | 28/32 | 29/32 |
| val | clean | 40 | 31/40 | 30/40 | 31/40 |
| val | 2 s lead idle | 40 | 20/40 | 25/40 | 28/40 |
| val | 2 s idle each side | 40 | 13/40 | 26/40 | 27/40 |
| **Aggregate top-1** | | **184** | **117 (63.6%)** | **141 (76.6%)** | **146 (79.3%)** |
| **Aggregate top-5** | | **184** | **160 (87.0%)** | **172 (93.5%)** | **177 (96.2%)** |

## What the numbers say

**C is the best configuration on both metrics**, and the improvement over A is
large: +15.7 points top-1, +9.2 points top-5, aggregated over 184 evaluations.

**Most of the gain is the mask, not the z channel.** B (mask, x/y only) already
reaches 76.6%; adding the five z members takes it to 79.3%. That is consistent
with Dev's own finding that omitting the mask was an export defect rather than a
modelling choice — the mask recovers the model that was trained, and z is a
genuine but secondary improvement.

**On clean validation clips alone, C shows no improvement over A** (31/40 both).
The gain is concentrated in recordings with idle frames, which is exactly where
an attention mask over absent-hand frames should pay:

- 2 s lead: **20 → 28** (+8)
- 2 s each side: **13 → 27** (+14)

A degrades steeply as idle is added; B and C barely move. That is the mask doing
its job.

**Train clean favours B over C** (32/32 vs 31/32). One clip. Not a reason to
prefer B, but not something to hide either.

## Segmentation re-check under the new ensemble

A new model could change how trimming behaves, so it was re-measured rather than
assumed.

Clean clips, validation (n=40):

| | A | B | C |
| --- | --: | --: | --: |
| full interval | 31 | 30 | 31 |
| trimmed | 28 | 32 | 30 |

Padded clips (2 s each side, val n=40):

| | A | B | C |
| --- | --: | --: | --: |
| full interval | 13 | 26 | 27 |
| trimmed | 28 | 32 | 29 |

**The V4.2 segmentation decision still holds and is not re-tuned.** Trimming
continues to be worth it on padded recordings by a wide margin, and continues to
be uncertain on clean ones — under C, trimming gives 30 vs 31 clean (1 clip
down, inside noise) and 29 vs 27 padded (+2). No threshold was adjusted against
these clips: the decision rests on the same V4.2 evidence, only cross-checked
here. Nothing was selected by maximising confidence.

## Size and latency

| | A (four) | C (nine) |
| --- | --: | --: |
| Members | 3.38 MB | 8.11 MB |
| Detector | 7.46 MB | 7.46 MB |
| **Detector-inclusive** | **10.84 MB** | **15.57 MB** |
| APK | 77.81 MB | 83.35 MB |
| Per-member latency (host CPU) | 0.05 ms | 0.05 ms |
| Ensemble latency (host CPU) | 0.2 ms | 0.5 ms |

**These latencies are a laptop measurement and are not Android phone numbers.**
Dev's reported 6.56 ms is likewise a laptop result. On-device figures remain
unmeasured — see `V4_3_RELEASE_STATUS.md`.

## Caveats that limit how far these numbers go

1. **The historical test split has already been used during model development.**
   Dev's 77.02% is not an independent estimate, and nothing here reproduces it —
   these are 72 authorised clips (32 train signers, 5 validation signers), not
   the 1,249-clip test set.
2. **Small samples.** Validation is 5 signers. A 2-clip difference is noise.
3. **Idle is simulated** by prepending/appending frames carrying no detected
   hand. That is what an idle camera produces, but it is not a live recording.
4. **No phone.** Every figure is Python on authorised clips.
5. **Not a like-for-like with Dev's numbers**: different clips, different
   population, no test split.

## Recommendation

Ship **C** (nine members). It is best on both aggregates, survives parity, and
its extra 4.7 MB is acceptable against a 15.57 MB model bundle. Keep A's APK as
the fallback, which is untouched.

If model size ever becomes binding, **B is the better trade than dropping
members from C**: B captures most of the gain at 3.38 MB.