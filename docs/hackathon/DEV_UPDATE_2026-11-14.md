# Dev's changes, 13–14 Nov — read this first

A single entry point for the last two days of work. Six commits on
`hackathon/dev-model`; the previous handoff ([DEV_HANDOFF_MODEL.md](DEV_HANDOFF_MODEL.md))
is still accurate except where this file corrects it.

**Headline: test top-1 went 72.78% → 77.02%. Almost all of that was a bug, not
tuning.** There is one change you have to make on your side, and the app will
not start until you do.

---

## 1. The app will not start — this is the thing to act on

`LandmarkClassifier.kt` asserts:

```kotlin
require(interpreter.inputTensorCount == 1) { ... }                       // now 2
check(inputTensor.shape().contentEquals(intArrayOf(1, 24, 255))) { ... } // now 381 for 5 of 9
```

Both fail now, **at member construction, before any inference**. Loud and
early, which is the right way for it to fail — but it does fail.

Full instructions in **[APP_MIGRATION_MASKED_MODELS.md](APP_MIGRATION_MASKED_MODELS.md)**.
The short version:

1. Accept **two** inputs, resolved **by name** (`"features"` and `"mask"`).
   Do not index positionally: the converter reports the mask as input 0.
2. Read the feature width **per member** from `members_detail` in
   `landmark_model_meta.json` — 255 for the four legacy members, 381 for the
   five z-channel ones.

My suggestion: land the two-input plumbing first, still at 255-wide. That is a
smaller change, takes the app from 72.78% to 74.06% on its own, and proves the
mask path works before width enters the picture. The `use_z` branch of
`FeatureTransform` is the piece to be most careful with — a reordered feature
vector produces a model that runs perfectly and predicts nonsense.

---

## 2. What was actually wrong

The classifier was **always trained** with a hand-presence mask applied to its
attention: `train.TemporalModel` takes `key_padding_mask`, and `train.py`
always supplies it, so frames with no detected hand get zero attention weight.

`build_tf_model` declared only a `features` input. **Every artifact this
project ever shipped was therefore a different model than the one that was
trained** — it attended over absent-hand frames carrying `present=0` and
all-zero coordinates.

`src/main.py` shows the consequence:

```python
feats, _mask = build_features(lm, present, expected_frames=target_frames)
```

It computed the mask and threw it away, because nothing downstream accepted it.

Worth, measured on one checkpoint with identical weights:

| | val | test |
| --- | --: | --: |
| no mask | 80.87% | 70.30% |
| **with mask** | **83.77%** | **72.14%** |

### Why our own safety gate missed it

`export_ensemble.py` checks parity between the TF port and PyTorch. Both sides
lacked the mask, so **they agreed perfectly**. A parity gate catches drift
between two paths; it cannot catch both paths being wrong.

What exposed it was comparing the exported artifact's accuracy against the
number the training run had already printed. That check is now worth having
anywhere we convert a model.

---

## 3. Current numbers — all verified by reloading from `models/`

| | val | test |
| --- | --: | --: |
| before (4 members, no mask) | 83.77% | 72.78% |
| after (9 members, masked) | **87.25%** | **77.02%** |

- Top-5: **92.79%** (was 87.51%)
- Per-signer mean: **77.53%** (was 73.36%); min 61.27% (was 53.5%), max 88.66%
- **Worst signer gained the most** (P42: 53.5% → 61.3%) — the shape a
  systematic fix makes, not the shape tuning makes
- 9 members, 8.51 MB total (was 4 members, 3.46 MB)
- All 9 at **100% argmax parity** with PyTorch; max|d| 3.8e-06 to 6.1e-05

The five new members are `mirror_handdrop_tcn_z`, identical recipe and
architecture to the existing TCNs but taking 381-wide x,y,z features.
Measured paired by seed against a matched no-z run over 5 seeds:
**+0.88 test (4/5 seeds positive), +0.52 val (3/5)**. Neither split reaches
significance alone (paired t 1.72 and 0.98 at df=4) — the z members earn their
place in the *combination*, not individually. The 9-member ensemble against the
shipped 4 is `+66/-29` on test, McNemar **z=3.80** — recomputed by reloading
every `.tflite` from `models/`, so it is reproducible from the shipped files.

A note on how easily this number moves: an earlier 3-seed version of this
measurement said val +1.35 / test +0.53, and I initially read z as "a val
effect that does not hold on test". Adding two seeds swapped it to
test +0.88 / val +0.52. I have recorded both readings in
[EXPERIMENTS.md](../../results/EXPERIMENTS.md) rather than quietly keeping the
flattering one — at this effect size, seed count changes the story.

**Latency is not re-measured on device.** The old 1.21 ms p50 was four
members; nine measure 6.56 ms/clip on my laptop, but the on-device figure is
yours to take. Everything scales linearly so it should be fine — it just is
not a measurement I can make.

---

## 4. It knows when it does not know — useful for the demo

Confidence is well calibrated:

| confidence | clips | accuracy | share of all errors |
| --- | --: | --: | --: |
| below 0.2 | 148 | 31.1% | **35.5%** |
| 0.2 – 0.4 | 211 | 47.4% | **38.7%** |
| 0.6 – 0.8 | 232 | 90.9% | 7.3% |
| above 0.8 | 462 | **98.9%** | 1.7% |

**74% of all mistakes are below 0.4 confidence.** Showing a result only when
confidence is high, and asking for a repeat otherwise, converts the most
visible failure mode — a fluent-looking wrong label — into an honest "sign
again". Abstaining below 0.5 still answers 63% of clips, at 94.3% accuracy.

The top1-minus-top2 margin is sharper still: below 0.05 is 26.8% accurate,
above 0.5 is 96.8%. Either signal works; the margin separates harder.

Caution for the demo: on real clips I saw one *correct* answer at 0.103
confidence and one wrong one at 0.342. Low confidence is a prompt to re-sign,
not proof of an error.

---

## 5. Hand detection is fine

54% of sampled frames have no hand, which looks alarming until you ask *where*
they are. Presence by clip decile:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| hand found | 6% | 32% | 60% | 70% | **77%** | 73% | 54% | 37% | 30% | 27% |

A clean inverted U. The clips carry their own lead-in and lead-out — the
signer standing, raising hands, lowering them. ASL Citizen is not pre-trimmed
to the sign. **If the app shows "no hand" for a second before a sign starts,
that is correct behaviour, not a fault.**

---

## 6. Everything that did NOT work

Kept so nobody repeats it. Full detail in
[results/EXPERIMENTS.md](../../results/EXPERIMENTS.md):

| tried | result |
| --- | --- |
| more data (merge val, +19% clips) | +0.24, noise |
| wider model (dim 256) | −15 pts |
| 32 frames instead of 24 | −8 pts |
| speed perturbation | −5.45 pts |
| strict resample "phantom hand" fix | −8 pts |
| mirror TTA | −1.01 pts |
| INT8 quantization | −6.37 pts |
| multi-crop TTA | 0, won val / lost test |
| mean-probability aggregation | +0.9 test / −0.9 val, rejected |
| bigger ensembles beyond 5 members | negative — members fail on the same clips |

182 of 1,249 test clips are wrong for **every** member. That correlated-error
core is the binding constraint, and it is a data limit: ASL Citizen filmed
these 100 signs with only ~32 participants.

---

## 7. Nothing here is verified on Android

Every number in this document comes from Python, reloading `.tflite` from
`models/`. The app has not been built or run against these models — that is
your side, and section 1 is what it needs.

---

## Commits

| | |
| --- | --- |
| `ccfd810` | export the attention mask; +4.24 points |
| `d514fb1` | main.py feeds the mask and per-member width |
| `4bf5b28` | evaluate.py same fix |
| `0c20c52`, `a15c2f1`, `93fbd49`, `7784d19` | handoff, app migration note, results, detection/calibration |
