# Judging scoreboard — Track 02

Official weights from deck p.10: **Execution & Efficiency 40%** (model
quantization 20% · edge latency 20%), **Quality & Accuracy 60%** (top-1 35% ·
robustness & HUD 25%).

**No judge scores are invented here.** This maps our measured evidence onto each
criterion so you can see where points are likely and where they are not. The
"likely credit" column is my estimate of the *evidence* available to a judge,
not a prediction of their score.

Every number below is quoted from [`results/results.csv`](../../results/results.csv).
That file is the single source; this document adds no measurement of its own.

> **Rewritten 2026-11-22.** An earlier version of this page was written on
> 2026-11-13 and carried figures from the superseded pre-rebuild landmark cache
> (72.78% test, 84.39% val, 275.8 ms p95, 11.36 MB) plus two robustness
> conditions that no longer exist in any current measurement. Every one of those
> was wrong by the deadline, and several contradicted `results.csv` outright.
> The measurements that replaced them are below.

---

## Execution & Efficiency — 40%

| Criterion | Weight | Our evidence | Likely credit | Gap |
| --- | --: | --- | --- | --- |
| Model quantization | 20% | Nine FP16 TFLite members, **8.51 MB**, plus a 7.82 MB FP16 detector — **16.33 MB total**, under both the 20 MB and 25 MB readings. FP16 costs **0.00 pt** against FP32 (`accuracy_drop_quantization`). INT8 was tried and measured: **−6.37 pt**, and two TFLite runtimes disagreed on the file, so it was rejected. | **Partial** — size passes every stated budget; the literal word "INT8" is not satisfied | Needs the organizer's answer on whether "quantized TFLite <20 MB" is satisfied by FP16. That clarification is drafted and **not sent**. |
| Edge latency | 20% | On a physical Vivo V2404 (SM7550) with the GPU delegate: detector **p50 34–45 ms**, **p95 46–54 ms with an empty frame**, **68–70 ms once a hand is present**. Nine-member classifier itself is 3.91–4.78 ms p95. | **Low on p95, real on p50** — the ≤50 ms target is met at p50 in all four runs and met at p95 only in the empty-frame regime. | The hand-present p95 of 68–70 ms is the honest number and it fails. It was measured on the **debuggable** build, so it is an upper bound; the signed release build was not re-measured for p95. |

**Section estimate: mixed.** Size is genuinely good. Latency passes at p50 and
fails at p95 with a hand in frame — say exactly that; do not round it into
"under 50 ms".

---

## Quality & Accuracy — 60%

| Criterion | Weight | Our evidence | Likely credit | Gap |
| --- | --: | --- | --- | --- |
| Top-1 accuracy | 35% | **85.45%** top-1, **96.64%** top-5, over **1,251** signer-disjoint test clips from 11 unseen signers against a 1.00% chance floor. Per-signer mean 86.02%, worst signer 72.54%. Best single member 81.53%, so the ensemble is worth +3.92 pt. | **Good** — clears the 70% target by 15 points, signer-disjoint, reproducible from `scripts/ml/eval_tflite_cache.py` | This is a **desktop** measurement on a corpus seen during development. No phone accuracy figure exists. The organizer's own held-out set is unknown to us. |
| Robustness & delivery (occlusions) | 25% (shared) | Six landmark corruptions. Five pass: camera-further 25% **−0.24**, half framerate **−1.52**, drop landmarks 15% **−2.16**, jitter σ0.03 **−3.20**; jitter σ0.01 is *better* than clean (**+0.16**). Only **`drop_hand_25pct`** fails, at **+5.76 pt** raw. At the acceptance gate the app actually ships (margin ≥ 0.15) it is **+2.39 pt** on accepted clips, at 86% clean / 80% occluded coverage. Separately, **pixel** robustness is **+4.48 pt PASS** across four corruptions — but that is 512-letterbox geometry, not the app's native 640×480 path. | **Partial** — one of six landmark conditions fails raw; it passes at the shipped operating point, and both rows are in `results.csv` | Native *pixel* robustness (the app's actual geometry) is **unmeasured**. Retraining with occlusion augmentation is the only real fix for `drop_hand_25pct`, and **it was tried and measured: a harder-hand-dropout retrain scored 80.26% clean against the shipped 85.45%, moving the drop only 5.76 → 5.12** — i.e. it bought 0.64 pt of robustness for 5.2 pt of accuracy, because 17–24 clips per class cannot support harder augmentation. Do **not** quote the gated number as unconditional robustness — it is a coverage trade. |
| Robustness & delivery (HUD clarity) | 25% (shared) | The app never guesses. A capture with no hands or too few frames is **declined with a specific reason** instead of labelled: "No hands in view", "Hands were visible for too little of that recording". A low-margin reading is still shown, but labelled *"Uncertain — this is the model's best guess, not a confirmed reading."* A model score is captioned as a ranking over 100 signs, never as a probability of correctness. Offline posture is visible on screen (Offline badge, no INTERNET permission). | **Good** — the distinction between *declined* and *uncertain* is implemented in one place and was exercised on the physical device | The refusal path was observed once (a 25 s no-hands capture). It is not a measured HUD-usability study with real signers. Full evidence: [`docs/HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md`](../HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md). |

**Section estimate: strong.** Accuracy clears its bar by 15 points and the HUD
behaviour is specific and honest. Robustness is half-passing, disclosed.

---

## Where the submission actually stands

| Area | State |
| --- | --- |
| Accuracy (35%) | **Passes** — 85.45% vs ≥70%, signer-disjoint, desktop measurement |
| Robustness, landmark (part of 25%) | **1 of 6 conditions fails raw** (+5.76); **passes** at the shipped gate (+2.39, 86%/80% coverage) |
| Robustness, pixel (part of 25%) | **Passes** — +4.48 pt worst case across four corruptions, but at 512-letterbox geometry; the app's native geometry is unmeasured |
| HUD (part of 25%) | **Built and verified on a physical phone** — declines rather than guesses |
| Size (part of 20%) | **Passes** both the 20 MB and 25 MB readings at 16.33 MB |
| Quantization format | **FP16**, disclosed; INT8 measured at −6.37 pt and rejected; requirement wording ambiguous |
| Latency (20%) | **p50 passes** (34–45 ms); **p95 fails with a hand present** (68–70 ms) |

## The three levers, in order of value

1. **Live-camera p95 on the release build.** Every latency figure came from the
   debuggable build, which carries instrumentation overhead the shipped APK does
   not. Re-measuring `detect_p95` with a hand in frame on the release build is
   the only way the latency row can move, and it is a measurement, not a claim.
2. **The INT8 clarification.** If "quantized TFLite <20 MB" is accepted, the
   project already passes that budget without touching the model.
3. **Occlusion augmentation.** The single real robustness fix. It has been tried
   and measured: it loses 5.2 points of clean accuracy for 0.64 of robustness, so
   it is closed by measurement rather than by lack of time.

What will **not** help: more UI polish, more slide design, or restating estimates
as measurements. Judges re-measure latency on their own device (deck p.11).
