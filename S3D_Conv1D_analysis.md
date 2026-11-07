# S3D-Conv1D — authoritative extraction (MDPI full text, fetched 2026-11-07)

Source: https://www.mdpi.com/1999-4893/19/4/248
DOI: 10.3390/a19040248 · Algorithms 2026, 19(4), 248 · published 25 Mar 2026 · CC-BY gold OA
Authors: Samuel Longwani Kimpinde, Peter O. Olukanmi (Univ. of Johannesburg)
Code + FP32/INT8 models: https://github.com/Samuel-KIMPINDE/S3D-Conv1D-for-SLR-for-Microcontrollers
Data: WLASL-Processed (Kaggle risangbaskoro), WLASL2000-Resized (Kaggle sttaseen), SemLex (github leekezar/SemLex), ASL-Kimpinde (zenodo 18754451)

## Discrepancy resolved
OpenAlex said 98.4% / 895.9 KB; Semantic Scholar said 98.96% / 883 KB.
AUTHORITATIVE (MDPI meta description + Table 5 + Table 15): **98.96% FP32, 98.60% INT8, 883.1 KB INT8**.
OpenAlex 98.4% was a mis-extraction. Semantic Scholar was right.

## Input representation — CRITICAL
**Grayscale RGB video frames at 64×64, T=24 frames, normalized [0,1], stored as .npy.**
NOT MediaPipe landmarks. NOT pose. Raw pixel pipeline.
Preprocessing: OpenCV decode → grayscale → resize 64×64 → normalize → temporal resample to T=24 → augment.
Rationale given: grayscale "reduces memory usage and stabilizes activation ranges, beneficial for INT8 calibration."

## Architecture (Table 2 + Algorithm 1)
- Conv2D-1: 3×3×1→12 filters, stride 1, same pad, ReLU, MaxPool 2×2
- Conv2D-2: 3×3×12→24, stride 1, same pad, ReLU, MaxPool 2×2
- Flatten each frame → 24×24×24 = wait: peak act 32×32×24 → flatten to 6144-dim per frame
- Conv1D: kernel k=3, stride 1, same pad, 6144→48 channels, ReLU
- GlobalAveragePooling over T
- Dense 48→K, softmax
- Total params: 892,420 (892 K). INT8 size 883.1 KB (vs FP32 3,492 KB, ×4.0 compression)

## Results (Table 15 accuracy, Table 20 size/latency)
| Model | WLASL100 FP32 | WLASL100 INT8 | SemLex100 FP32 | SemLex100 INT8 |
|---|---|---|---|---|
| S3D | 99.29 | 99.20 | 64.29 | 60.38 |
| CNN+RNN | 99.29 | 99.29 | 89.62 | 89.62 |
| e-LSTM | 44.20 | 20.09 | 59.78 | 56.95 |
| MobileNet variant | 99.46 | 96.36 | 97.57 | 91.04 |
| **S3D-Conv1D** | **98.96** | **98.60** | **82.48** | **79.92** |

ULTRA LIGHT variant (Table 17): 24.7 KB INT8, WLASL100 98.51% QAT (drop −0.39), WLASL300 74.11% QAT (drop −3.11), 0.94 MFLOPs.

## Runtime (Table 22 — CPU proxy, NOT device)
Intel Celeron N4500, 1.10 GHz, single-threaded TFLite, 20-run median:
| Model | FLOPs (M) | Latency (ms) | FPS |
|---|---|---|---|
| S3D | 0.20 | 9126 | 0.11 |
| CNN+RNN | 1267 | 84.9 | 11.8 |
| e-LSTM | 0.60 | 39.3 | 25.5 |
| MobileNet | 105.4 | 1895 | 0.53 |
| **S3D-Conv1D** | **191.1** | **26.0** | **38.4** |

Energy proxy = CPU% × latency (comparative only, authors explicitly disclaim cross-platform comparison).

## Split methodology — RED FLAG
- **NOT signer-disjoint.** Paper says explicitly: "The official WLASL benchmark does not enforce a fully signer-disjoint split; our protocol follows this convention, meaning that reported accuracies reflect within-distribution performance."
- Split: 70/20/10 stratified random (Table 3): WLASL100+ASL-Kimpinde 7835/2239/1120; WLASL300 11085/3167/1584.
- Cross-signer partially measured via SemLex100 (independent signers, official release, 1253 curated videos / 7412 after aug) — **82.48% FP32 / 79.92% INT8**.
- Custom ASL-Kimpinde (zenodo 18754451): 250 videos, 50 classes, **single signer** — used for augmentation + calibration.

## Quantization recipe (directly reusable)
- PTQ: affine INT8, TFLITE_BUILTINS_INT8, no Flex.
- QAT: fake-quant nodes via quantize_apply(), Phase A warm-up 5 epochs, Phase B up to 50 epochs lr 1e-4, clipnorm=1.0, early stopping.
- Finding: PTQ stable on WLASL100 (k=1: 98.51±0.10, k=3: 98.60±0.97, k=5: unstable 88.15±11.47, k=7: 98.15±0.64); WLASL300 PTQ unstable (60.2–76.8% spread 16.6%), QAT nearly eliminates (74.11±5.16).
- Operator compatibility (Table 21): Conv2D/Conv1D/GAP/Dense/Softmax/ReLU all ✓ across TFLite/CMSIS-NN/NNoM. No GRU/LSTM/attention/Conv3D/BatchNorm.

## Why this matters for EchoGesture
Differences from our locked plan:
1. **We planned MediaPipe landmarks; S3D-Conv1D uses raw grayscale 64×64 frames.** This is a fundamentally different front-end.
2. But 98.7% (INT8) vs our 61.12% HRC landmark baseline — massive accuracy gap, same benchmark (WLASL100).
3. However: paper admits within-distribution; cross-signer SemLex test gives **79.92% INT8**, still above our 70% target.
4. Latency 26 ms CPU proxy on Celeron — very likely under 50 ms on Snapdragon 8 Elite.
5. Size 883 KB (INT8) — far under 20 MB.
6. **Code + trained FP32/INT8 models are on GitHub — could be a drop-in baseline, not a from-scratch build.**

## ON-DEVICE hardware numbers (NUCLEO-H753ZI, Cortex-M7 @480MHz, STM32CubeAI / ST Edge AI Core v3.0)
Extracted from the repo's `NUCLEO-H753ZI_LOGS.txt` — real board measurements, not proxies.

| Variant | Params | MACC | RAM | Latency/sample | cycles/MACC |
|---|---|---|---|---|---|
| QAT INT8, 100 classes | 892,420 (872 KiB) | 97,328,596 | 323,180 B (315.6 KiB) | **353.1 ms** | 1.74 |
| QAT INT8, Ultra-light | 17,256 (17.3 KiB) | 514,738 | 34,048 B (33.3 KiB) | **24.7 ms** | 22.99 |

Implication: the full 100-class model is **353 ms on a 480 MHz Cortex-M7** — 7× over the 50 ms budget on MCU-class hardware. The 24.7 ms ultralight fits an MCU budget but carries 74.11% WLASL300 / 98.51% WLASL100 (and its cycles/MACC of 22.99 shows it is memory-bound, not compute-bound).
On Snapdragon 8 Elite (8 cores, GHz-class, XNNPACK/Hexagon), the 97.3 MMACC full model should land roughly 5-25 ms; MediaPipe front-end is the dominant cost. **MCU numbers do not transfer to phone.** Use the phone as the target and the MCU result as the "fits on the smallest thing we tried" credibility line in the deck.

## Repo verification (GitHub API, 2026-11-07)
- Repo exists: `Samuel-KIMPINDE/S3D-Conv1D-for-SLR-for-Microcontrollers`, **MIT licensed**, 25.7 MB, last pushed 2026-04-04.
- Contains: `scripts/Python training scripts/{config.py, train_s3dconv1d.py, utils.py}`, `scripts/Kaggle Notebooks/{ptq,qat} kaggle notebook.ipynb`
- `Trained models/PTQ models/{100 Classes, 300 Classes, Ultralight variant}/` → `ptq_100_int8.tflite` (883.1 KB), `label_to_idx 100.json` (class index map), NUCLEO logs, error heatmaps
- `Trained models/QAT models/{...}` → `s3d_conv1d_qat_int8.tflite` (880.1 KB), ultralight `s3d_conv1d_qat_int8.tflite` (24.7 KB), `.keras` checkpoint
- **Weights ship with class index map — an INT8 TFLite baseline is available today, no training needed for a first number.**

## Recommended decision
ADOPT as baseline + reference implementation, NOT replace our pipeline wholesale:
- Pull GitHub repo, verify model card + license in repo
- Test trained INT8 model on held-out WLASL100 clips (signer-disjoint split) — key validation
- If signer-disjoint holds near 80%, adopt as classifier replacing TCN; keep MediaPipe as fallback / hybrid
- If signer-disjoint collapses below 65%, adopt architecture + training recipe but OUR data split and augment
- Either way: use their QAT recipe verbatim (it's exactly what the deck's quantization 20% is scored on)