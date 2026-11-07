# EchoGesture — Team Onboarding & Project Master Guide

> **Welcome to the team!**  
> This document brings you 100% up to speed on everything we have researched, tested, discovered, and built so far for **Track 2 of the Samsung XR Hackathon 2026**.  
> Read this first before diving into the code.

---

## 0. Current Status (read this first)

**The submission is not ready.** Three of the six scored metrics are unmeasured or failing:

- Accuracy on unseen signers: **1.47%** (fail, target 70%)
- End-to-end latency: **117.8 ms p95** (fail, target 50 ms)
- Robustness / power / thermal: **not measured**

What does pass: model size (0.88 MB) and model inference (3.2 ms p95).
See [RESULTS.md](RESULTS.md) for every measured number and how it was produced.

---

## 1. Challenge Overview & Ground Rules

- **Hackathon:** Samsung XR Hackathon 2026 (Samsung R&D Institute India, Bangalore — XR Perception Group & PRISM)
- **Track:** **Track 02 — EchoGesture: Sign Language Prediction on Edge Device**
- **Core Mission:** Build a fully on-device, exocentric (third-person webcam/phone camera) pipeline that translates conversational sign language video into predicted text glosses or audio.
- **The Two Core Tenets:**
  1. **The Conversation Gap:** Latency must be $\le 50\text{ ms}$ (p95) so two-way dialogue remains fluid.
  2. **The On-Device Privacy Principle:** Strictly 100% local processing on device — zero raw video or bystander footage may ever leave the device.

### Crucial Dates (Do Not Miss!)
- **Registration Deadline:** **TOMORROW, 8 November 2026, 11:59 PM IST**  
  *Link:* [Google Form](https://forms.gle/oGBXsydqbhE3HS2o9)  
  *Important Note on Registration:* The form's college dropdown lists five institutions (VIT Vellore, VIT Chennai, SRM, Thapar, Manipal). If our team college is not on that list, we must immediately notify `prism@samsung.com` while registering.
- **Final Submission Deadline:** **18 November 2026, 11:59 PM IST** (Strictly no extensions).
- **Top 15 Announcement:** 23 November 2026
- **Final Live Demo to Jury:** 27 November 2026
- **Results & Awards:** 2 December 2026 (Top 4 teams win prizes worth ₹1.0 Lakh + PRISM worklet + Samsung R&D internship opportunities).

---

## 2. 100-Point Jury Evaluation Matrix

| Category | Criterion | Target | Weight | EchoGesture Current Status |
|---|---|---|---|---|
| **Execution & Efficiency (40%)** | Model Quantization | 20 MB INT8 | 20% | **PASS** - 0.88 MB measured |
| | Edge Latency | 50 ms p95 | 20% | **FAIL** - 117.8 ms p95 end-to-end measured (model alone is 3.2 ms; video decode dominates) |
| **Quality & Accuracy (60%)** | Top-1 Accuracy | 70% on held-out | 35% | **FAIL** - 1.47% signer-disjoint measured vs 1.02% chance. 100% in-distribution control confirms the harness is correct |
| | Robustness & HUD | 5 pt drop | 25% | **NOT MEASURED** - harness not written yet |

---

## 3. The 7-Round Investigation Journey: What We Found

We conducted seven rounds of deep global research using automated multi-agent workflows across arXiv, Hugging Face, Kaggle, GitHub, Qualcomm AI Hub, and OpenAlex. Here is the unvarnished timeline:

### Round 1–3: The Landscape & False Leads
- We surveyed isolated sign language benchmarks: **WLASL (2,000 words)**, **MS-ASL (1,000 words)**, and **How2Sign (80+ hours)**.
- *Crucial Realization:* How2Sign is continuous, sentence-level co-articulated signing without pre-cut word clips. Trying to segment 80h of continuous video during a 10-day sprint is a trap. WLASL-100 / WLASL-300 is the exact isolated word-level format required by the evaluator contract.
- *Search for Drop-in Models:* Zero ready-to-use WLASL INT8 TFLite models existed on Hugging Face or public GitHub repos. Models like I3D (25M params, 25 MB) or SlowFast (35M params) are 400–1800 ms per inference — completely unviable on edge CPUs.

### Round 4–5: The S3D-Conv1D Discovery
- We discovered a newly published paper: *“Efficient Word-Level Sign Language Recognition Using Quantized Spatiotemporal Deep Learning for Low-Power Microcontrollers”* (Kimpinde & Olukanmi, *Algorithms* 2026, 19(4), 248, DOI: [10.3390/a19040248](https://doi.org/10.3390/a19040248)).
- They designed **S3D-Conv1D**:
  - Factored 3D convolution: spatial 2D Conv per frame + temporal 1D Conv across time.
  - Input: 24 frames × 64×64 grayscale.
  - Reported: **98.96% float32 / 98.60% INT8 on WLASL100**, compressing from 3.5 MB to **883.1 KB INT8**.
  - All operators are natively supported in INT8 across TFLite, CMSIS-NN, and NNoM (no Flex delegates).
  - Open source on GitHub under **MIT License** with pre-trained INT8 TFLite weights!

### Round 6: Our Empirical Breakthrough — The Cross-Signer Discovery
We did not take the 98.60% number on faith. We downloaded their shipped INT8 model and benchmarked it locally against independent data:

1. **Positive Control (WLASL100 in-distribution):**
   - **Result: 17/17 = 100.0% Top-1 Accuracy.**
   - Verified that our OpenCV frame extraction, 24-frame linspace temporal sampling, and LiteRT quantization math are 100% correct.

2. **Cross-Signer Stress Test (ASL Citizen, 1,705 videos across 11 unseen signers):**
   - **Result: 1.47% Top-1 (PTQ) / 1.00% Top-1 (QAT)** vs 1.02% random chance!
   - *The Root Cause:* S3D-Conv1D processes raw 64×64 pixel frames. Because the authors used a random 70/20/10 split rather than a participant-disjoint split, the 2D convolutions partially memorized signer clothing, background, and skin tone. When tested on completely unseen signers in different rooms, accuracy collapsed.
   - *Why this matters:* Most competing teams will either overclaim 98% and fail on the jury's held-out test videos, or not know why their model breaks. We have empirically proven the domain collapse and diagnosed the root cause. **We have not yet fixed it** - that is the main open work item.

### Round 7: The Solution from Kaggle GISLR Champions
- We analyzed Google’s *Isolated Sign Language Recognition (GISLR)* competition (evaluated on unseen signers with TFLite $\le 40\text{ MB}$):
- Top teams (1st place hoyso48, 3rd place SabaiSabai, 4th place ohkawa) solved cross-signer generalization using **invariant coordinate representations**:
  1. **Root-Centering:** Subtracting wrist or mid-shoulder coordinates eliminates camera position shift.
  2. **Scale Invariance:** Dividing by shoulder width or wrist-to-MCP distance eliminates distance-to-camera bias.
  3. **Pairwise Inter-Joint Distances:** The distance between thumb tip and index tip is mathematically invariant to 3D rotation and translation.
  4. **Data Augmentation:** Random time-warp, joint masking (simulating occlusions), and synthetic noise during QAT.

---

## 4. Current Repository Architecture & Layout

Everything is pushed to our GitHub repository: **`dev25bansal-ops/EchoGesture`**

```
EchoGesture/
├── README.md                          # Master project summary & hackathon targets
├── TEAM_GUIDE.md                      # This onboarding document
├── S3D_Conv1D_analysis.md             # Complete architecture & hardware log analysis
├── S3D_signer_disjoint_evaluation.md  # Full empirical report on the 1.47% cross-signer test
├── wlasl100_asllex_vocab_justification.csv # 100-word linguistic justification (ASL-LEX 2.0)
│
├── src/                               # SUBMISSION DIRECTORY (Evaluator executes this)
│   ├── main.py                        # def predict(video_path: str) -> dict
│   ├── requirements.txt               # numpy, opencv-python, ai-edge-litert
│   └── README.md                      # Execution guide for evaluators
│
├── models/                            # MODEL ARTIFACTS (< 2 MB total)
│   ├── s3d_conv1d_qat_int8.tflite     # 880.1 KB QAT INT8 model
│   ├── ptq_100_int8.tflite            # 883.1 KB Post-Training Quantized baseline
│   └── label100.json                  # 100-class vocabulary mapping
│
├── results/                           # REQUIRED SUBMISSION RESULTS
│   ├── results.csv                    # 6-metric team results sheet
│   └── predictions.jsonl              # Evaluator prediction manifest
│
└── evaluation/                        # RESEARCH & BENCHMARK HARNESSES
    ├── eval_asl_citizen.py            # Reproducible cross-signer evaluation script
    ├── eval_wlasl_signers.py          # WLASL in-distribution validation script
    ├── asl_citizen_predictions.csv    # 1,705 clip predictions log
    ├── asl_citizen_ptq_results.json   # PTQ metrics output
    └── asl_citizen_qat_results.json   # QAT metrics output
```

---

## 5. How to Run the Code on Your Machine Right Now

### 1. Clone the Repo
```bash
git clone https://github.com/dev25bansal-ops/EchoGesture.git
cd EchoGesture
```

### 2. Set Up Python Environment
We recommend Python 3.11:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r src/requirements.txt
```

### 3. Test Prediction on a Video Clip
```bash
python src/main.py path/to/any_video.mp4
```

Expected JSON Output:
```json
{
  "clip_id": "test_sample",
  "prediction": "AFRICA",
  "top5": ["AFRICA", "BOOK", "COMPUTER", "DRINK", "CHAIR"],
  "confidence": 0.8421,
  "latency_ms": 24.35
}
```

---

## 6. Action Plan & Work Distribution (7 Nov – 18 Nov)

### Immediate Priorities (Next 24 Hours: 7–8 Nov)
1. **Confirm Phase 1 Registration:** Submit Google Form before **8 Nov 11:59 PM**. Check college name with `prism@samsung.com` if needed.
2. **Review the Handover:** Make sure you can clone the repo and run `src/main.py` on your machine.

### Work Streams for the 10-Day Sprint

| Work Stream | Owner / Focus | Key Deliverable |
|---|---|---|
| **Stream 1: Landmark Pipeline (HIGHEST PRIORITY)** | Dev / ML Lead | Replace the pixel front-end with MediaPipe landmarks using the Kaggle GISLR winning representation (118 landmarks, XY + velocity + acceleration, root-centred, scale-normalised). Train on participant-disjoint splits. This is the only approach with published evidence of clearing 70% on unseen signers. |
| **Stream 2: HUD & Demo Application** | Teammate / Frontend | Build an interactive visual HUD (OpenCV/Streamlit/Webcam demo) showing live webcam video, cyan skeleton overlay toggle, Top-5 probabilities, real-time latency counter, and `SIGNING` / `IDLE` / `DEGRADED` state indicators. |
| **Stream 3: Robustness Benchmarking** | Shared | Run the 5 paired corruption transforms (occlusion, low light, motion blur, crop, frame drop) to prove $\le 5\text{ pt}$ accuracy drop for the `results.csv` table. |
| **Stream 4: Demo Video & Pitch Deck** | Teammate / Story Lead | Prepare the 2–3 minute demo video (screen recording of HUD + picture-in-picture signer) and the 10-slide presentation deck covering our empirical findings, architecture, and edge metrics. |

---

## 7. Useful Reference Links

- **Our GitHub:** https://github.com/dev25bansal-ops/EchoGesture
- **S3D-Conv1D Paper:** https://doi.org/10.3390/a19040248
- **Kaggle GISLR 1st Place Solution:** https://www.kaggle.com/competitions/asl-signs/discussion/406612
- **Qualcomm AI Hub Snapdragon 8 Elite Benchmarks:** https://aihub.qualcomm.com/models/mediapipe_hand
- **Google Form for Registration:** https://forms.gle/oGBXsydqbhE3HS2o9
- **Samsung XR Hackathon Slides:** Stored in root directory as `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf`.
