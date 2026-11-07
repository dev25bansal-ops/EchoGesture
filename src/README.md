# EchoGesture: Source Code & Execution Guide

Track 2: Sign Language Prediction on Edge Device  
Samsung R&D Institute India, Bangalore (XR Perception Group & PRISM)

---

## 1. Prerequisites

- Python 3.11 (tested on CPython 3.11.x 64-bit and Python 3.12/3.14)
- CPU or mobile edge accelerator

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Model Files

The inference pipeline uses the optimized INT8 quantized model:
- `models/s3d_conv1d_qat_int8.tflite` (880.1 KB, Quantization-Aware Training)
- Fallback: `models/ptq_100_int8.tflite` (883.1 KB, Post-Training Quantization)
- Vocabulary mapping: `models/label100.json` (100 word glosses)

Both models conform strictly to the **< 20 MB / < 25 MB** hard budget.

---

## 3. Running Prediction

The primary evaluation entrypoint is `main.py`.

### Python API

```python
from main import predict

result = predict("path/to/test_clip.mp4")
print(result)
```

Output format (per Slide 11 specification):
```json
{
  "clip_id": "africa_01386",
  "prediction": "AFRICA",
  "top5": ["AFRICA", "BOOK", "COMPUTER", "DRINK", "CHAIR"],
  "confidence": 0.0266,
  "latency_ms": 23.4
}
```

### CLI Execution

```bash
python main.py path/to/test_clip.mp4
```

---

## 4. Key Architectural Highlights

- **Pure integer INT8 execution** without Flex delegates.
- **Grayscale 64×64 spatial decomposition** with temporal Conv1D.
- **Zero cloud transmission** — all computation runs locally on the host CPU/NPU.
