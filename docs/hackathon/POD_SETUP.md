# Running on the Alibaba PAI-DSW GPU pod

A recipe for taking the landmark cache from Hugging Face and reproducing the
shipped accuracy on a rented A10. Verified 13 Nov 2026 on a fresh instance
(`dsw-571959`): download, load, evaluate, exact match to the reported numbers.

## What the pod gives you

| | |
| --- | --- |
| GPU | NVIDIA A10, 23.8 GB, driver 550.54.15 |
| CPU / RAM | 8 cores / 30 GB |
| Python | 3.11.11, `torch 2.9.1+cu128`, CUDA available |
| Disk | `/` has ~380 GB free; `/mnt/workspace` is a network volume that reports its size wrongly and is slow |

**Put large files on `/`, not `/mnt/workspace`.** The workspace mount is fine
for scripts and the notebook, but it is a NAS share and its `disk_usage`
reports a nonsense total.

## Get the code and the cache

The landmark cache is what makes the pod worth using: 3,390 `.npz` files,
19 MB, the exact input to training, already verified bit-for-bit against the
split CSVs. Pulling it avoids an hour of MediaPipe over the clip corpus.

```python
import os
os.environ["HF_HOME"] = "/mnt/workspace/.hf"
from huggingface_hub import snapshot_download
snapshot_download(
    "Dev2506/echogesture-full", repo_type="dataset",
    token="<your HF token>",
    allow_patterns=[
        ".scratch/landmarks/*/*.npz",
        ".scratch/vocab_best100.json",
        ".scratch/eval_research/aslcit/splits/*.csv",
        "models/*", "src/*", "scripts/*", "results/*", "docs/*",
    ],
    local_dir="/mnt/workspace/eg", max_workers=16,
)
```

That is about 35 MB and finishes in seconds. `scripts/release/hf_push_for_pod.py`
is the other side of this — the script that put it there.

## Install the runtime

```python
pip install ai-edge-litert          # runs the shipped TFLite members
pip install mediapipe==1.2.2        # only if you want to re-extract from video
```

`ai_edge_litert` is what `scripts/ml/evaluate.py` and the ensemble tools
import. TensorFlow is **not** needed to *run* the models; it is only needed to
convert new ones.

## Reproduce the shipped numbers

This is the check that the download is intact. It must print
**83.77% val / 72.78% test**; anything else means the cache or the models did
not land correctly.

```python
import os, sys, json, numpy as np
B = "/mnt/workspace/eg"
sys.path.insert(0, os.path.join(B, "src"))
from features import build_features
import ai_edge_litert.interpreter as lit

meta = json.load(open(os.path.join(B, "models/landmark_model_meta.json")))
models = []
for m in meta["members"]:
    it = lit.Interpreter(model_path=os.path.join(B, "models", m))
    it.allocate_tensors()
    models.append((it, it.get_input_details()[0]["index"],
                   it.get_output_details()[0]["index"]))

vocab = json.load(open(os.path.join(B, "models/landmark_labels.json")))
labels = [g for g, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
idx = {g: i for i, g in enumerate(labels)}

def logits(f):
    lg = []
    for it, ii, oi in models:
        it.set_tensor(ii, f[None].astype(np.float32)); it.invoke()
        lg.append(it.get_tensor(oi)[0].astype(np.float32))
    return np.mean(lg, axis=0)

for split in ("val", "test"):
    d = os.path.join(B, ".scratch/landmarks", split); ok = n = 0
    for fn in sorted(os.listdir(d)):
        z = np.load(os.path.join(d, fn), allow_pickle=False)
        if float(z["present"].max()) < 0.5:
            continue
        g = str(z["gloss"])
        if g not in idx:
            continue
        f, _ = build_features(z["landmarks"], z["present"], expected_frames=24)
        ok += int(np.argmax(logits(f)) == idx[g]); n += 1
    print(split, ok, n, round(100 * ok / n, 2))
```

Measured on the pod: `val 289 345 83.77`, `test 909 1249 72.78`.

## Training from the cache

The `.npz` files are the input; nothing needs to be decoded from video again.
Each holds `landmarks (32, 2, 21, 3)`, `present (32, 2)`, plus scalar
`gloss`, `participant` and `split`. Pass them through `src/features.py`
(`build_features`, `expected_frames=24`) exactly as the app does — that module
is the contract, and training that drifts from it produces a model the app
cannot use.

The A10 is a real GPU, so the training scripts in `scripts/ml/` run far faster
than the laptop. Latency figures measured on the pod are **not** device
figures; the competition is scored on-device, so any speed claim has to come
from the phone.

## What the pod cannot give you

- **More accuracy on its own.** The 72.78% is limited by data, not compute:
  about 17 clips per class from 32 training signers, evaluated on 11 unseen
  signers. A bigger GPU trains the same model faster; it does not add signers.
- **A device number.** Everything here is desktop-class x86 with an A10.
  On-device latency and the Android app have to be measured on the phone or
  emulator — see [ANDROID_BUILD.md](ANDROID_BUILD.md).

## Reaching the kernel from outside the UI

JupyterLab's own REST API is enough to drive the pod programmatically; the
terminal UI is not required.

```python
# create (or list) a kernel
POST /<base>/api/kernels   {"name": "python3"}
GET  /<base>/api/kernels
```

Then open `<base>/api/kernels/<id>/channels?session_id=<uuid>` as a WebSocket
and send standard Jupyter `execute_request` messages. The `<base>` is the path
segment after the host — `/dsw-571959` on the verified instance.

One caution learned the hard way: a kernel executes one cell at a time. A long
`snapshot_download` blocks every later call on that kernel, so use a second
kernel to poll while the first one works, or the calls just time out.
