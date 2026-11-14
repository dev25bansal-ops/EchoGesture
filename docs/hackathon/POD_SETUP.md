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
**87.25% val / 77.02% test**; anything else means the cache or the models did
not land correctly.

> If you are reading this before 13 Nov 2026: the number used to be
> 83.77% / 72.78%. The models changed — they now take a hand-presence mask,
> and five of the nine take 381-wide x,y,z features. The script below handles
> both. Anything *other* than one of those two pairs means the download is
> wrong.

```python
import os, sys, json, numpy as np
B = "/mnt/workspace/eg"
sys.path.insert(0, os.path.join(B, "src"))
from features import build_features
import ai_edge_litert.interpreter as lit

meta = json.load(open(os.path.join(B, "models/landmark_model_meta.json")))
detail = {d["file"]: d["use_z"] for d in meta.get("members_detail", [])}

models = []
for m in meta["members"]:
    it = lit.Interpreter(model_path=os.path.join(B, "models", m))
    it.allocate_tensors()
    # Match inputs by NAME. The converter reports the mask as input 0, so
    # positional indexing feeds a (1,24) mask where (1,24,F) features belong.
    dets = {d["name"]: d for d in it.get_input_details()}
    f_det = next(d for n, d in dets.items() if "features" in n)
    m_det = next((d for n, d in dets.items() if "mask" in n), None)
    models.append((it, f_det["index"], m_det["index"] if m_det else -1,
                   it.get_output_details()[0]["index"], detail.get(m, False)))

vocab = json.load(open(os.path.join(B, "models/landmark_labels.json")))
labels = [g for g, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
idx = {g: i for i, g in enumerate(labels)}

def logits(feats, mask):
    lg = []
    for it, fi, mi, oi, _uz in models:
        it.set_tensor(fi, feats[None].astype(np.float32))
        if mi >= 0:
            it.set_tensor(mi, mask[None].astype(np.float32))
        it.invoke()
        lg.append(it.get_tensor(oi)[0].astype(np.float32))
    return np.mean(lg, axis=0)

for split in ("val", "test"):
    d = os.path.join(B, ".scratch/landmarks", split)
    ok = n = 0
    for fn in sorted(os.listdir(d)):
        z = np.load(os.path.join(d, fn), allow_pickle=False)
        if float(z["present"].max()) < 0.5:
            continue
        g = str(z["gloss"])
        if g not in idx:
            continue
        # Build each width once; a member's own graph declares which it needs.
        W = {uz: build_features(z["landmarks"], z["present"],
                                expected_frames=24, use_z=uz)
             for uz in {uz for *_x, uz in models}}
        per_member = []
        for it, fi, mi, oi, uz in models:
            f, m = W[uz]
            it.set_tensor(fi, f[None].astype(np.float32))
            if mi >= 0:
                it.set_tensor(mi, m[None].astype(np.float32))
            it.invoke()
            per_member.append(it.get_tensor(oi)[0])
        ok += int(np.argmax(np.mean(per_member, axis=0)) == idx[g])
        n += 1
    print(split, ok, n, round(100 * ok / n, 2))
```

Measured on the pod: `val 301 345 87.25`, `test 962 1249 77.02`.

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

- **More accuracy on its own.** The 77.02% is limited by data, not compute:
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
