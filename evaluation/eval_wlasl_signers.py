import sys, os, json, collections
sys.path.append(r"C:\Users\dev25\AppData\Roaming\Python\Python312\site-packages")
import cv2, numpy as np
from ai_edge_litert.interpreter import Interpreter

base = r"E:\samsung\.scratch\eval_research"
work = os.path.join(base, "wlasl_signer_split")
lab = json.load(open(os.path.join(base, "s3d", "label100.json"), encoding="utf-8"))
idx2 = {v: k.upper() for k, v in lab.items()}
it = Interpreter(model_path=os.path.join(base, "s3d", "ptq_100_int8.tflite"))
it.allocate_tensors()
ind = it.get_input_details()[0]
outd = it.get_output_details()[0]
scale, zp = ind["quantization"]
oscale, ozp = outd["quantization"]


def is_video(p):
    try:
        with open(p, "rb") as f:
            h = f.read(12)
        return h[4:8] == b"ftyp"
    except OSError:
        return False


def prep(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, (64, 64))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(np.expand_dims(f, -1).astype(np.float32) / 255.0)
    cap.release()
    n = len(frames)
    if n == 0:
        return None
    if n >= 24:
        idx = np.linspace(0, n - 1, 24).astype(int)
        arr = np.stack([frames[i] for i in idx])
    else:
        arr = np.stack(frames + [frames[-1]] * (24 - n))
    return arr[None]


rows = [l.strip().split(",") for l in open(os.path.join(work, "downloaded.csv"),
                                           encoding="utf-8") if l.strip()]
ok = top5 = n = dead = 0
per = collections.defaultdict(lambda: [0, 0])
for p, g, sid in rows:
    ap = os.path.join(work, "videos", os.path.basename(p.replace("\\", "/")))
    if not is_video(ap):
        dead += 1
        continue
    x = prep(ap)
    if x is None:
        dead += 1
        continue
    q = np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8)
    it.set_tensor(ind["index"], q)
    it.invoke()
    y = it.get_tensor(outd["index"])[0].astype(np.float32)
    s = (y - ozp) * oscale
    o = np.argsort(-s)
    top = [idx2[i] for i in o[:5]]
    hit = top[0] == g.upper()
    ok += hit
    top5 += g.upper() in top
    n += 1
    per[sid][0] += hit
    per[sid][1] += 1
print("WLASL100 clips from the 12 largest signers (all present in training)")
print("  top1=%d/%d=%.1f%%  top5=%d/%d=%.1f%%  dead=%d" % (ok, n, 100 * ok / n, top5, n, 100 * top5 / n, dead))
for sid, (c, t) in sorted(per.items(), key=lambda kv: -kv[1][1])[:8]:
    print("  signer %s: %d/%d" % (sid, c, t))
