"""Signer-disjoint evaluation of S3D-Conv1D (shipped INT8 TFLite) on ASL Citizen test split.

Selectively extracts only the needed .mp4 members from the ASL_Citizen.zip
using HTTP range requests on the ZIP central directory, then runs the shipped
S3D-Conv1D model with the paper's exact preprocessing.

Usage:
  py -3.12 -I eval_s3d_aslcitizen.py --workdir <dir> --records <s3d100_test_records.json>
      --splits <splits dir> --labelmap <label_to_idx 100.json> --model <ptq_100_int8.tflite>
      --limit N --workers 8
"""
import argparse, csv, json, os, re, struct, sys, time, zlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.append(r"C:\Users\dev25\AppData\Roaming\Python\Python312\site-packages")
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

URL = ("https://download.microsoft.com/download/b/8/8/"
       "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")
IMG_SIZE, NUM_FRAMES = 64, 24


def http_range(start, end, retries=4):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
            last = RuntimeError(f"short read {len(data)} != {end-start+1}")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    raise last


def extract_member(rec, out_dir):
    lho = rec["lho"]
    hdr = http_range(lho, lho + 29)
    if hdr[:4] != b"PK\x03\x04":
        raise RuntimeError("bad local header")
    (ver, flag, method, mt, md, crc, csize, usize, fnlen, extralen) = struct.unpack(
        "<HHHHHIIIHH", hdr[4:30])
    fn = http_range(lho + 30, lho + 30 + fnlen - 1).decode("utf-8", "replace")
    start = lho + 30 + fnlen + extralen
    comp = http_range(start, start + rec["csize"] - 1)
    raw = zlib.decompress(comp, -15) if method == 8 else comp
    if len(raw) != rec["usize"]:
        raise RuntimeError(f"size mismatch {len(raw)} != {rec['usize']}")
    out = os.path.join(out_dir, os.path.basename(fn))
    with open(out, "wb") as f:
        f.write(raw)
    return out


def preprocess(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(np.expand_dims(frame, -1).astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        return None
    n = len(frames)
    if n >= NUM_FRAMES:
        idx = np.linspace(0, n - 1, NUM_FRAMES).astype(int)
        arr = np.stack([frames[i] for i in idx])
    else:
        arr = np.stack(frames + [frames[-1]] * (NUM_FRAMES - n))
    return arr[None, ...]  # (1, 24, 64, 64, 1)


def quantize(x, scale, zero_point):
    q = np.round(x / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--labelmap", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    recs = json.load(open(args.records, encoding="utf-8"))
    label_to_idx = {k.upper(): v for k, v in
                    json.load(open(args.labelmap, encoding="utf-8")).items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    test_rows = list(csv.DictReader(
        open(os.path.join(args.splits, "test.csv"), encoding="utf-8", errors="replace")))
    fn2gloss = {r["Video file"]: r["Gloss"] for r in test_rows}
    fn2part = {r["Video file"]: r["Participant ID"] for r in test_rows}

    def norm(g):
        return re.sub(r"\d+$", "", g.strip().upper())

    items = []
    for r in recs:
        base = os.path.basename(r["name"])
        g = fn2gloss.get(base)
        if g is None:
            continue
        c = norm(g)
        if c not in label_to_idx:
            continue
        items.append((r, base, c, fn2part[base]))
    items.sort(key=lambda t: (t[3], t[1]))
    if args.limit:
        # stride-sample across participants so every signer is represented
        items = items[:: max(1, len(items) // args.limit)][: args.limit]

    vid_dir = os.path.join(args.workdir, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    t0 = time.time()
    paths = [None] * len(items)

    def work(i):
        p = os.path.join(vid_dir, items[i][1])
        if os.path.exists(p) and os.path.getsize(p) == items[i][0]["usize"]:
            return p
        return extract_member(items[i][0], vid_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, p in enumerate(ex.map(work, range(len(items)))):
            paths[i] = p
    print(f"fetched {len(paths)} videos in {time.time()-t0:.0f}s", flush=True)

    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    scale, zp = ind["quantization"]
    oscale, ozp = outd["quantization"]

    ok = 0
    conf = []
    per_signer = {}
    t1 = time.time()
    for i, ((rec, base, cls, part), path) in enumerate(zip(items, paths)):
        x = preprocess(path)
        if x is None:
            continue
        interp.set_tensor(ind["index"], quantize(x, scale, zp))
        interp.invoke()
        y = interp.get_tensor(outd["index"])[0].astype(np.float32)
        scores = (y - ozp) * oscale
        pred = int(np.argmax(scores))
        correct = int(idx_to_label[pred] == cls)
        ok += correct
        conf.append((base, cls, idx_to_label[pred], part, int(correct)))
        s = per_signer.setdefault(part, [0, 0])
        s[0] += correct
        s[1] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} acc={ok/(i+1):.4f}", flush=True)

    n = len(conf)
    acc = ok / n if n else float("nan")
    print(f"\n=== RESULT: top-1 = {acc*100:.2f}% ({ok}/{n}) over {len(per_signer)} signers ===")
    for p in sorted(per_signer, key=lambda x: int(x[1:])):
        c, t = per_signer[p]
        print(f"  {p}: {c}/{t} = {c/t*100:.1f}%")
    out = args.out or os.path.join(args.workdir, "s3d_aslcitizen_predictions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "true_gloss", "pred_gloss", "participant", "correct"])
        w.writerows(conf)
    json.dump({"top1": acc, "n": n, "ok": ok,
               "per_signer": {k: per_signer[k] for k in per_signer}},
              open(os.path.join(args.workdir, "s3d_aslcitizen_result.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()