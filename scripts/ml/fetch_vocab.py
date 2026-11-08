"""Download the training clips for a chosen vocabulary from ASL_Citizen.zip.

Uses HTTP range requests against the ZIP central directory, so only the needed
members are fetched (~0.54 MB each) rather than the full 46 GB archive.

This is Dev's data step (DEV-01). Reads the vocabulary from
.scratch/vocab_best100.json and pulls every train/val/test clip whose gloss
appears in it.

Usage:
    python scripts/ml/fetch_vocab.py --splits train val test
"""

import argparse
import csv
import json
import os
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASLCIT = ROOT / ".scratch" / "eval_research" / "aslcit"
OUT = ROOT / ".scratch" / "aslcit_vocab_videos"
VOCAB = ROOT / ".scratch" / "vocab_best100.json"

URL = ("https://download.microsoft.com/download/b/8/8/"
       "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")


def http_range(start, end, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
            last = RuntimeError(f"short read {len(data)} != {end - start + 1}")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
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
    if not os.path.exists(out):
        with open(out, "wb") as f:
            f.write(raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    vocab = set(json.load(open(VOCAB, encoding="utf-8")))
    records = json.load(open(ASLCIT / "records.json", encoding="utf-8"))
    by_name = {os.path.basename(r["name"]): r for r in records
               if r["name"].startswith("ASL_Citizen/videos/") and r.get("lho")}
    print(f"zip members indexed: {len(by_name)}, vocab: {len(vocab)}")

    OUT.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        rows = list(csv.DictReader(open(ASLCIT / "splits" / f"{split}.csv", encoding="utf-8")))
        want = [r for r in rows if r["Gloss"] in vocab]
        todo = [r for r in want if not (OUT / r["Video file"]).exists()]
        print(f"{split}: {len(want)} in vocab, {len(todo)} to fetch", flush=True)
        if not todo:
            continue

        t0 = time.perf_counter()
        done = failed = 0

        def work(row):
            rec = by_name.get(row["Video file"])
            if rec is None:
                return False
            try:
                extract_member(rec, str(OUT))
                return True
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ok in ex.map(work, todo):
                done += 1
                if not ok:
                    failed += 1
                if done % 100 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {done}/{len(todo)} {el:.0f}s ({done/max(el,1e-9):.1f}/s) failed={failed}",
                          flush=True)

        el = time.perf_counter() - t0
        print(f"{split}: fetched {done-failed}, failed {failed} in {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()