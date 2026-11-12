"""A08 V7: failure/diagnostic analysis of recovered V6 artifacts.

Reads only V6 result JSONs (never raw videos) and derives a failure report.
Also diagnoses the systematic misbehaviours of the V6 wrappers.
"""
import json
from collections import Counter
from pathlib import Path

V6 = Path("/tmp/echogesture-v7-agents/A08/evaluation/device/v6_robustness/results")
CORRUPTIONS = ["low_light", "crop", "occlusion", "motion_blur", "frame_drop"]


def load_diagnostic(c):
    d = V6 / f"diagnostic_{c}_30" / "diagnostics.json"
    return json.load(d.open()) if d.exists() else None


def main():
    out = {"corruptions": {}}
    for c in CORRUPTIONS:
        j = load_diagnostic(c)
        recs = [r for r in j["records"] if "clean" in r]
        info = {"n": len(recs), "flips": {}}
        for m in ("fp16", "int8"):
            c2w = sum(r["clean"]["models"][m]["prediction"] == r["ground_truth"]
                      and r["corrupted"]["models"][m]["prediction"] != r["ground_truth"]
                      for r in recs)
            w2c = sum(r["clean"]["models"][m]["prediction"] != r["ground_truth"]
                      and r["corrupted"]["models"][m]["prediction"] == r["ground_truth"]
                      for r in recs)
            info["flips"][m] = {"clean_correct_corrupted_wrong": c2w,
                                "clean_wrong_corrupted_correct": w2c,
                                "clean_correct": sum(r["clean"]["models"][m]["prediction"] == r["ground_truth"] for r in recs),
                                "corrupt_correct": sum(r["corrupted"]["models"][m]["prediction"] == r["ground_truth"] for r in recs)}
        # reliability deltas
        for k in ("hand_detection_rate", "both_hand_rate", "one_hand_rate", "no_hand_rate"):
            vals = [r["corrupted"]["reliability"][k] - r["clean"]["reliability"][k] for r in recs]
            info[k + "_delta_mean"] = sum(vals) / len(vals) if vals else None
        out["corruptions"][c] = info
    # chronic clips: FP16 flips across corruptions
    per_clip = {}
    for c in CORRUPTIONS:
        j = load_diagnostic(c)
        for r in j["records"]:
            if "clean" not in r:
                continue
            cc = r["clean"]["models"]["fp16"]["prediction"] == r["ground_truth"]
            co = r["corrupted"]["models"]["fp16"]["prediction"] == r["ground_truth"]
            per_clip.setdefault(r["clip_id"], []).append((c, cc, co, r["clean"]["models"]["fp16"]["prediction"],
                                                          r["corrupted"]["models"]["fp16"]["prediction"], r["ground_truth"]))
    chronic = []
    for cid, v in sorted(per_clip.items()):
        if not v[0][1]:
            continue
        drops = [x for x in v if x[1] and not x[2]]
        if drops:
            chronic.append({"clip_id": cid, "truth": v[0][5], "clean_pred": v[0][3],
                            "n_drop_corruptions": len(drops),
                            "corrupt_preds": sorted(set(x[4] for x in drops))})
    out["chronic_clips_fp16"] = chronic
    # V6 wrapper behaviour
    out["v6_wrappers"] = {}
    for wrapper in ("guard_swap", "sticky_hand", "low_scale_velocity_0p5"):
        sub = {}
        for c in CORRUPTIONS:
            p = V6 / f"compare_{wrapper}_{c}_30" / "comparison.json"
            if not p.exists():
                continue
            j = json.load(p.open())
            if not j["baseline"]["fp16"]["clean"]["n"]:
                sub[c] = {"n": 0}
                continue
            d = {}
            for m in ("fp16", "int8"):
                base_cl = sum(r["clean"]["models"][m]["baseline"]["prediction"] == r["ground_truth"] for r in j["records"])
                cand_cl = sum(r["clean"]["models"][m]["candidate"]["prediction"] == r["ground_truth"] for r in j["records"])
                base_co = sum(r["corrupted"]["models"][m]["baseline"]["prediction"] == r["ground_truth"] for r in j["records"])
                cand_co = sum(r["corrupted"]["models"][m]["candidate"]["prediction"] == r["ground_truth"] for r in j["records"])
                d[m] = {"clean_base": base_cl, "clean_cand": cand_cl, "corrupt_base": base_co, "corrupt_cand": cand_co}
            sub[c] = d
        out["v6_wrappers"][wrapper] = sub
    outpath = Path(__file__).parent / "results" / "failure_analysis.json"
    outpath.parent.mkdir(exist_ok=True)
    outpath.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
