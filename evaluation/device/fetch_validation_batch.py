"""Download a deterministic labelled ASL Citizen validation subset outside Git."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_smoke_clip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_csv", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clips", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--archive-size", type=int, default=45924134223)
    args = parser.parse_args(argv)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.split_csv.open(encoding="utf-8")))
    eligible = [row for row in rows if row["Gloss"] in labels]
    if not eligible or args.clips <= 0:
        parser.error("No eligible clips or invalid --clips")
    indices = {round(index * (len(eligible) - 1) / (min(args.clips, len(eligible)) - 1))
               for index in range(min(args.clips, len(eligible)))}
    selected = [eligible[index] for index in sorted(indices)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def download(row):
        output = args.output_dir / row["Video file"]
        if not output.is_file():
            fetch_smoke_clip.extract(args.archive_size, "ASL_Citizen/videos/" + row["Video file"], output)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        downloaded = list(executor.map(download, selected))
    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clip_id", "path", "ground_truth", "signer_id"])
        for row in downloaded:
            writer.writerow([Path(row["Video file"]).stem, row["Video file"], row["Gloss"], row["Participant ID"]])
    print(json.dumps({"clips": len(downloaded), "signers": sorted({row["Participant ID"] for row in downloaded}),
                      "glosses": len({row["Gloss"] for row in downloaded}), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
