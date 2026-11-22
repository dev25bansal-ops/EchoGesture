"""Generate the app's 100 fixed sign pronunciations from CMU Flite."""
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "models" / "landmark_labels.json"
DEST = ROOT / "app/android_final/app/src/main/assets/pronunciations"
PHRASES = {
    "8HOUR": "eight hours",
    "ALLOFSUDDEN": "all of a sudden",
    "BASEBALLCAP": "baseball cap",
    "CHEESEGRATER": "cheese grater",
    "GOODYGOODYSHOEOPPOSITE": "opposite of goody two shoes",
    "HURDLE/TRIP": "hurdle or trip",
    "LONGLINE": "long line",
    "MEACULPA": "mea culpa",
    "PULLCONVINCE": "pull or convince",
    "ROCKINGCHAIR": "rocking chair",
    "SCROLLDOWN": "scroll down",
    "VLOG": "video blog",
    "WALK-TIGHTROPE-CL": "walk a tightrope",
    "WHATFOR": "what for",
}
DEST.mkdir(parents=True, exist_ok=True)
labels = json.loads(LABELS.read_text())
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    pronunciations = {}
    for label, index in labels.items():
        stem = re.sub(r"[0-9]+$", "", label)
        pronunciations[index] = PHRASES.get(stem, stem.lower())
    for index, text in pronunciations.items():
        subprocess.run(
            ["flite", "-voice", "slt", "-t", text, "-o", str(root / f"{index}.wav")],
            check=True,
        )
    for index in pronunciations:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(root / f"{index}.wav"),
             "-c:a", "libmp3lame", "-b:a", "32k", str(DEST / f"{index}.mp3")],
            check=True,
        )
