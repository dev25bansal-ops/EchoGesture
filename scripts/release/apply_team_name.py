#!/usr/bin/env python
"""Apply the real CollegeName_TeamName string to the deck, PDF and filenames.

The organizer's naming rule (consolidated deck, slide 24) makes a submission
named with a placeholder a paperwork failure. Everything is already built with
the literal string "CollegeName_TeamName"; this replaces it in one move.

Usage (PowerShell, from the repo root):

    python scripts/release/apply_team_name.py "IIT_Delhi_EchoGesture"

It performs four steps, in order, and stops on the first failure:

  1. Rebuild the PPTX from the organizer's template with --team <name>.
  2. Export the PPTX to PDF with PowerPoint COM (native export, not a rasteriser).
  3. Rename both files to <name>.pptx / <name>.pdf.
  4. Update the three places a path or display string is hard-coded:
       - submission/RELEASE_MANIFEST.json  (deck/pdf entries)
       - README.md                          (deck link)
       - submission/FINAL_DELIVERY/CHECKLIST.md (deck/pdf rows)

Requirements: python with python-pptx, and Microsoft PowerPoint installed for
the PDF export. Run WITHOUT -I (that flag strips site-packages for this build).

If PowerPoint is not available, pass --no-pdf and export manually.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PRES = ROOT / "submission" / "presentation"
PLACEHOLDER = "CollegeName_TeamName"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_name(name: str) -> None:
    """The rule is CollegeName_TeamName: exactly two underscores separating
    non-empty parts, no spaces, safe as a filename."""
    if name == PLACEHOLDER:
        fail("that is still the placeholder")
    if not re.fullmatch(r"[A-Za-z0-9]+(_[A-Za-z0-9]+)+", name):
        fail(
            f"{name!r} is not a clean CollegeName_TeamName string.\n"
            "Expected letters/digits separated by underscores, no spaces, e.g.\n"
            "  IIT_Delhi_EchoGesture\n"
            "The organizer concatenates college and team, so at least one "
            "underscore is required."
        )


def build_pptx(template: pathlib.Path, name: str) -> pathlib.Path:
    out = PRES / f"{name}.pptx"
    cmd = [
        sys.executable,
        str(PRES / "build_template_deck.py"),
        "--template",
        str(template),
        "--team",
        name,
        "--out",
        str(out),
    ]
    print(f"[1/4] building {out.name}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:], file=sys.stderr)
        fail("deck build failed")
    if not out.is_file():
        fail(f"build reported success but {out} does not exist")
    print(f"      ok  {out.stat().st_size:,} B")
    return out


def export_pdf(pptx: pathlib.Path, name: str) -> pathlib.Path:
    out = PRES / f"{name}.pdf"
    print(f"[2/4] exporting {out.name} via PowerPoint COM")
    # PowerPoint's COM API: Presentations.Open, SaveAs with ppSaveAsPDF (32).
    script = f"""
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject PowerPoint.Application
try {{
    $pres = $app.Presentations.Open('{pptx}', $true, $false, $false)
    $pres.SaveAs('{out}', 32)
    $pres.Close()
}} finally {{
    $app.Quit()
}}
"""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not out.is_file():
        print(r.stdout[-1500:])
        print(r.stderr[-1500:], file=sys.stderr)
        fail(
            "PDF export failed. Export manually (File > Save As > PDF) from\n"
            f"  {pptx}\n"
            f"and save it as\n  {out}"
        )
    print(f"      ok  {out.stat().st_size:,} B")
    return out


def update_manifest(name: str) -> None:
    path = ROOT / "submission" / "RELEASE_MANIFEST.json"
    if not path.is_file():
        print("      (no RELEASE_MANIFEST.json, skipped)")
        return
    raw = path.read_text(encoding="utf-8")
    new = raw.replace(PLACEHOLDER, name)
    if new == raw:
        print("      manifest already has no placeholder")
        return
    # Validate before writing: a broken manifest is worse than a stale one.
    json.loads(new)
    path.write_text(new, encoding="utf-8")
    print(f"      manifest updated")


def update_text_refs(name: str, files: list[pathlib.Path]) -> None:
    for f in files:
        if not f.is_file():
            continue
        raw = f.read_text(encoding="utf-8")
        if PLACEHOLDER not in raw:
            continue
        f.write_text(raw.replace(PLACEHOLDER, name), encoding="utf-8")
        print(f"      {f.relative_to(ROOT)} updated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="CollegeName_TeamName, e.g. IIT_Delhi_EchoGesture")
    ap.add_argument(
        "--template",
        default=str(pathlib.Path.home() / "Downloads" / "Submission_Final_forGithub.pptx"),
        help="path to the organizer's Submission_Final_forGithub.pptx",
    )
    ap.add_argument("--no-pdf", action="store_true", help="skip the PowerPoint export")
    args = ap.parse_args()

    name = args.name.strip()
    check_name(name)

    template = pathlib.Path(args.template)
    if not template.is_file():
        fail(
            f"organizer template not found at {template}\n"
            "Pass --template <path to Submission_Final_forGithub.pptx>"
        )

    pptx = build_pptx(template, name)

    if args.no_pdf:
        print("[2/4] skipped (--no-pdf)")
    else:
        export_pdf(pptx, name)

    print("[3/4] removing the placeholder-named artifacts")
    for stale in (PRES / f"{PLACEHOLDER}.pptx", PRES / f"{PLACEHOLDER}.pdf"):
        if stale.is_file():
            stale.unlink()
            print(f"      removed {stale.name}")

    print("[4/4] updating references")
    update_manifest(name)
    update_text_refs(
        name,
        [
            ROOT / "README.md",
            ROOT / "submission" / "FINAL_DELIVERY" / "CHECKLIST.md",
        ],
    )

    print(
        f"\nDone. The deck is {pptx.name}"
        + ("" if args.no_pdf else f" with {name}.pdf beside it.")
        + "\nCheck the deck renders (slide 1 and 9 carry the team string), then commit."
    )


if __name__ == "__main__":
    main()
