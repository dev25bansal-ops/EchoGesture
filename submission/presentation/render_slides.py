"""Rasterise each slide so it can actually be looked at.

python-pptx gives geometry, not pixels. This draws every shape, its fill, and
its wrapped text at the right position and size, using a real font metric, so
overflow and collisions are visible. It is a layout check, not a fidelity
proof - fonts will differ from PowerPoint's - but it catches the failures that
matter: text running off the slide, boxes overlapping, rows colliding.
"""

from __future__ import annotations

import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

DPI = 110
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]


def _font(size_pt):
    px = max(6, int(size_pt * DPI / 72))
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _px(emu):
    return int(Emu(emu).inches * DPI)


def _wrap(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(pptx_path, outdir):
    prs = Presentation(pptx_path)
    W, H = _px(prs.slide_width), _px(prs.slide_height)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    problems = []

    for idx, slide in enumerate(prs.slides, 1):
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)

        for sh in slide.shapes:
            if sh.left is None:
                continue
            x, y = _px(sh.left), _px(sh.top)
            w, h = _px(sh.width), _px(sh.height)

            # fill
            try:
                if sh.fill.type is not None and sh.fill.type == 1:
                    d.rounded_rectangle([x, y, x + w, y + h], radius=int(0.04 * min(w, h)),
                                        fill=tuple(sh.fill.fore_color.rgb))
            except Exception:
                pass

            if sh.shape_type == 13:  # picture
                try:
                    from io import BytesIO
                    pic = Image.open(BytesIO(sh.image.blob)).convert("RGB")
                    pic = pic.resize((max(1, w), max(1, h)))
                    img.paste(pic, (x, y))
                except Exception:
                    d.rectangle([x, y, x + w, y + h], outline="gray")
                continue

            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue

            cy = y
            for para in sh.text_frame.paragraphs:
                runs = [r for r in para.runs if r.text]
                if not runs:
                    continue
                size = max((r.font.size.pt for r in runs if r.font.size), default=18)
                color = (0, 0, 0)
                try:
                    if runs[0].font.color and runs[0].font.color.rgb:
                        color = tuple(runs[0].font.color.rgb)
                except Exception:
                    pass
                f = _font(size)
                text = "".join(r.text for r in runs)
                for line in _wrap(d, text, f, w):
                    d.text((x, cy), line, font=f, fill=color)
                    cy += int(size * DPI / 72 * 1.28)
                cy += int(6 * DPI / 72)

            # collision / overflow report
            if cy > y + h + int(0.06 * DPI):
                problems.append(f"slide {idx}: text overflows its box by "
                                f"{cy - (y + h)}px ('{sh.text_frame.text[:40]}')")
            if x + w > W + 2 or cy > H + 2:
                problems.append(f"slide {idx}: shape runs off the slide "
                                f"({x + w}>{W} or {cy}>{H})")

        img.save(outdir / f"slide_{idx:02d}.png")

    return problems, len(prs.slides)


if __name__ == "__main__":
    pptx = sys.argv[1]
    out = sys.argv[2]
    probs, n = render(pptx, out)
    print(f"rendered {n} slides to {out}")
    if probs:
        print("LAYOUT PROBLEMS:")
        for p in probs:
            print("  ", p)
    else:
        print("no overflow/collision detected")