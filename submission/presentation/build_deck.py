"""Build the Track 02 submission deck.

Content follows the organizer's required coverage (PDF p25): theme ID, project
title and team, problem in our own words, solution and architecture, tools and
tech stack, innovation, results and limitations.

Every number here is traceable to a measurement in the repo. Nothing is
invented. Sources are named on the slide that uses them.
"""

from __future__ import annotations

import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = pathlib.Path(__file__).resolve().parent
SHOTS = HERE / "screens"

# Palette: near-white ground, dark navy type, restrained blue, green for verified
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
BLUE = RGBColor(0x2C, 0x5C, 0xE0)
GREEN = RGBColor(0x0E, 0x8A, 0x4F)
GREY = RGBColor(0x5A, 0x6B, 0x82)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG = RGBColor(0xFA, 0xFB, 0xFD)
CARD = RGBColor(0xF1, 0xF4, 0xF9)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def bg(slide, prs):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    s.fill.solid(); s.fill.fore_color.rgb = BG; s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, size=18, color=NAVY, bold=False,
         align=PP_ALIGN.LEFT, space=6, line=1.15):
    """runs: str or list of (text, {size,color,bold})"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0; tf.margin_top = tf.margin_bottom = 0
    if isinstance(runs, str):
        runs = [(runs, {})]
    for i, (t, opt) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.space_after = Pt(opt.get("space", space))
        p.line_spacing = opt.get("line", line)
        r = p.add_run(); r.text = t
        f = r.font
        f.size = Pt(opt.get("size", size))
        f.color.rgb = opt.get("color", color)
        f.bold = opt.get("bold", bold)
        f.name = "Helvetica"
    return tb


def accent(slide, x, y, w=0.9, color=BLUE):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.055))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    s.shadow.inherit = False
    return s


def card(slide, x, y, w, h, fill=CARD):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background(); s.shadow.inherit = False
    s.adjustments[0] = 0.05
    return s


def title_slide(prs):
    s = blank(prs); bg(s, prs)
    from pptx.enum.shapes import MSO_SHAPE
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.22), prs.slide_height)
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background(); bar.shadow.inherit = False

    text(s, 0.95, 1.75, 11, 1.0, "EchoGesture", size=60, bold=True)
    accent(s, 0.98, 2.78, 1.5)
    text(s, 0.95, 3.05, 10.5, 1.2,
         "Offline, on-device recognition of isolated ASL signs", size=30, color=BLUE)
    text(s, 0.95, 3.95, 10.5, 1.6, [
        ("Track 02 — Sign Language on Edge Device", {"size": 19, "color": NAVY, "bold": True, "space": 4}),
        ("Samsung XR Hackathon 2026", {"size": 17, "color": GREY}),
    ])
    card(s, 0.95, 5.35, 7.4, 0.95)
    text(s, 1.15, 5.52, 7.0, 0.7, [
        ("Team and college: CollegeName_TeamName", {"size": 15, "color": NAVY, "bold": True, "space": 2}),
        ("Fill from the registered Google Form entry — do not guess the string.", {"size": 12, "color": GREY}),
    ], size=15)
    return s


def problem(prs):
    s = blank(prs); bg(s, prs)
    text(s, 0.95, 0.62, 11, 0.6, "The problem", size=38, bold=True)
    accent(s, 0.98, 1.30)
    text(s, 0.95, 1.62, 5.6, 0.3, "WHAT IS ACTUALLY BROKEN", size=12, color=GREY, bold=True)
    text(s, 0.95, 1.95, 5.6, 3.4, [
        ("A deaf signer in a conversation is expected to interpret in real time, for people who do not sign.",
         {"size": 17, "space": 10}),
        ("Cloud translation is the obvious answer and the wrong one: it means uploading bystanders to a server.",
         {"size": 17, "space": 10}),
        ("The constraint is therefore not accuracy alone — it is accuracy that fits on the device that is already in the room.",
         {"size": 17, "color": BLUE, "bold": True}),
    ])
    text(s, 6.95, 1.62, 5.2, 0.3, "OUR READING OF THE BRIEF", size=12, color=GREY, bold=True)
    for i, (t, sub) in enumerate([
        ("Fully on-device", "no cloud streaming, no network at inference"),
        ("Exocentric camera", "a phone watching a signer at conversational distance"),
        ("Real constraints", "≤20 MB quantised model, ≤50 ms latency budget"),
    ]):
        y = 1.98 + i * 1.18
        card(s, 6.95, y, 5.2, 0.98)
        text(s, 7.18, y + 0.15, 4.8, 0.3, t, size=17, bold=True)
        text(s, 7.18, y + 0.52, 4.8, 0.3, sub, size=13, color=GREY)
    return s


def solution(prs):
    s = blank(prs); bg(s, prs)
    text(s, 0.95, 0.62, 11, 0.6, "Our solution", size=38, bold=True)
    accent(s, 0.98, 1.30)
    text(s, 0.95, 1.62, 11, 0.9,
         "Record a complete sign, then classify the interval you recorded.",
         size=27, color=BLUE, bold=True)
    steps = [
        ("1", "Tap Start sign", "The app begins buffering timestamped hand landmarks — not photographs."),
        ("2", "Perform the sign", "One isolated sign, one or two hands, any comfortable speed."),
        ("3", "Tap Finish sign", "The recorded interval is trimmed to where motion actually is."),
        ("4", "Read the caption", "A gloss appears on screen, with alternatives and a model score."),
    ]
    for i, (n, t, sub) in enumerate(steps):
        x = 0.95 + i * 2.95
        card(s, x, 2.85, 2.65, 2.55)
        text(s, x + 0.24, 3.02, 0.5, 0.4, n, size=26, color=BLUE, bold=True)
        text(s, x + 0.24, 3.55, 2.2, 0.5, t, size=15, bold=True)
        text(s, x + 0.24, 4.10, 2.18, 1.1, sub, size=12, color=GREY, line=1.2)
    card(s, 0.95, 5.65, 11.5, 0.85, fill=RGBColor(0xE8, 0xF5, 0xEC))
    text(s, 1.18, 5.83, 11.1, 0.5,
         "Everything after the camera runs on the phone. The app has no internet permission to upload anything with.",
         size=15, color=GREEN, bold=True)
    return s


def architecture(prs):
    s = blank(prs); bg(s, prs)
    text(s, 0.95, 0.55, 11, 0.6, "Architecture", size=38, bold=True)
    accent(s, 0.98, 1.22)
    stages = [
        ("CameraX", "camera frames"),
        ("MediaPipe", "2 hands × 21 points"),
        ("Sign interval", "trimmed recording"),
        ("Features", "24 × 255 / 381"),
        ("9-model ensemble", "masked FP16"),
        ("Caption", "gloss + top-5"),
    ]
    w, gap, x0 = 1.72, 0.28, 0.95
    for i, (t, sub) in enumerate(stages):
        x = x0 + i * (w + gap)
        card(s, x, 2.35, w, 1.35, fill=WHITE if i % 2 else CARD)
        text(s, x + 0.12, 2.62, w - 0.24, 0.4, t, size=13.5, bold=True, align=PP_ALIGN.CENTER)
        text(s, x + 0.12, 3.18, w - 0.24, 0.4, sub, size=10.5, color=GREY, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            text(s, x + w + 0.02, 2.78, 0.26, 0.4, "→", size=17, color=BLUE, bold=True)
    text(s, 0.95, 1.62, 11, 0.4,
         "Camera → landmarks → a recorded interval → features → an ensemble of nine classifiers → text.",
         size=16, color=GREY)

    text(s, 0.95, 4.05, 5.4, 0.3, "THE PART THAT MATTERS", size=12, color=GREY, bold=True)
    text(s, 0.95, 4.38, 5.4, 2.3, [
        ("The classifier never sees pixels. It sees hand geometry, so it cannot memorise who was signing.",
         {"size": 15, "space": 8}),
        ("A pixel model on the same split scored 1.47% on unseen signers. The landmark path is why this works at all.",
         {"size": 15, "color": BLUE, "bold": True}),
    ])
    text(s, 6.95, 4.05, 5.3, 0.3, "SHIPPED ON DEVICE", size=12, color=GREY, bold=True)
    for i, t in enumerate([
        "CameraX · MediaPipe HandLandmarker · LiteRT",
        "Kotlin, nine TFLite interpreters",
        "Chosen at launch from the model sidecar",
    ]):
        text(s, 6.95, 4.40 + i * 0.46, 5.3, 0.4, "·  " + t, size=14, color=NAVY)
    return s


def innovation(prs):
    s = blank(prs); bg(s, prs)
    text(s, 0.95, 0.55, 11, 0.6, "Technical innovation", size=38, bold=True)
    accent(s, 0.98, 1.22)
    items = [
        ("Attention masking",
         "The model was always trained with a hand-presence mask — and every export omitted it. "
         "Supplying it made the shipped artifact finally match the trained one: +4.24 points."),
        ("Mixed feature widths",
         "Four members take x,y (255 wide); five take x,y,z (381 wide). Width is read per member "
         "from the sidecar, never guessed from a filename."),
        ("Record the interval, not a photo",
         "The user marks Start and Finish. The app classifies that interval and trims idle motion, "
         "instead of sliding a window over whatever the camera last saw."),
        ("Parity you can check",
         "Android and Python agree to 0.000000 on features and 5e-6 on logits across nine clips, "
         "both widths, five signers — including one clip both get wrong."),
    ]
    for i, (t, sub) in enumerate(items):
        x = 0.95 + (i % 2) * 5.95
        y = 1.72 + (i // 2) * 2.62
        card(s, x, y, 5.55, 2.35)
        accent(s, x + 0.26, y + 0.28, 0.55)
        text(s, x + 0.26, y + 0.50, 5.0, 0.45, t, size=19, bold=True)
        text(s, x + 0.26, y + 1.02, 5.05, 1.2, sub, size=13.5, color=GREY, line=1.22)
    return s


def results(prs):
    s = blank(prs); bg(s, prs)
    text(s, 0.95, 0.55, 11, 0.6, "Evaluation", size=38, bold=True)
    accent(s, 0.98, 1.22)
    text(s, 0.95, 1.52, 11.5, 0.4,
         "Measured on ASL Citizen, signer-disjoint. These are corpus results, not phone results.",
         size=15, color=GREY)
    stats = [("77.02%", "top-1, test", "1,249 clips, 11 unseen signers"),
             ("92.79%", "top-5, test", "same clips"),
             ("8.51 MB", "nine members", "15.57 MB with detector"),
             ("110 / 110", "Android unit tests", "0 failures")]
    for i, (big, lbl, sub) in enumerate(stats):
        x = 0.95 + i * 2.95
        card(s, x, 2.25, 2.68, 1.85)
        text(s, x + 0.22, 2.42, 2.3, 0.6, big, size=30, color=BLUE, bold=True)
        text(s, x + 0.22, 3.05, 2.3, 0.3, lbl, size=13, bold=True)
        text(s, x + 0.22, 3.38, 2.35, 0.5, sub, size=10.5, color=GREY, line=1.15)
    text(s, 0.95, 4.35, 5.6, 0.3, "READ THIS CORRECTLY", size=12, color=GREY, bold=True)
    text(s, 0.95, 4.68, 5.6, 2.0, [
        ("77.02% is a corpus figure from the model owner's re-run over the shipped files. We do not claim it as phone accuracy.",
         {"size": 13.5, "space": 7}),
        ("A separate A/B/C study covered 72 unique clips across 184 evaluations including synthetic idle. That is not 184 independent signs.",
         {"size": 13.5}),
    ], color=GREY)
    text(s, 6.95, 4.35, 5.4, 0.3, "WHERE IT STILL FALLS SHORT", size=12, color=GREY, bold=True)
    for i, (t, v, col) in enumerate([
        ("Pixel-level robustness", "FAIL — 10.0 pp worst drop", RGBColor(0xB4, 0x2A, 0x2A)),
        ("Latency ≤50 ms", "FAIL — MediaPipe dominates", RGBColor(0xB4, 0x2A, 0x2A)),
        ("INT8 quantisation", "FP16 shipped — clarification open", RGBColor(0xB4, 0x7A, 0x0A)),
    ]):
        text(s, 6.95, 4.70 + i * 0.50, 5.4, 0.4, "·  " + t, size=13.5, bold=True)
        text(s, 7.22, 5.00 + i * 0.50, 5.1, 0.35, v, size=12, color=col)
    return s


def demo(prs):
    s = blank(prs); bg(s, prs)
    text(s, 0.95, 0.55, 11, 0.6, "The app, running", size=38, bold=True)
    accent(s, 0.98, 1.22)
    text(s, 0.95, 1.50, 11.5, 0.4,
         "Captured from the shipped build on an Android 14 emulator. Real UI, real model identity.",
         size=14, color=GREY)
    shots = [("01_camera_nine_model.png", "Camera — nine-model identity in the footer"),
             ("02_glossary.png", "Glossary — the 100 supported signs"),
             ("03_history.png", "History — stored on device only")]
    for i, (f, cap) in enumerate(shots):
        p = SHOTS / f
        x = 0.95 + i * 2.95
        if p.exists():
            s.shapes.add_picture(str(p), Inches(x), Inches(2.05), height=Inches(4.15))
        text(s, x, 6.32, 2.68, 0.5, cap, size=10.5, color=GREY, line=1.15)
    text(s, 9.85, 2.05, 2.6, 0.3, "OFFLINE, VERIFIED", size=11, color=GREY, bold=True)
    for i, t in enumerate([
        "CAMERA is the only permission granted.",
        "INTERNET is explicitly removed in the manifest and absent from the built APK.",
        "No raw camera frame is written to disk or sent anywhere.",
        "Footer shows the live ensemble SHA-256.",
    ]):
        text(s, 9.85, 2.42 + i * 1.05, 2.6, 1.0, "·  " + t, size=11.5, color=NAVY, line=1.2)
    return s


def impact(prs):
    s = blank(prs); bg(s, prs)
    text(s, 0.95, 0.55, 11, 0.6, "Impact, limits and what comes next", size=34, bold=True)
    accent(s, 0.98, 1.22)
    text(s, 0.95, 1.60, 5.5, 0.3, "WHAT THIS IS", size=12, color=GREEN, bold=True)
    text(s, 0.95, 1.95, 5.5, 2.6, [
        ("A working offline recogniser for 100 isolated ASL signs, running on Android.",
         {"size": 16, "space": 8}),
        ("It runs with no network, stores nothing, and shows the model it is actually running.",
         {"size": 16, "space": 8, "color": NAVY}),
    ])
    text(s, 0.95, 4.55, 5.5, 0.3, "WHAT IT IS NOT", size=12, color=RGBColor(0xB4, 0x2A, 0x2A), bold=True)
    text(s, 0.95, 4.90, 5.5, 2.0, [
        ("Not a sign-language translator. It recognises isolated signs from a fixed 100-word vocabulary.",
         {"size": 14, "space": 6, "color": GREY}),
        ("Not validated on a physical phone. Not robust to heavy occlusion or extreme low light.",
         {"size": 14, "color": GREY}),
    ])
    text(s, 6.95, 1.60, 5.4, 0.3, "ROADMAP, IN ORDER", size=12, color=GREY, bold=True)
    for i, (n, t) in enumerate([
        ("1", "Physical-device validation across lighting, distance and signers"),
        ("2", "INT8 export that holds accuracy, to meet the stated quantisation budget"),
        ("3", "Robustness to occlusion and real camera conditions, not synthetic scaling"),
        ("4", "Continuous signing — the research problem this does not solve"),
    ]):
        y = 1.95 + i * 1.15
        text(s, 6.95, y, 0.4, 0.4, n, size=15, color=BLUE, bold=True)
        text(s, 7.35, y, 5.0, 0.9, t, size=14, color=NAVY, line=1.2)
    return s


def main():
    prs = Presentation()
    prs.slide_width = Emu(12192000)
    prs.slide_height = Emu(6858000)

    title_slide(prs)
    problem(prs)
    solution(prs)
    architecture(prs)
    innovation(prs)
    results(prs)
    demo(prs)
    impact(prs)

    out = HERE / "EchoGesture_Track02.pptx"
    prs.save(out)
    print("wrote", out, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()