"""Build the Track 02 submission deck INSIDE the organizer's template.

The organizer ships `Submission_Final_forGithub.pptx` (9 slides, 10 x 5.625 in)
and slide 24 of the consolidated deck makes non-conformance a direct
disqualification. So this builds onto that file rather than beside it: the
template is opened, its slides are filled in place, and the result keeps the
template's own palette, fonts and header band.

Template palette, read out of ppt/theme/theme1.xml and the slide XML:
    0D1B3E navy      the header band
    00B4D8 cyan      the rule under it, and the accent
    1565C0 blue      secondary accent
    0A0E1A near-black body text
    E8EDF5 pale      card fill
    Arial Black display / Calibri body

Every number here is traceable to a measurement in the repo. Nothing is
invented. Sources are named on the slide that uses them.

Usage:
    python submission/presentation/build_template_deck.py \
        --template  <path to Submission_Final_forGithub.pptx> \
        --team      "CollegeName_TeamName" \
        --out       submission/presentation/CollegeName_TeamName.pptx

Run it twice for the final artifact: once with the real team string.
"""

from __future__ import annotations

import argparse
import copy
import pathlib

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# The organizer's body slides are DARK: a 0D1B3E header band over a 0A0E1A
# ground. All body text must therefore be light. INK is for text that sits on
# a PALE card; BODY_* is for text directly on the slide ground.
NAVY = RGBColor(0x0D, 0x1B, 0x3E)
CYAN = RGBColor(0x00, 0xB4, 0xD8)
BLUE = RGBColor(0x15, 0x65, 0xC0)
LIGHT_BLUE = RGBColor(0x5B, 0x9B, 0xD5)
INK = RGBColor(0x0A, 0x0E, 0x1A)         # only on PALE / WHITE cards
PALE = RGBColor(0xE8, 0xED, 0xF5)
CARD_DARK = RGBColor(0x16, 0x25, 0x45)   # card fill on the dark ground
CARD_EDGE = RGBColor(0x24, 0x38, 0x5E)
GREY = RGBColor(0x8A, 0x9A, 0xB0)        # muted, readable on dark
BODY_TEXT = RGBColor(0xE8, 0xED, 0xF5)   # primary text on the dark ground
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xE8, 0x5D, 0x5D)         # lifted for dark ground
AMBER = RGBColor(0xF0, 0xA8, 0x3C)
GREEN = RGBColor(0x4A, 0xC8, 0x80)

DISPLAY = "Arial Black"
BODY = "Calibri"

# Team members as given by the team lead on 2026-11-22. These are the full
# names, which is what the first slide a judge sees should carry.
MEMBERS = ["Dev Datya Pratap Bansal", "Miskin Abhijeet Laxman"]

# The template is 10 x 5.625 in. The header band occupies y=0..0.90, so every
# content slide gets the rectangle below that and nothing above it.
TOP = 1.00
LEFT = 0.40
WIDTH = 9.20


def _solid(shape, colour):
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    shape.shadow.inherit = False


def _runs(p, spec, default_size=13, default_colour=BODY_TEXT, default_font=BODY):
    """spec: str, or list of (text, {size,colour,bold,font,space,align})."""
    if isinstance(spec, str):
        spec = [(spec, {})]
    for t, opt in spec:
        r = p.add_run()
        r.text = t
        f = r.font
        f.size = Pt(opt.get("size", default_size))
        f.color.rgb = opt.get("colour", opt.get("color", default_colour))
        f.bold = opt.get("bold", False)
        f.name = opt.get("font", default_font)


def block(slide, x, y, w, h, paras, size=13, colour=BODY_TEXT, font=BODY,
          space=4, line=1.12, align=PP_ALIGN.LEFT):
    """A text block. `paras` is a list of str or run-lists."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    if isinstance(paras, str):
        paras = [paras]
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        p.line_spacing = line
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], dict):
            spec, meta = spec
            p.alignment = meta.get("align", align)
            p.space_after = Pt(meta.get("space", space))
            p.line_spacing = meta.get("line", line)
            _runs(p, spec, size, colour, font)
        else:
            _runs(p, spec, size, colour, font)
    return tb


def card(slide, x, y, w, h, fill=CARD_DARK):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(h))
    _solid(s, fill)
    s.adjustments[0] = 0.06
    return s


def rule(slide, x, y, w=0.70, colour=CYAN, thick=0.045):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(thick))
    _solid(s, colour)
    return s


def find(slide, fragment):
    """Find the first shape whose text contains `fragment` (case-insensitive)."""
    for sh in slide.shapes:
        if sh.has_text_frame and fragment.lower() in sh.text_frame.text.lower():
            return sh
    raise KeyError(f"no shape containing {fragment!r} on this slide")


def retitle(slide, new_text):
    """Replace a section header in place, keeping the template's styling."""
    sh = find(slide, "")
    for cand in slide.shapes:
        if cand.has_text_frame and cand.text_frame.text.strip() and cand.top < Inches(0.9):
            sh = cand
            break
    tf = sh.text_frame
    p = tf.paragraphs[0]
    if p.runs:
        p.runs[0].text = new_text
        for extra in p.runs[1:]:
            extra.text = ""
    else:
        p.add_run().text = new_text
    return sh


def clear_body(slide):
    """Remove everything that is not the header band / rule / title."""
    for sh in list(slide.shapes):
        if sh.top is not None and sh.top >= Inches(0.88):
            sh._element.getparent().remove(sh._element)


# --- slide builds -----------------------------------------------------------

def s1_title(slide, team: str):
    """Track No / project title / team + college / members."""
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        txt = sh.text_frame.text.strip()
        if txt == "Track No":
            tf = sh.text_frame
            tf.paragraphs[0].runs[0].text = "TRACK 02"
            r = tf.paragraphs[0].runs[0].font
            r.size = Pt(13)
            r.bold = True
            r.name = BODY
            r.color.rgb = CYAN
        elif txt.startswith("XR Hackathon"):
            tf = sh.text_frame
            p = tf.paragraphs[0]
            p.runs[0].text = "EchoGesture"
            p.runs[0].font.size = Pt(44)
            p.runs[0].font.name = DISPLAY
            p.runs[0].font.color.rgb = WHITE
            p2 = tf.add_paragraph()
            p2.space_before = Pt(6)
            r2 = p2.add_run()
            r2.text = "Sign Language Prediction on Edge Device"
            r2.font.size = Pt(22)
            r2.font.name = BODY
            r2.font.color.rgb = CYAN
        elif txt.startswith("Team Name"):
            tf = sh.text_frame
            tf.word_wrap = True
            # The template's box is 7.50 x 1.00 in and it holds TWO lines
            # (team on one, college on the next). Squeezing a single
            # "College_Team" string into a 18 pt run overflowed it and clipped
            # to "College Name". Split on the underscore the naming rule
            # implies, and shrink the type so a long pair still fits.
            label = team.replace("_", " · ").strip() or team
            p = tf.paragraphs[0]
            p.runs[0].text = label
            # The template's box holds two lines (team, then college). Our
            # single string already carries both, so blank the second one
            # rather than leaving the template's literal "College Name".
            for extra in list(tf.paragraphs[1:]):
                for r in extra.runs:
                    r.text = ""
            p.runs[0].font.size = Pt(15)
            p.runs[0].font.name = BODY
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = WHITE
        elif txt == "Members Details":
            tf = sh.text_frame
            # The template ships the label "Members Details" where the actual
            # names belong. Leaving it reads as an unfilled form on the first
            # slide a judge sees. Names come from the repo's own authorship.
            p = tf.paragraphs[0]
            p.runs[0].text = "Members"
            for r in p.runs:
                r.font.size = Pt(9)
                r.font.color.rgb = GREY
                r.font.name = BODY
            for name in MEMBERS:
                para = tf.add_paragraph()
                r = para.add_run()
                r.text = name
                r.font.size = Pt(11)
                r.font.bold = True
                r.font.color.rgb = WHITE
                r.font.name = BODY


def s2_problem(slide):
    retitle(slide, "THE PROBLEM")
    block(slide, LEFT, TOP, 4.30, 3.9, [
        ("A deaf signer in a conversation is expected to interpret in real time, for "
         "people who do not sign.", {"size": 14, "space": 10}),
        ("Cloud translation is the obvious answer and the wrong one: it means uploading "
         "bystanders to a server.", {"size": 14, "space": 10}),
        ("So the constraint is not accuracy alone — it is accuracy that fits on the device "
         "already in the room.", {"size": 14, "bold": True, "colour": CYAN, "space": 10}),
    ])
    block(slide, 5.10, TOP, 4.40, 0.3, "WHAT THE BRIEF ASKS FOR", size=9, colour=GREY)
    for i, (t, sub, note) in enumerate([
        ("Fully on-device", "no cloud streaming at inference", True),
        ("Exocentric, third-person", "a camera watching a signer at conversation distance", True),
        ("Hard budgets", "under 50 ms latency, under 20 MB of model", True),
    ]):
        y = TOP + 0.34 + i * 1.14
        card(slide, 5.10, y, 4.40, 0.98)
        block(slide, 5.26, y + 0.11, 4.1, 0.3, t, size=13, font=BODY, colour=BODY_TEXT)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, 5.26, y + 0.44, 4.1, 0.5, sub, size=10.5, colour=GREY)


def s3_solution(slide):
    retitle(slide, "SOLUTION")
    block(slide, LEFT, TOP - 0.02, 9.2, 0.55,
          "Record a complete sign, then classify the interval you recorded.",
          size=19, colour=CYAN, font=BODY)
    slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
    steps = [
        ("1", "Tap Start", "One button begins it. The app buffers timestamped hand landmarks — not photographs."),
        ("2", "Sign it", "One isolated sign, one or two hands, any speed."),
        ("3", "No Finish tap", "The app finds the sign's own start and end — you never tap a second time."),
        ("4", "Read the caption", "A gloss appears, with alternatives and a model score."),
    ]
    for i, (n, t, sub) in enumerate(steps):
        x = LEFT + i * 2.30
        card(slide, x, TOP + 0.62, 2.10, 2.15)
        block(slide, x + 0.16, TOP + 0.72, 0.5, 0.35, n, size=20, colour=CYAN)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, x + 0.16, TOP + 1.10, 1.80, 0.30, t, size=11.5, colour=BODY_TEXT)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, x + 0.16, TOP + 1.44, 1.80, 1.20, sub, size=9.5, colour=GREY, line=1.18)
    card(slide, LEFT, TOP + 2.95, 9.20, 0.72, fill=RGBColor(0x10, 0x33, 0x3F))
    block(slide, LEFT + 0.18, TOP + 3.10, 8.9, 0.5,
          "Everything after the camera runs on the phone. The app has no INTERNET permission to upload anything with.",
          size=11.5, colour=CYAN)


def s4_architecture(slide):
    retitle(slide, "ARCHITECTURE")
    block(slide, LEFT, TOP - 0.04, 9.2, 0.35,
          "Camera → landmarks → a recorded interval → features → nine classifiers → text.",
          size=11, colour=GREY)
    stages = [
        ("CameraX", "camera frames"),
        ("MediaPipe", "2 hands × 21 points"),
        ("Sign interval", "trimmed recording"),
        ("Features", "24 × 255 / 381"),
        ("9-model ensemble", "masked FP16"),
        ("Caption", "gloss + top-5"),
    ]
    w, gap = 1.36, 0.24
    for i, (t, sub) in enumerate(stages):
        x = LEFT + i * (w + gap)
        card(slide, x, TOP + 0.44, w, 1.05, fill=CARD_EDGE if i % 2 else CARD_DARK)
        block(slide, x + 0.08, TOP + 0.62, w - 0.16, 0.28, t, size=9.5, colour=BODY_TEXT,
              align=PP_ALIGN.CENTER)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, x + 0.08, TOP + 0.98, w - 0.16, 0.40, sub, size=8, colour=GREY,
              align=PP_ALIGN.CENTER, line=1.05)
        if i < len(stages) - 1:
            block(slide, x + w + 0.02, TOP + 0.72, 0.20, 0.25, "→", size=13, colour=CYAN)
    block(slide, LEFT, TOP + 1.74, 4.45, 0.25, "THE PART THAT MATTERS", size=9, colour=GREY)
    block(slide, LEFT, TOP + 2.00, 4.45, 1.4, [
        ("The classifier never sees pixels. It sees hand geometry, so it cannot memorise who was signing.",
         {"size": 10.5, "space": 6}),
        ("A pixel model on the same split scored 1.47%. The landmark path is why this works at all.",
         {"size": 10.5, "bold": True, "colour": CYAN}),
    ])
    block(slide, 5.10, TOP + 1.74, 4.40, 0.25, "SHIPPED ON DEVICE", size=9, colour=GREY)
    block(slide, 5.10, TOP + 2.00, 4.40, 1.3, [
        ("·  CameraX · MediaPipe HandLandmarker · LiteRT", {"size": 10.5, "space": 5}),
        ("·  Kotlin, nine TFLite interpreters", {"size": 10.5, "space": 5}),
        ("·  Chosen at launch from the model sidecar", {"size": 10.5, "space": 5}),
    ], colour=BODY_TEXT)


def s5_output(slide):
    """'Track Final Output' — the results and the results-sheet verdicts."""
    retitle(slide, "TRACK FINAL OUTPUT")
    block(slide, LEFT, TOP - 0.04, 9.2, 0.30,
          "ASL Citizen, signer-disjoint. Corpus results, not phone accuracy.",
          size=10, colour=GREY)
    stats = [
        ("85.45%", "top-1, test", "1,251 clips, 11 unseen signers"),
        ("96.64%", "top-5, test", "same clips"),
        ("8.51 MB", "nine members", "16.33 MB with detector"),
        ("150 / 150", "Android unit tests", "0 failures"),
    ]
    for i, (big, lbl, sub) in enumerate(stats):
        x = LEFT + i * 2.30
        card(slide, x, TOP + 0.34, 2.10, 1.16)
        block(slide, x + 0.14, TOP + 0.44, 1.85, 0.40, big, size=20, colour=CYAN)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, x + 0.14, TOP + 0.86, 1.85, 0.22, lbl, size=9, colour=BODY_TEXT)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, x + 0.14, TOP + 1.08, 1.85, 0.36, sub, size=8, colour=GREY, line=1.05)

    block(slide, LEFT, TOP + 1.68, 4.45, 0.25, "RESULTS SHEET — WHERE IT STANDS", size=9, colour=GREY)
    rows = [
        ("top1_accuracy", "85.45%", "PASS", GREEN),
        ("latency_p95", "72–100 ms, live camera", "FAIL", AMBER),
        ("model_size", "16.33 MB", "PASS", GREEN),
        ("accuracy_drop_robustness", "5.76 pt", "FAIL", RED),
        ("avg_power", "2.7–2.9 W (camera path)", "FAIL", RED),
        ("thermal_drift_10min", "19.2%", "PASS", GREEN),
    ]
    for i, (m, v, verdict, col) in enumerate(rows):
        y = TOP + 1.94 + i * 0.42
        block(slide, LEFT, y, 2.05, 0.22, m, size=8.5, colour=BODY_TEXT, font=BODY)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.name = "Consolas"
        block(slide, LEFT + 2.05, y, 1.80, 0.22, v, size=8.5, colour=GREY)
        block(slide, LEFT + 3.80, y, 0.65, 0.22, verdict, size=8.5, colour=col)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True

    block(slide, 5.10, TOP + 1.68, 4.40, 0.25, "WHAT THE NUMBERS DO NOT SAY", size=9, colour=GREY)
    block(slide, 5.10, TOP + 1.94, 4.40, 2.0, [
        ("85.45% is a corpus figure at native 640×480 — the geometry the app feeds. Not phone accuracy.",
         {"size": 9, "space": 5}),
        ("The 5.76 pt robustness drop is real; the app's gate recovers it to 2.39 at 86%/80% coverage.",
         {"size": 9, "space": 5}),
        ("Latency is measured on the SHIPPING Auto 6.2 APK, live camera with a hand in frame, 19 sessions: detect p95 72–100 ms (median 85), all 19 over the 50 ms target. The analyser, not the camera, is the limit — it processed 40% of delivered frames. Power still came from a DEBUGGABLE build and remains an upper bound.",
         {"size": 9, "space": 5}),
        ("The old 77.02% was these weights on a cache with a wrong presence mask on 29% of clips.",
         {"size": 9, "colour": CYAN}),
    ], colour=GREY)


def s6_differentiation(slide):
    retitle(slide, "DIFFERENTIATION")
    items = [
        ("A HUD that says when it is unsure",
         "The score chip is colour-coded and the caption is gated: a margin rule calibrated "
         "on the validation split accepts 86.1% of captures at 92.1% accuracy on the "
         "held-out test split and declines the 13.9% that score 43.3%. Capture facts and "
         "inference time are separate lines, so a slow analyser cannot be misread as a slow model."),
        ("Attention masking",
         "The model was always trained with a hand-presence mask — and every export omitted it. "
         "Supplying it made the shipped artifact match the trained one: +4.24 points."),
        ("Record the interval, not a photo",
         "One tap. The app classifies the recorded interval and trims it to where the motion "
         "actually is, instead of sliding a window over whatever the camera last saw."),
        ("Parity you can check",
         "Nine members at two widths (255 x,y / 381 x,y,z, read per member from the sidecar). "
         "Android and Python agree to 0.000000 on features and 5e-6 on logits across nine clips, "
         "both widths, five signers — including one clip both get wrong."),
    ]
    for i, (t, sub) in enumerate(items):
        x = LEFT + (i % 2) * 4.65
        y = TOP + (i // 2) * 1.72
        card(slide, x, y, 4.55, 1.55)
        rule(slide, x + 0.20, y + 0.20, 0.45)
        block(slide, x + 0.20, y + 0.34, 4.1, 0.28, t, size=12, colour=BODY_TEXT)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, x + 0.20, y + 0.66, 4.1, 0.85, sub, size=9.5, colour=GREY, line=1.16)


def s7_demo(slide, shots_dir: pathlib.Path):
    retitle(slide, "DEMO & LOCAL SETUP")
    shots = [("01_camera_nine_model.png", "Settings — nine-model identity and ensemble SHA-256"),
             ("02_glossary.png", "Glossary — 100 of 100 supported signs"),
             ("03_history.png", "History — stored on this device only")]
    for i, (f, cap) in enumerate(shots):
        p = shots_dir / f
        x = LEFT + i * 2.05
        if p.exists():
            slide.shapes.add_picture(str(p), Inches(x), Inches(TOP + 0.10), height=Inches(3.05))
        block(slide, x, TOP + 3.22, 1.95, 0.45, cap, size=8, colour=GREY, line=1.12)

    block(slide, 6.85, TOP + 0.06, 2.70, 0.25, "RUN IT YOURSELF", size=9, colour=GREY)
    block(slide, 6.85, TOP + 0.32, 2.70, 2.6, [
        ("docker build -t echogesture .", {"size": 8.5, "font": "Consolas", "space": 3}),
        ('docker run --rm -v "$PWD/clips:/clips" echogesture /clips/sign.mp4',
         {"size": 8.5, "font": "Consolas", "space": 8}),
        ("python src/main.py path/to/clip.mp4", {"size": 8.5, "font": "Consolas", "space": 8}),
        ("Python 3.11 · requirements.txt pinned · model init is verified at image build time.",
         {"size": 8.5, "colour": GREY, "space": 8}),
        ("DEMO VIDEO: <paste the 2–3 min YouTube or Drive link here>",
         {"size": 8.5, "bold": True, "colour": RED}),
    ], colour=CYAN)


def s8_brownie(slide):
    retitle(slide, "WHAT THIS IS NOT")
    block(slide, LEFT, TOP - 0.02, 4.45, 0.25, "LIMITS, STATED NOT BURIED", size=9, colour=GREY)
    block(slide, LEFT, TOP + 0.26, 4.45, 3.6, [
        ("Not a sign-language translator. It recognises isolated signs from a fixed 100-word vocabulary.",
         {"size": 10, "space": 6, "colour": GREY}),
        ("Validated on one physical phone. Not robust to heavy occlusion or extreme low light.",
         {"size": 10, "space": 6, "colour": GREY}),
        ("Live camera is the weak path. The model is rotation-variant and the corpus is landscape "
         "third-person, so sign into the rear camera held in landscape, at conversation distance. "
         "For anything that must be deterministic, the file-pick path is still there.",
         {"size": 9, "colour": GREY}),
    ])
    block(slide, 5.10, TOP - 0.02, 4.40, 0.25, "ROADMAP, IN ORDER", size=9, colour=GREY)
    for i, (n, t) in enumerate([
        ("1", "Make the live camera match the corpus: landscape capture, or retrain with rotation "
              "and distance augmentation"),
        ("2", "INT8 export that holds accuracy, to meet the stated quantisation budget"),
        ("3", "Robustness to occlusion and real camera conditions, not synthetic scaling"),
        ("4", "Continuous signing — the research problem this does not solve"),
    ]):
        y = TOP + 0.26 + i * 0.86
        block(slide, 5.10, y, 0.25, 0.25, n, size=10, colour=CYAN)
        slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        block(slide, 5.38, y, 4.10, 0.75, t, size=9.5, colour=BODY_TEXT, line=1.14)


def s9_thanks(slide, team: str):
    for sh in slide.shapes:
        if sh.has_text_frame and "thank you" in sh.text_frame.text.lower():
            tf = sh.text_frame
            tf.paragraphs[0].runs[0].text = "Thank You"
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            r = p2.add_run()
            r.text = team
            r.font.size = Pt(12)
            r.font.name = BODY
            r.font.color.rgb = GREY


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", required=True, type=pathlib.Path)
    ap.add_argument("--team", default="CollegeName_TeamName")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--shots", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent / "screens")
    a = ap.parse_args()

    from pptx import Presentation
    prs = Presentation(str(a.template))
    if len(prs.slides) != 9:
        raise SystemExit(f"expected the 9-slide organizer template, got {len(prs.slides)}")

    s = prs.slides
    s1_title(s[0], a.team)
    s2_problem(s[1])
    s3_solution(s[2])
    s4_architecture(s[3])
    s5_output(s[4])
    s6_differentiation(s[5])
    s7_demo(s[6], a.shots)
    s8_brownie(s[7])
    s9_thanks(s[8], a.team)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(a.out))
    print(f"wrote {a.out}  slides={len(prs.slides)}  size={prs.slide_width/914400:.2f}x{prs.slide_height/914400:.2f}in")


if __name__ == "__main__":
    main()
