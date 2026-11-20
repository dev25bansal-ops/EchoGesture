# EchoGesture — demo narration, 2:50

Written 16 Nov 2026. Every number in this script is measured and appears in
`results/results.csv` or `CLAIMS_AND_LIMITS.md`. Nothing here is a claim you cannot
defend in the Q&A.

Target 2:50, hard cap 3:00.

---

## Beat 1 — the hook (0:00–0:25)

> "Samsung Track 2: sign language prediction, fully on-device. Two minutes."
>
> *(Screen: the app on a held-up phone, or the desktop demo window.)*
>
> "One hundred ASL signs. No network. Nothing leaves the device. And I want to start
> with a number that should make you suspicious of every other team's accuracy."

## Beat 2 — the finding that sets you apart (0:25–1:05)

> "This is a published, peer-reviewed, INT8 model — 883 kilobytes, MIT licensed,
> designed for microcontrollers. It advertises 98.6 percent on a held-out test.
>
> I downloaded it and ran it myself, on signers it had never seen."
>
> *(Screen: the 1.47 % vs 1.02 % chance chart, or the results table.)*
>
> "**1.47 percent. Chance is 1.02.** It wasn't broken — I proved my harness worked
> with a 17-out-of-17 positive control on clips from signers it *had* trained on.
> The model had memorised the people, not the signs. Raw pixels carry the signer's
> clothes, room and skin tone.
>
> So I rebuilt it on hand landmarks — the representation every winning entry in
> Google's Kaggle sign-language competition independently converged on."

**This is your strongest 40 seconds.** No other team will have run the published
baseline and caught it out. It proves rigour rather than asserting it.

## Beat 3 — the result (1:05–1:30)

> "Same dataset. Signers never seen in training. Eleven of them, twelve hundred and
> fifty-one clips."
>
> *(Screen: `results.csv`, or the per-signer table.)*
>
> "**85.45 percent top-1. 96.64 top-5.** Against a 70 percent bar.
>
> Eight point one megabytes across nine models, plus a seven point five megabyte
> detector — **fifteen point six total, inside the twenty megabyte budget.** The
> classifier itself takes **four milliseconds** a clip. The hand detector is the
> real cost, and I'll be straight about that in a moment."

## Beat 4 — the honest part (1:30–1:55)

*Do this on purpose. Judged demos that volunteer their weaknesses read as credible.*

> "Three things I want to be straight about before you ask.
>
> One: the four milliseconds is the **classifier only**. The per-frame detector on
> the phone is **eighty-four to one hundred twelve milliseconds**, measured with
> the app's own profiler — six samples, two installs. That is the debug build, so
> the shipping number is better, but I could not read it off the device and I
> won't quote a number I haven't measured.
>
> Two: robustness passes on pixels — blur, low light, downscaling, JPEG, all under
> five points. But if a hand leaves the frame for a quarter of a clip, accuracy
> drops five point seven six points. That's a real limitation. The mask is working
> correctly — I measured it three ways — the information is just gone.
>
> Three: the weakest signs in the vocabulary are genuinely weak. CHEESEGRATER is
> forty percent, SINK and SCARF2 are fifty. I can tell you exactly which ones and
> why, and I'd rather do that than quote a single average and hope you don't ask."

## Beat 5 — the live moment (1:55–2:35)

*This is what wins the HUD half of the rubric. Rehearse it.*

> *(Screen: live camera, sign something.)*
>
> "Watch what it does when it's unsure."
>
> *(Sign clearly → a result card appears.)*
>
> "Read the label with me: **'Best match — still confirm before relying on it.'**
> That is deliberate. The number under it ranks the hundred known signs; it is not
> a calibrated probability that the reading is right, and the app says so.
>
> Now watch a sloppy one."
>
> *(Sign badly or partly out of frame → the card reads 'Uncertain — this is the
> model's best guess, not a confirmed reading'.)*
>
> "It tells you which of the two it is. Across the full test split, when the app
> shows a confident answer it is right **ninety-two percent** of the time — and
> it declines **fourteen percent** of captures instead of guessing at them. That
> turns the failure mode every other system hides — a confident wrong answer —
> into an honest 'not sure'."

**Say the number the app actually implements.** The app gates on a
**top-1 minus top-2 margin of 0.15**, which is 92.1% accurate at 86% coverage on
the 1,251-clip test split (`results/gate_measured`). Two other figures exist in
`CLAIMS_AND_LIMITS.md` — 95.93% at ≥0.5 confidence and 99.7% at ≥0.8 — but those
describe **confidence thresholds the app does not use**. All three are real;
quoting the wrong one while the audience watches the app is the kind of mismatch
a judge will catch.

**Also note what the UI literally says**, because you may be asked to read it
aloud:
- accepted → *"Best match — still confirm before relying on it"*
- declined → *"Uncertain — this is the model's best guess, not a confirmed reading"*
- and under the score: *"A model score ranks the 100 known signs; it is not a
  calibrated probability that the reading is correct."*

That last line is your strongest honesty card. Use it.

**Timing discipline matters more than it looks.** The capture window is a fixed
3.7 seconds ending at the tap, so tap Finish promptly once the sign ends. A
recording that dragged to 15.9 s for a sign taking about 3 s is what prompted a
long investigation; if a take goes wrong, re-record with tighter timing before
assuming the model is at fault.

**Sign directly at the phone. Do not film a screen.** During development the app
was tested by pointing the phone at a laptop playing corpus video, and it failed
— which sent a day into investigating the model, the feature transform, the
window placement and the frame rate. All three were measured and cleared. Filming
a screen puts the signer at a fraction of their real size, adds moiré, glare and
a second round of compression, and the model has never seen anything like it.
The app is fine; that input is not.

## Beat 6 — close (2:35–2:50)

> "Landmarks instead of pixels. Nine small models instead of one big one. A model
> that knows when it doesn't know.
>
> Eighty-five point four five percent on people it has never seen, fifteen point
> six megabytes, fully offline. Thank you."

---

## Q&A — the answers you need ready

| They ask | Say |
|---|---|
| "Why not INT8?" | "Tried it: −6.37 points, and two TFLite runtimes disagreed on the same file. FP16 loses zero against FP32 at half the size. The size criterion is already met." |
| "Is 85 % real, or did you tune on test?" | "The test split was inspected during development — I'll say that before you ask. It's held-out, not blind. Val is 89.31 % on a separate five signers." |
| "How does it do on *my* phone?" | "Measured on an arm64 phone now: 84–112 ms p50 per frame, **8.7–11.6** analysed frames per second, 76–82% of camera frames dropped, bottleneck the analyser. That is a debug build, so the shipping figure is better — but I'm quoting what I measured, not what I hope." |
| "What's the worst case?" | "Two honest answers. A hand leaving frame for a quarter of a clip: +5.76 points, the one criterion I fail. And the weakest glosses: CHEESEGRATER 40%, SINK and SCARF2 50%. Those are the signs a demo should avoid." |
| "Why nine models?" | "Members make different errors; averaging logits cancels part of that. The ensemble beats the best single member by 3.92 points. Adding more stops helping at nine." |
| "Is it really offline?" | "The manifest strips the INTERNET permission. There is no network call anywhere in `src/` or the app." |
| "What would you do with another week?" | "More signer-diverse data. The corpus is 17–24 clips per class; that, not the architecture, is the ceiling — I measured every other lever and all of them lost." |

## Recording notes

- Record the **desktop** demo if the phone isn't cooperating. `app/offline_demo.py`
  runs the real `predict()` on a clip file and draws the HUD. **Verified working
  2026-11-19** on two corpus clips (CHEW → CHEW1 0.277; SLIDE2 → SLIDE2 0.948).
  Run it from the repo root:

  ```
  C:/Python314/python.exe app/offline_demo.py --video "<clip>.mp4" --render-output hud.png
  ```

  Use `C:/Python314/python.exe` — it is the only interpreter here with cv2,
  mediapipe and the model runtime together. Do **not** pass `-I`.
- Have `results/results.csv` and the per-signer table open in a second window for
  Beat 3; screen-record both.
- The live-sign moment in Beat 5 is the only shot that needs a camera. If it fails,
  cut to the pre-recorded version and say so.
- Do **not** claim INT8 compliance, sub-50 ms on-device, or phone accuracy anywhere.
  Every one of those is marked unmeasured in the repo, and a judge who checks will
  find it.