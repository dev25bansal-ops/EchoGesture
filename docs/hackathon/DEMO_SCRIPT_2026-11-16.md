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
> Eight and a half megabytes across nine models, plus a 7.8 megabyte detector —
> sixteen point three total, inside the twenty megabyte budget. And on the machine
> that scores it: **fifteen milliseconds per frame** for the detector, **four
> milliseconds** for the classifier. Both inside the fifty millisecond target."

## Beat 4 — the honest part (1:30–1:55)

*Do this on purpose. Judged demos that volunteer their weaknesses read as credible.*

> "Two things I want to be straight about before you ask.
>
> One: these latency numbers are **desktop**, not phone. The emulator can't run
> MediaPipe — Google's library ships no x86-64 build — and I never got a physical
> device attached. On-device latency, power and thermal are **unmeasured**, and I
> won't estimate them.
>
> Two: robustness passes on pixels — blur, low light, downscaling, JPEG, all under
> five points. But if a hand leaves the frame for a quarter of a clip, accuracy
> drops five point seven six points. That's a real limitation. The mask is working
> correctly — I measured it three ways — the information is just gone."

## Beat 5 — the live moment (1:55–2:35)

*This is what wins the HUD half of the rubric. Rehearse it.*

> *(Screen: live camera, sign something.)*
>
> "Watch what it does when it's unsure."
>
> *(Sign clearly → confident result appears.)*
>
> "That's a high-confidence answer. Below zero point eight confidence the model is
> right ninety-nine point seven percent of the time — over 594 clips.
>
> Now watch a sloppy one."
>
> *(Sign badly or partly out of frame → the app shows 'Uncertain — sign again'.)*
>
> "Rather than guess, it asks me to repeat. If it only answers above fifty percent
> confidence, it's right **ninety-five point nine percent** of the time — on
> seventy-three percent of inputs. That turns the failure mode every other system
> hides — a confident wrong answer — into an honest 'again, please'."

## Beat 6 — close (2:35–2:50)

> "Landmarks instead of pixels. Nine small models instead of one big one. A model
> that knows when it doesn't know.
>
> Eighty-five point four five percent on people it has never seen, sixteen megabytes,
> fully offline. Thank you."

---

## Q&A — the answers you need ready

| They ask | Say |
|---|---|
| "Why not INT8?" | "Tried it: −6.37 points, and two TFLite runtimes disagreed on the same file. FP16 loses zero against FP32 at half the size. The size criterion is already met." |
| "Is 85 % real, or did you tune on test?" | "The test split was inspected during development — I'll say that before you ask. It's held-out, not blind. Val is 89.31 % on a separate five signers." |
| "How does it do on *my* phone?" | "Unmeasured. The emulator can't run MediaPipe, and no phone was attached. Everything I've shown is a desktop measurement and labelled as such." |
| "What's the worst case?" | "A hand leaving frame for a quarter of a clip: +5.76 points. That's the one criterion I fail." |
| "Why nine models?" | "Members make different errors; averaging logits cancels part of that. The ensemble beats the best single member by 3.92 points. Adding more stops helping at nine." |
| "Is it really offline?" | "The manifest strips the INTERNET permission. There is no network call anywhere in `src/` or the app." |
| "What would you do with another week?" | "More signer-diverse data. The corpus is 17–24 clips per class; that, not the architecture, is the ceiling — I measured every other lever and all of them lost." |

## Recording notes

- Record the **desktop** demo if the phone isn't cooperating — `app/offline_demo.py`
  runs the real `predict()` on a clip file and draws the HUD. It is honest and it works.
- Have `results/results.csv` and the per-signer table open in a second window for
  Beat 3; screen-record both.
- The live-sign moment in Beat 5 is the only shot that needs a camera. If it fails,
  cut to the pre-recorded version and say so.
- Do **not** claim INT8 compliance, sub-50 ms on-device, or phone accuracy anywhere.
  Every one of those is marked unmeasured in the repo, and a judge who checks will
  find it.