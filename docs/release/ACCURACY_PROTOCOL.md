# Real-phone recognition accuracy protocol

Repeatable measurement of top-1, top-5 and rejection on a physical Android
phone. Companion to [RC_PHONE_ACCEPTANCE.md](RC_PHONE_ACCEPTANCE.md), which
covers functional acceptance; this covers **accuracy**.

Status: **NOT EXECUTED.** No physical device has been attached to the machine
this was written on. Nothing below is a measurement.

## What this protocol is for

The app already records, per capture, the frame count, the real capture
duration, the achieved FPS, the hand-bearing frame count, the presence mask,
the predicted label, its score, the top-5 and the inference time
(`files/diagnostics.txt`). What it deliberately does **not** record is the sign
you *intended*. That is the one thing a test run must supply, because accuracy
is only computable against ground truth.

Scores are a ranking signal, not a probability of correctness. A 0.9 score on a
wrong label is still wrong, and a 0.2 score on the right label is still right.
Only the join of prediction with intent measures accuracy.

## Session setup

1. Install the candidate APK (recorded in `evaluation/device/rc_unified/RESULT.md`).
2. Settings → **Capture diagnostics** on. This deletes any previous log.
3. Settings → camera: use the same lens for the whole session and note it.
4. Perform each sign from an **authoritative demonstration** — the ASL Citizen
   clip for that gloss, or a fluent signer. Do not improvise a sign from a
   written gloss; a wrong or non-canonical performance measures nothing.

## Sampling plan (minimum)

| Factor | Minimum |
| --- | --- |
| Supported glosses | 10, drawn from the 100 shipped labels |
| Attempts per gloss | 5 |
| Signers | 1 (more if available; note each separately) |
| Backgrounds | 2 (e.g. plain wall, cluttered room) |
| Lighting | 2 (e.g. daylight, indoor lamp) |
| Distance | 2 (e.g. ~60 cm, ~120 cm) |

Vary the last three across attempts rather than making a separate grid, but
record which condition each attempt used. 50 captures minimum; 150 if three
signers are available.

Choose glosses that are visually distinct **and** at least one known-confusable
pair, so the confusion matrix has something to say.

## Per-attempt record (annotations file)

While capturing, keep a plain text file alongside. One line per capture, in the
same order the captures were made:

```
<attempt#> <expected_label> <background> <lighting> <distance_cm>
```

Example:

```
1 EAT1 plain daylight 60
2 EAT1 cluttered lamp 120
```

This file contains no images and no personal data. Keep it out of Git.

## Collection

1. Keep both hands inside the frame guide for the whole capture. The status
   line counts the 3.7 s window down.
2. Tap **Recognise sign** once per attempt. Do not retry a failed capture
   silently — a capture that reports "not clear enough to read" is a
   **failed capture** and is one of the outcomes being measured.
3. Note anything unusual (hand left frame, glare, phone moved).

## Retrieval

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
```

Then join with the annotations:

```bash
python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

The script prints per-attempt rows, top-1/top-5, rejection rate, accuracy at
accepted coverage, a confusion matrix, and a per-sign table. It refuses to run
if the two files disagree on capture count.

## Failure categories

Every attempt lands in exactly one bucket. The point of the categories is that
they demand different responses:

| Category | Test | Response |
| --- | --- | --- |
| Failed capture | app reported a non-result state, or `hand_frames`/`frames` below 15% | capture/UX fix |
| Failed detection | `hand_frames` low but the hand was genuinely in frame | detector or framing question |
| Feature extraction | parity test passes and the clip replays correctly in Python, but the phone path differs | pipeline bug |
| Model error | high coverage (≥40%), correct performance, wrong label | model question for Dev |
| Out of vocabulary | the intended sign is not one of the 100 labels | not a failure — but the app must not pretend otherwise |
| Uncertain | score below the 0.35 cut | working as designed; report as rejected |

A single failure is not evidence of a category. A pattern across attempts is.

## Reporting rules

- Never report a rounded-up figure. 12/12 on selected clips is not a
  generalisation claim.
- Report accuracy **with** its rejection rate. Accuracy on accepted outputs and
  coverage are two numbers, always quoted together.
- Report per-signer figures separately. A pooled number hides a single bad
  signer.
- Do not compare against a different held-out set. Before/after must use the
  same clips or the same phone session.