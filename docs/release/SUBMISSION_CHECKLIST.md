# Submission checklist — Track 02

Deadline **18 Nov 2026 11:59 PM** (5 days from 2026-11-13). One submission per
team via the Google Form. **Do not submit until you have read item 0.**

## 0. Do this before anything else

- [ ] **Confirm the team is registered.** Registration closed **8 Nov 2026**.
      Deck p.24: the final submission link is "ONLY shared with Teams who have
      completed the Team registration process". If you are not registered, there
      is no link to fill in. Nothing else on this page matters until this is
      confirmed.
- [ ] **Find the Google Form link** in the email sent to the registered address.
      It is not in the repository and I have no access to it.

## 1. Blocking external inputs

- [ ] Send `docs/release/ORGANIZER_CLARIFICATION.md` from the registered address.
- [ ] If a **Track 02 vocabulary** arrives, re-run `predict()` over the official
      manifest and regenerate `results/predictions.jsonl` + `.json`.
- [ ] If a **test manifest** arrives, verify every `clip_id` matches exactly:
      `python3 scripts/release/predictions_json_from_jsonl.py` refuses
      duplicates and malformed rows, but exact ID matching is yours to check.

## 2. Package to build

Required layout (deck p.12):

```
src/
  main.py            predict(video_path) -> dict
  requirements.txt
  README.md          setup, Docker, evaluator contract
  features.py, landmark_frontend.py, parallel_tracker.py
results/
  predictions.jsonl  required by p.11  ✅ 1249 rows
  predictions.json   required by p.12  ✅ generated from the same source
  results.csv        6 metrics          ✅ 2 are NOT_MEASURED
Dockerfile           builds + runs     ✅ verified
README.md            contract + Docker ✅
```

- [ ] `python3 scripts/release/build_bundle.py build` — produces the release ZIP.
      It **refuses** an `--official` build without a real organizer manifest.
      That refusal is correct; do not work around it.
- [ ] Record the ZIP's SHA-256 and include it with the upload.

- [ ] **Model weights are not in the ZIP.** They are not redistributable. The
      evaluator must receive them through an authorized channel, or be told
      where to fetch them. Decide this with whoever owns the ASL Citizen terms.

## 3. Presentation

- [ ] Rename to **`CollegeName_TeamName`** exactly. Deck p.25: "Teams NOT
      following the submission guideline would lead to direct disqualification."
- [ ] Source deck: `docs/submission/EchoGesture_Submission.pptx` (11 slides,
      numbers verified against evidence this session).
- [ ] Optionally insert `docs/submission/architecture.svg` on the architecture
      slide — it is a real diagram and that slide currently has only text.
- [ ] Re-check slide 6 and 7 numbers against `results/results.csv` after any
      re-measurement.

## 4. Demo video

- [ ] **Not recorded.** Maximum 3 minutes (p.25; p.10 says 2–3 min, p.24 says 5
      min, p.25 is stricter).
- [ ] Script: `docs/demo/DEMO_SCRIPT.md`; shot list: `docs/demo/SHOT_LIST.md`.
- [ ] Must show: a real phone, genuine moving signs, offline mode, a correct
      prediction, an uncertain/rejected one, latency on screen.
- [ ] Upload to YouTube or Drive; put the link in the form. Do not upload video
      containing bystanders without consent.

## 5. Forms and approvals

- [ ] Google Form — **submit it yourself.** I will not.
- [ ] Confirm you have the right to submit under the team name you registered.

## What is verified right now

| Item | State |
| --- | --- |
| Python suite | PASS |
| Docker build | PASS |
| Real inference in Docker | PASS (`EAT1` → `EAT1`, 0.5274) |
| `predictions.jsonl` structure | PASS (1249 rows, unique, uppercase, top5-consistent) |
| `predictions.json` | PASS (generated, same source) |
| README documents contract + Docker | PASS |
| Android APK identity | `530f5ec4…`, 77,710,351 B |
| Phone accuracy | **NOT_TESTED** |
| Phone latency / power / thermal | **NOT_TESTED** |
| INT8 compliance | **NOT CLAIMED** (FP16 shipped) |
| Official vocabulary / manifest | **BLOCKED** on organizer |

## Do not

- Do not claim INT8 compliance.
- Do not claim <50 ms.
- Do not claim phone-verified accuracy.
- Do not call the package official-complete without the organizer manifest.
- Do not upload model weights or ASL Citizen data to a public location.