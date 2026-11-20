# Demo script audit, 2026-11-19

The narration script is what gets spoken over the recording, so a wrong number in
it is worse than a wrong number in a document nobody reads aloud. It was checked
against the app's actual UI strings and against `results/results.csv`.

## Errors found and fixed

### The UI text the script quoted does not exist

Beat 5 said the app shows **"Uncertain — sign again"**. The real string in
`res/values/strings.xml` is:

```
result_confidence_uncertain = "Uncertain — this is the model's best guess,
                               not a confirmed reading"
result_confidence_accepted  = "Best match — still confirm before relying on it"
```

A presenter reading the script would describe words the audience cannot see.

### The confidence figure described a rule the app does not implement

The script claimed *"if it only answers above fifty percent confidence, it's
right 95.9% of the time — on 73% of inputs."* Both numbers are real, but they
describe a **confidence ≥ 0.5** rule. The shipped app gates on
**top-1 minus top-2 margin ≥ 0.15**. Measured on the same 1,251 clips:

| rule | coverage | accuracy |
|---|---|---|
| **shipped: margin ≥ 0.15** | **86.3%** | **92.1%** |
| confidence ≥ 0.5 (quoted) | 72.6% | 95.9% |
| confidence ≥ 0.8 (quoted) | 47.5% | 99.7% |

Three real numbers, three different rules. The script quoted the wrong one while
the audience watches the app.

### Model weights were overstated

| | script said | actual (from the APK) |
|---|---|---|
| nine members | 8.5 MB | **8.11 MB** |
| detector | 7.8 MB | **7.46 MB** |
| total | 16.3 MB | **15.57 MB** |

### Analyser rate was overstated

The Q&A said "9–14 analysed frames per second". Measured across six samples and
two installs: **8.7–11.6**. The upper bound was too generous.

### A figure dangled after its own removal

Beat 4 opened *"those latency numbers are desktop, not fifteen"* — referring to a
15 ms/desktop line that was itself wrong and is now gone. Rewritten to lead with
the measured phone figure.

## What was checked and is correct

- **85.45% / 96.64%** — `results.csv` rows `top1_accuracy`, `top5_accuracy`.
- **Ensemble gain 3.92 points** — best single member is 81.53%
  (`top1_single_best_member`); 85.45 − 81.53 = 3.92. The claim holds.
- **Robustness in Beat 4** — pixel +4.48 (under five, as stated) separated
  correctly from landmark +5.76. Accurate.
- **"17–24 clips per class"** — correct, and I nearly broke it. Checking all
  three splits together gives 29–45, which made the claim look wrong; the
  *training* split is exactly 17–24. **Verified before changing anything, and the
  original left alone.**

## Still needs a human

The script is narration, not the deck. It still needs to be matched to whatever
actually appears on screen during the take — in particular the exact moment the
result card appears, since Beat 5 now asks the presenter to read its label aloud.
