# Why landmark robustness fails, and why it is not fixable in preprocessing

Written 16 Nov 2026 by Dev. The only failing criterion left is
`drop_hand_25pct` at **+5.76** against a 5.00 budget. This is what it is.

## The decomposition

Same 1,249 test clips, native cache, clean baseline **85.45%**. Three versions
of the corruption, differing only in which half of the hand is erased:

| condition | accuracy | drop |
| --- | --: | --: |
| clean | 85.45% | — |
| presence flag zeroed, coordinates kept | 79.94% | +5.51 |
| both zeroed (what the harness does) | 79.70% | +5.76 |
| coordinates zeroed, presence kept | 77.94% | +7.51 |

## What that says

**The attention mask is working correctly.** Telling the model "this frame has
no hand" while leaving stale coordinates in place is *better* (+5.51) than
leaving coordinates that still look like a hand (+7.51). The mask is doing
exactly its job.

**The residual is genuine information loss.** Even signalled perfectly, losing
one hand for a quarter of a clip costs about 5.5 points. The model is not
confused by bad input — it is missing input that some signs need.

That closes off the obvious fix. There is no preprocessing change that recovers
this, because the information is not there to recover: if a sign is two-handed
and one hand is absent for a quarter of it, the classifier cannot see what it
needs. The options are training-time (augment harder so the model leans on the
remaining hand) or accept it.

## An earlier version of this measurement was wrong

I first ran this diagnostic without setting `robustness.CACHE_ROOT`, so it
silently scored the **old** cache and returned a clean baseline of 76.94% with
drop-hand at 71.58%. The conclusion looked the same (mask fine, residual real)
but every number was ~8.5 points low. The figures above are the corrected ones,
with `R.CACHE_ROOT` pointed at `.scratch/landmarks_native`.

Worth recording because it is the same class of mistake this session kept
producing: a script that looks right, runs without error, and measures a
different corpus than intended.

## Where that leaves the criterion

`drop_hand_25pct` fails by 0.76 points. It is the only failing corruption —
the other five pass, and pixel robustness passes at +4.48. Whether 0.76 points
over is worth a retraining campaign is a judgement call, but it should be made
knowing that:

- it is a model limitation, not a bug;
- the cache fix did not help it (+5.36 → +5.76);
- a 60-epoch retrain at app geometry scored 76.26%, well below the current
  85.45%, so a naive retrain makes things worse, not better.

The realistic options are (a) report it honestly as one known weakness, or
(b) run a proper augmentation sweep with the original recipe, which the earlier
`search.py` work already showed struggles to beat the shipped ensemble.

## Reproducing

```
python .scratch/drop_hand_diag2.py    # the three-way decomposition
python .scratch/drop_hand_base.py     # the clean baseline
```

Both set `R.CACHE_ROOT = .scratch/landmarks_native` explicitly.
