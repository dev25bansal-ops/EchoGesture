# V8 Dev INT8 Handoff (minimum)

**Verdict: INT8 NOT SELECTED as default. Keep FP16.** On the 346 official val clips,
FP16 272/346 (Wilson 95% [73.99%, 82.61%]) vs INT8 254/346 [68.52%, 77.79%] —
FP16 lead 18 clips / +5.20 pp. Paired (345 successful, 1 shared no-hand failure):
both correct 242, FP16-only 30, INT8-only 12, both wrong 61; exact McNemar p=0.007916;
clip-bootstrap paired-difference 95% [0.017, 0.090]. Top-5: FP16 309/346, INT8 298/346.
Argmax agreement 281/345 (81.4%). Failure: `5989539775823847-INTUITIVE` (P5)
counted incorrect for both models in the primary 346 denominator.

**Baseline to beat** (same 346 val, identical features, no test use): FP16 272/346.
Any new INT8/PQ candidate must report on this set with feature hashes, per-signer
(P12: FP16 54.2% / INT8 33.3% is the worst signer gap; P5 and P12 are the weak strata),
per-class (INT8 worst: CHEEK 0/4, EMPTY2 0/3, ELEVATOR1 1/5, FINE1 1/4), Top-1/Top-5,
agreement, logit/prob drift (mean |logit| 1.74, L2 5.35, prob-L1 0.58), and float-op
list; ship checkpoint SHA, representative-set identity, converter command, and dtype
map. INT8 saves size/speed but classifier time is negligible vs frontend — accuracy
regression is not offset. Test split remains locked; no tuning/retraining was done here.
