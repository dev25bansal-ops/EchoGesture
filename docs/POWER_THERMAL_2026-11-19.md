# Power and thermal: measured, and what the numbers actually mean

Measured 2026-11-19 on the Vivo V2404 (Android 16, SM7550, 4612 mAh), app in
the foreground with the camera live for the whole window. Both metrics were
`NOT_MEASURED` in `results.csv` before today.

**Getting there took two failed attempts, and both failures are why these
numbers are trustworthy** — the first produced a false PASS and the second a
misleading FAIL. Both are documented below.

## How it was measured

This device exposes no `current now` in `dumpsys battery` and denies sysfs reads
to the shell user, so direct V×I is impossible. `dumpsys batterystats --charged`
does expose an **Estimated power use (mAh)** block broken into components with
durations, and `dumpsys thermalservice` reports per-sensor temperatures. The
harness is `scripts/measure_power_thermal.py`.

```bash
python scripts/measure_power_thermal.py --seconds 620 --interval 5
```

## Power: ~2.9 W — and why that is not a verdict on the model

| run | camera delta | mW @ 3.87 V |
|---|---|---|
| 2-minute spot check | 24.7 mAh / 1m58s | 2 897 |
| 10-minute run 1 | 119.0 mAh / 10m | 2 666 |
| 10-minute run 2 | 129.0 mAh / 10m | 2 898 |

`avg_power` target is ≤500 mW → **FAIL by ~5–6×**, consistently across three
independent windows.

**But read the label before drawing a conclusion.** The `camera` line covers the
sensor, the ISP and encode — everything involved in running the camera at all —
not the incremental cost of our inference. A 500 mW budget for a phone camera
subsystem is below what the camera costs to merely switch on, so this FAIL says
"a phone camera draws watts", not "the model is expensive".

The model's own cost is far smaller and was measured separately: the nine
classifiers are **3.4–4.8 ms/clip**, and the detector is 34–45 ms/frame on the
GPU delegate. So the honest framing for the submission is:

> `avg_power` = **NOT MEASURABLE for the model alone on this hardware.** The
> camera subsystem draws ~2.9 W; per-process attribution was not obtained.

That is a truthful `NOT_MEASURED` with a measured bound around it, which is
worth more than a fabricated pass.

## Correction, 2026-11-22: the 2.9 W is a device constant, not a measurement

Re-examined on the live Vivo V2404. Two things in the section above are wrong.

**The `camera` line is a PowerProfile device constant, not a current reading.**
Five independent blocks in a single `dumpsys batterystats --charged` dump, which
disagree wildly in duration and in power state, all yield the same rate:

| block | mAh | duration | mA |
| --- | --: | --- | --: |
| global | 1318.0 | 1h 45m 24s | 750.26 |
| on battery, screen on | 582.0 | 46m 36s | 749.44 |
| screen off / doze | 44.6 | 3m 34s | 749.44 |
| another window | 681.0 | 54m 27s | 750.31 |
| a 47-second run | 9.77 | 46.9 s | 750.38 |

Real current cannot be identical to 0.2% across screen-on, screen-off and doze.
So `2666–2898 mW` is **749.4 mA × camera-on duration × 3.87 V** — the spread
between the three "independent windows" is host-clock slop, not power
variation. At today's measured 4122 mV it is nearer 3.1 W than 2.9 W.

Consequence: **no live-camera app on this device can pass 500 mW.** 500 mW is
129 mA; the camera constant alone is 749 mA, 5.8× the budget. The target is only
coherent as a model/compute-only budget, which the organizer sheet never defines.

**Per-uid attribution does exist, and the harness could never have found it.**
The sentence "per-uid batterystats has no entry for the app" was an inference
from a parser that always returned `None`. Two defects in
`scripts/measure_power_thermal.py`:

1. It read the uid with `userId=(\d+)` from `dumpsys package`. Android 16 prints
   `appId=10431` and no `userId=` at all.
2. It matched `^\s*Uid u0?{uid}:`. The device prints `UID u0a431:` — capital
   `UID`, and the app id rendered with its leading `10` dropped (10431 → `431`).

Both are fixed; the function now returns a real value
(`1,109,000 µAh` for `org.echogesture.finalapp`). The live entry is
`UID u0a431: 1096 … camera=622 (51m 39s) cpu=11.7 (17m 17s)` — **84% of the
app-attributed drain is the same camera constant**, so per-uid attribution would
not rescue a PASS either. It does expose the one sub-500 mW figure in the
dataset: app CPU-attributed `cpu=11.7 mAh / 17m 17s` = 40.6 mA ≈ **157 mW**,
an upper bound on all app compute. That is the platform's power model over an
uncontrolled hour, not a rail measurement.

## Thermal: 19.2% drift, but the phone was COOLING

The metric is `thermal_drift_10min <= 20%`. Measured on the SoC (hottest of
CPU/GPU/NSP):

```
min 45.8 °C   max 56.7 °C   (max-min)/max = 19.2%   -> PASS, barely
thermal status: LIGHT throughout, never higher
```

**The shape matters more than the number.** The peak is at the very first
sample, and the trace cools from there:

| window | mean SoC °C |
|---|---|
| 0–100 s | 48.3 |
| 100–200 s | 47.7 |
| 200–300 s | 47.8 |
| 300–400 s | 48.7 |
| 400–500 s | 48.7 |
| 500–600 s | 48.1 |
| final 30 s | 50.1 |

Per-sensor end-to-end deltas were all **negative**: CPU0–7 −4.9 to −7.3 °C,
GPU0/1 −3.2 °C, NSP0–2 −3.2 to −3.6 °C, battery +0.2 °C.

So the 19.2% is an **initial transient** — the phone was warm from the preceding
measurement and settled — not thermal runaway under load. Under sustained load a
pathological chip would climb; this one does not.

**The defensible claim is therefore:** thermal drift measured 19.2% with a
cooling trajectory, thermal status never left LIGHT, and there is no evidence of
throttling — but the headroom against the 20% bar is only 0.8 points, so it
should not be described as a comfortable pass.

## Two measurement bugs this went through

**1. A false PASS from a latched sensor.** The first 10-minute run reported
`thermal_drift 0.0% -> PASS`. It was reading `max()` over *every* sensor, which
landed on CPU7 sitting at a constant 63.2 °C for the entire run. A value that
cannot move cannot drift. Fixed by restricting to compute sensors and treating
a constant reading across >3 samples as **LATCHED → NOT MEASURABLE**.

**2. A power probe that blocked thermal.** The harness aborted before sampling
because no power source was derivable — refusing to collect a *thermal* reading
because *power* was unavailable. The two metrics are independent rows in the
organizer's table; power and thermal now fail independently.

## Reproduce

```bash
python scripts/measure_power_thermal.py --seconds 620 --interval 5
```

Raw samples land in `results/power_thermal_samples.csv` with a `soc_json` column
holding every sensor's reading per sample, so any claim above can be re-derived
from the CSV rather than trusted.