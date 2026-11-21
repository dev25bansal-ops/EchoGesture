"""Measure avg_power and thermal_drift_10min from the device's own counters.

Both metrics are in the organizer's table (p.11) and both are currently
NOT_MEASURED in results.csv. Neither needs a Monsoon meter: Android exposes
battery current and thermal status through `dumpsys`, which is enough to fill
the cells with a real figure instead of a blank.

WHAT THIS DOES
  - samples `dumpsys battery` (current now, in microamps, on devices that
    report it) and `dumpsys thermalservice` once a second
  - optionally drives the app so the GPU path is actually exercised; without
    that the numbers describe an idle phone and mean nothing
  - writes a CSV of raw samples plus a summary with the metric names, units
    and verdicts the submission table wants

WHAT IT CANNOT DO
  - It cannot run without a connected device. With none it says so and exits
    non-zero rather than inventing numbers.
  - `dumpsys battery` current is not available on every OEM build. If the field
    is absent the script reports that plainly instead of substituting 0 — a
    silent zero would read as "0 mW", which is the most misleading possible
    answer for a power metric.

USAGE
    python scripts/measure_power_thermal.py --seconds 600
    python scripts/measure_power_thermal.py --seconds 60 --no-launch   # idle baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
ADB = "adb"
PKG = "org.echogesture.finalapp"


def adb(*args, timeout=20):
    """Run an adb command, returning stdout or None if adb/device is missing."""
    try:
        r = subprocess.run([ADB, *args], capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def require_device():
    out = adb("devices")
    if out is None:
        raise SystemExit(
            "adb not found. Install platform-tools or put adb on PATH "
            "(this repo uses /d/android-sdk/platform-tools/adb.exe).")
    lines = [l for l in out.splitlines()[1:] if l.strip()]
    ready = [l for l in lines if "\tdevice" in l]
    if not ready:
        raise SystemExit(
            "no device connected (`adb devices` lists none). Power and thermal "
            "cannot be measured without hardware; refusing to emit a number.")
    return ready[0].split("\t")[0]


def battery_sample():
    """Return (current_uA, voltage_uV, level_pct) or Nones where unavailable.

    Many OEM builds — including the Vivo V2404 this was written for — do not
    report `current now` in `dumpsys battery`, and read `/sys/class/power_supply/
    battery/current_now` as permission denied to a shell user. Power then has to
    come from the battery-historiographer instead; see uid_drain_uah().
    """
    out = adb("shell", "dumpsys", "battery")
    if not out:
        return None, None, None
    cur = volt = lvl = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("current now:"):
            m = re.search(r"(-?\d+)", line)
            if m:
                cur = int(m.group(1))
        elif line.startswith("voltage:"):
            m = re.search(r"(\d+)", line)
            if m:
                volt = int(m.group(1)) * 1000  # mV -> uV
        elif line.startswith("level:"):
            m = re.search(r"(\d+)", line)
            if m:
                lvl = int(m.group(1))
    return cur, volt, lvl


def uid_drain_uah(pkg):
    """Charge consumed by `pkg`'s uid since boot, in uAh, from batterystats.

    `dumpsys batterystats --charged` prints a per-uid summary line like
        UID u0a431: 741 fg: 775 (1h 0m 38s) ... camera=622 (51m 39s) cpu=11.7 ...
    where the leading number is mAh-equivalent charge. Taking a delta across a
    known window turns it into an average current: uAh / hours = uA.

    Two things this got wrong until 2026-11-22, both proven against a live
    Android 16 Vivo V2404:

      * The uid was read from `dumpsys package` with `userId=`. That build prints
        `appId=10431` and no `userId=` at all, so the lookup always returned None
        and the caller concluded "no per-uid entry exists". It does exist.
      * The block pattern was `Uid u0?{uid}`. The device prints `UID u0a431`, with
        a capital UID and an `a` before the app id, so it could never match on any
        modern Android. The numeric uid is 10000 + appId, rendered `u0a<appid>`
        (or `u<uid>` for system uids below 10000).

    Returns None only when the app genuinely has no entry, which is the honest
    answer for a process that has not run since the stats were last reset.
    """
    out = adb("shell", "dumpsys", "batterystats", "--charged", timeout=60)
    if not out:
        return None
    app_id = None
    # Map package -> app id. Android 16 prints `appId=`; older builds `userId=`.
    pm = adb("shell", "dumpsys", "package", pkg, timeout=30)
    if pm:
        m = re.search(r"\b(?:appId|userId)=(\d+)", pm)
        if m:
            app_id = int(m.group(1))
    if app_id is None:
        return None
    # `UID u0a431` is how appId 10431 is rendered: `u` + the 0-a prefix + the
    # app id MINUS the leading 10 (10431 -> "431"). Verified on-device.
    uid_forms = []
    if app_id >= 10000:
        uid_forms.append(rf"UID u0a{app_id - 10000}")
    uid_forms.append(rf"UID u{app_id}")
    for form in uid_forms:
        m = re.search(rf"^\s*{form}:\s*([0-9.]+)", out, re.MULTILINE)
        if m:
            return float(m.group(1)) * 1000.0  # mAh -> uAh
    return None


def component_drain_mah():
    """Per-component charge use from `dumpsys batterystats --charged`, in mAh.

    This is the path that actually works on a Vivo/BBK build (Funtouch/OriginOS
    removes `current now` from `dumpsys battery` and denies sysfs reads). The
    "Estimated power use (mAh)" block breaks the total into screen / cpu /
    camera / audio / video / sensors / wakelock, each with a duration, which is
    far more useful than a single aggregate: for a camera app the `camera` line
    IS the inference-adjacent cost.

    Returns (dict of component->mAh, dict of component->seconds) or (None, None).
    """
    out = adb("shell", "dumpsys", "batterystats", "--charged", timeout=60)
    if not out:
        return None, None
    try:
        block = out.split("Estimated power use (mAh):", 1)[1]
    except IndexError:
        return None, None
    # Only the first (device-total) section, before any nested per-state repeat.
    block = block.split("(on battery", 1)[0]
    mah, secs = {}, {}
    for line in block.splitlines():
        m = re.match(r"\s+([a-z_]+):\s*([0-9.]+)", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(2))
        if name in ("Capacity", "Computed", "actual"):
            continue
        mah.setdefault(name, val)
        d = re.search(r"duration:\s*(?:(\d+)m\s*)?([0-9.]+)s", line)
        if d:
            secs[name] = (int(d.group(1) or 0) * 60) + float(d.group(2))
        else:
            # formats like "48m 52s 331ms"
            d2 = re.search(r"duration:\s*(?:(\d+)h\s*)?(?:(\d+)m\s*)?([0-9.]+)s", line)
            if d2:
                secs[name] = (int(d2.group(1) or 0) * 3600 + int(d2.group(2) or 0) * 60
                              + float(d2.group(3)))
    return mah, secs


def reset_batterystats():
    """Start a clean measurement window. Needs shell perms, which adb has."""
    return adb("shell", "dumpsys", "batterystats", "--reset", timeout=30) is not None


def thermal_sample():
    """Return (thermal_status, hottest_soc_temp_c).

    Takes the hottest CPU/GPU/NSP sensor specifically, NOT the max over every
    sensor on the device. An earlier version did `max(all mValue)` and latched
    onto CPU7 — which happened to sit at a constant 63.2 for a whole run — so
    the 10-minute drift read a flat 0.0% and looked like a clean PASS when it
    was really one latched core. Restricting to the compute sensors and keeping
    the whole per-sensor map makes a stale reading visible instead of hiding it.
    """
    out = adb("shell", "dumpsys", "thermalservice")
    status = None
    by_name = {}
    if out:
        m = re.search(r"Thermal Status:\s*(\d+)", out)
        if m:
            status = int(m.group(1))
        for val, name in re.findall(
                r"Temperature\{[^}]*mValue=([\d.]+)[^}]*mName=(\w+)", out):
            by_name[name] = float(val)
    compute = [v for n, v in by_name.items()
               if n.startswith(("CPU", "GPU", "nsp", "soc"))]
    return status, (max(compute) if compute else None), by_name


STATUS_NAMES = {0: "NONE", 1: "LIGHT", 2: "MODERATE", 3: "SEVERE",
                4: "CRITICAL", 5: "EMERGENCY", 6: "SHUTDOWN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600,
                    help="sampling window; 600 for the 10-minute thermal metric")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--no-launch", action="store_true",
                    help="do not start the app (measures an idle phone)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = require_device()
    print(f"device: {device}")

    if not args.no_launch:
        print(f"launching {PKG} so the detector actually runs...")
        if adb("shell", "monkey", "-p", PKG, "-c",
               "android.intent.category.LAUNCHER", "1") is None:
            print("  WARNING: could not launch the app; samples will describe "
                  "an idle device and understate power", file=sys.stderr)
        time.sleep(5)

    # Three possible power sources, in order of fidelity. Thermal sampling is
    # INDEPENDENT of all of them and must never be blocked by a power probe:
    # the two metrics are separate rows in the organizer's table, and refusing
    # to collect one because the other is unavailable would repeat the bug this
    # comment was written to fix (an earlier version exited before sampling a
    # single thermal reading because the power path had no data).
    c0, v0, _ = battery_sample()
    direct = c0 is not None
    drain0 = None
    cam0 = None
    if direct:
        print("power source: dumpsys battery current+voltage (direct)")
    else:
        drain0 = uid_drain_uah(PKG)
        if drain0 is not None:
            print(f"power source: batterystats per-uid drain (start {drain0:.0f} uAh)")
        else:
            mah0, _ = component_drain_mah()
            if mah0 and "camera" in mah0:
                cam0 = mah0["camera"]
                print(f"power source: batterystats COMPONENT drain "
                      f"(camera start {cam0:.1f} mAh)")
            else:
                print("power source: NONE available on this device — power will "
                      "be reported as NOT DERIVED. Thermal is still collected.")

    samples = []
    t_start = time.time()
    print(f"sampling every {args.interval:.1f}s for {args.seconds}s "
          f"(Ctrl-C stops early and still summarises)")
    try:
        while time.time() - t_start < args.seconds:
            cur, volt, lvl = battery_sample()
            status, temp, by_name = thermal_sample()
            samples.append(dict(
                t=round(time.time() - t_start, 2),
                current_uA=cur, voltage_uV=volt, level_pct=lvl,
                thermal_status=status, temp_c=temp,
                soc_json=json.dumps(by_name, sort_keys=True) if by_name else ""))
            if len(samples) % 30 == 0:
                el = time.time() - t_start
                print(f"  {el:5.0f}s  {len(samples):4d} samples  "
                      f"soc={temp} C  thermal={STATUS_NAMES.get(status, status)}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped early; summarising what was collected")

    if not samples:
        raise SystemExit("no samples collected")

    out = Path(args.out) if args.out else OUT / "power_thermal_samples.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0]))
        w.writeheader()
        w.writerows(samples)
    print(f"\nwrote {out}  ({len(samples)} samples)")

    # --- summary ---------------------------------------------------------
    elapsed_h = max(time.time() - t_start, 1e-6) / 3600.0
    powers_mw = []
    for s in samples:
        if s["current_uA"] is not None and s["voltage_uV"]:
            # P = V * I ; uV * uA = pW ; /1e9 -> mW
            powers_mw.append(abs(s["voltage_uV"] * s["current_uA"]) / 1e9)

    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    if powers_mw:
        mean_mw = statistics.mean(powers_mw)
        print(f"  avg_power      : {mean_mw:.1f} mW   "
              f"(target <=500)  -> {'PASS' if mean_mw <= 500 else 'FAIL'}")
        print(f"                   range {min(powers_mw):.1f}-{max(powers_mw):.1f} mW "
              f"over {len(powers_mw)} samples")
        print("                   DIRECT measurement: V x I at the battery.")
    elif drain0 is not None:
        drain1 = uid_drain_uah(PKG)
        if drain1 is None:
            print("  avg_power      : NOT DERIVED (batterystats entry vanished)")
        else:
            used_uah = max(drain1 - drain0, 0.0)
            mean_ua = used_uah / elapsed_h
            volt = samples[-1]["voltage_uV"] or 3_800_000  # nominal if unreported
            mean_mw = (volt * mean_ua) / 1e9
            print(f"  avg_power      : {mean_mw:.1f} mW   "
                  f"(target <=500)  -> {'PASS' if mean_mw <= 500 else 'FAIL'}")
            print(f"                   from {used_uah:.0f} uAh over "
                  f"{elapsed_h*60:.1f} min; V assumed {volt/1e6:.2f} V"
                  f"{' (measured)' if samples[-1]['voltage_uV'] else ' (nominal)'}")
            print("                   APPROXIMATE: batterystats is a coarse,")
            print("                   periodically-flushed counter. Use the")
            print("                   direct path where the device offers it.")
    else:
        # batterystats component path: the one this device actually supports.
        mah, secs = component_drain_mah()
        if not mah:
            print("  avg_power      : NOT DERIVED (no batterystats power block)")
            raise SystemExit(1)
        # Attribute the camera line to inference-adjacent work and the cpu line
        # to compute; both are reported because the metric is ambiguous and a
        # reader needs to see which one is being claimed.
        V_NOMINAL = 3.87  # Li-ion nominal for a 4612 mAh pack
        # Use the DELTA across our own window when a baseline was captured.
        # Using the all-time total would attribute every other app's camera use
        # in this battery cycle to us, which is how you get a believable-looking
        # but meaningless number.
        mah1, _ = component_drain_mah()
        print(f"  batterystats components, V assumed {V_NOMINAL}:")
        rows_out = []
        for comp in ("camera", "cpu", "sensors", "video", "screen"):
            if comp not in mah:
                continue
            # Only components with a captured baseline get a delta. Presenting an
            # all-time total divided by this run's duration is a meaningless
            # number (it produced a nonsense 128 W row once), so those are
            # labelled with their real accumulation time instead.
            if comp == "camera" and cam0 is not None and mah1 and comp in mah1:
                value = max(mah1[comp] - cam0, 0.0)
                h = elapsed_h
                label = "delta"
            else:
                value = mah[comp]
                h = secs.get(comp, elapsed_h)
                label = f"all-time/{h/60:.0f}min"
            mw = (value * V_NOMINAL) / max(h, 1e-6)
            rows_out.append((comp, value, h, mw, label))
            print(f"    {comp:10s} {value:7.2f} mAh ({label}) "
                  f"= {mw:6.0f} mW")
        cam = next((r for r in rows_out if r[0] == "camera"), None)
        if cam:
            verdict = "PASS" if cam[3] <= 500 else "FAIL"
            print(f"\n  avg_power (camera) : {cam[3]:.0f} mW  "
                  f"(target <=500) -> {verdict}")
            print("    The camera line covers sensor + ISP + encode, NOT inference")
            print("    alone, so this is an UPPER BOUND on what our pipeline costs.")
            print("    A 500 mW bar for the whole camera subsystem is far below what")
            print("    any phone camera costs to run, so FAIL here is expected and")
            print("    says little about the model specifically.")

    statuses = [s["thermal_status"] for s in samples if s.get("thermal_status") is not None]
    temps = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
    # A sensor that returns an identical value for the entire run is LATCHED,
    # not stable. Reporting 0.0% drift from it would be a false PASS: an earlier
    # run of this script did exactly that, reading a constant 63.2 C for 10
    # minutes while CPU cores actually ranged 45.4-63.2 and thermal status went
    # 0 -> 1. Check for staleness before trusting the drift number.
    latched = len(set(temps)) <= 1 and len(temps) > 3
    if latched:
        print(f"  thermal_drift  : NOT MEASURABLE — the SoC sensor returned the "
              f"same value ({temps[0]:.1f} C) for all {len(temps)} samples.")
        print("                   A constant reading is a latched sensor, not a")
        print("                   stable one; reporting 0.0% would be a false PASS.")
    elif temps and len(temps) > 1:
        drift = (max(temps) - min(temps)) / max(temps) * 100
        print(f"  thermal_drift  : {drift:.1f} %   (target <=20)  "
              f"-> {'PASS' if drift <= 20 else 'FAIL'}")
        print(f"                   SoC {min(temps):.1f} -> {max(temps):.1f} C")
    if statuses:
        worst = max(statuses)
        changed = len(set(statuses)) > 1
        print(f"  thermal_status : worst {STATUS_NAMES.get(worst, worst)}"
              f"{' (varied during the run)' if changed else ' (unchanged)'}")
        if latched or not temps:
            print("                   This is the thermal figure to quote when the")
            print("                   temperature sensor is unusable.")
    if not temps and not statuses:
        print("  thermal_drift  : NOT DERIVED")

    print("\n  These describe THIS device under THIS run. Cite the device and "
          "build in results.csv, as the samples' own comments require.")


if __name__ == "__main__":
    main()
