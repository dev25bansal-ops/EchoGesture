#!/bin/sh
# EchoGesture phone diagnostic collector.
#
# Run this on the Mac with the phone connected and USB debugging authorised.
# It reads the on-device diagnostics record the app already keeps. Nothing is
# uploaded; the file lands next to this script.
#
# Usage:  sh collect-phone-diagnostics.sh
set -eu

ADB="${ADB:-$HOME/AppData/ActionMesh-fork/android-toolchain/sdk/platform-tools/adb}"
PKG=org.echogesture.finalapp
OUT="phone-diagnostics-$(date +%Y%m%d-%H%M%S).txt"

if [ ! -x "$ADB" ]; then
  echo "adb not found at: $ADB" >&2
  echo "Set ADB=/path/to/adb and re-run." >&2
  exit 1
fi

echo "Devices:"
"$ADB" devices -l
echo

# A physical device is the point of this script; the emulator camera is
# synthetic and never contains a hand.
# adb separates serial from state with a TAB, so split on whitespace with awk
# rather than a grep character class that does not expand reliably under /bin/sh.
#
# Pick a REAL DEVICE, and pin it with -s for every later call. With two targets
# attached (an emulator left running plus the phone) an unpinned adb command uses
# whichever it likes, which could collect the emulator's record and present it as
# phone evidence. Emulators are filtered out first, so an attached phone always
# wins; if only emulators are present the script refuses.
AWK_DEV='$2=="device"{print $1}'
AWK_EMU='!/^emulator-/'

CANDIDATES=$("$ADB" devices | awk "$AWK_DEV")
TOTAL=$(printf '%s\n' "$CANDIDATES" | grep -c . || true)
REAL=$(printf '%s\n' "$CANDIDATES" | awk "$AWK_EMU" | grep -c . || true)

if [ "$TOTAL" -eq 0 ]; then
  echo "No authorised device. Enable USB debugging and accept the prompt." >&2
  echo "If the phone shows 'unauthorized', unlock it and tap Allow." >&2
  exit 1
fi

if [ "$REAL" -eq 0 ]; then
  echo "Only emulator(s) attached: $(printf '%s ' $CANDIDATES)" >&2
  echo "An emulator camera is synthetic and never contains a hand." >&2
  echo "Its capture statistics are not phone evidence, so this script will not" >&2
  echo "collect them. Attach the phone and re-run." >&2
  exit 1
fi

if [ "$REAL" -gt 1 ]; then
  echo "More than one non-emulator device is attached:" >&2
  printf '  %s\n' $(printf '%s\n' "$CANDIDATES" | awk "$AWK_EMU") >&2
  echo "Set SERIAL=<the phone you mean> and re-run, so the record cannot come" >&2
  echo "from the wrong handset." >&2
  exit 1
fi

SERIAL="${SERIAL:-$(printf '%s\n' "$CANDIDATES" | awk "$AWK_EMU" | head -1)}"
ADB="$ADB -s $SERIAL"
echo "Using device: $SERIAL"
echo

MODEL=$("$ADB" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
echo "Model: $MODEL"
if echo "$MODEL" | grep -qiE 'sdk_gphone|emulator'; then
  echo "WARNING: this looks like an emulator, not a phone." >&2
  echo "Live-camera results from it are not phone results." >&2
fi

echo "Package: $("$ADB" shell dumpsys package $PKG 2>/dev/null | grep -m1 versionName | tr -d '\r' | tr -s ' ')"
echo "Camera permission: $("$ADB" shell dumpsys package $PKG 2>/dev/null | grep -m1 'android.permission.CAMERA:' | tr -d '\r')"
echo

{
  echo "# EchoGesture phone diagnostics"
  echo "# model=$MODEL"
  echo "# collected=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# capture statistics only; no images, no landmarks"
  echo
  "$ADB" shell run-as $PKG cat files/diagnostics.txt 2>/dev/null || echo "(no diagnostics record)"
} > "$OUT"

echo "Wrote $OUT"
echo
echo "Now fill in PHONE_TRIALS.csv with what you intended versus what was shown,"
echo "then send both files back."
