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
if ! "$ADB" devices | grep -qE '^[0-9A-Za-z]+[[:space:]]+device$'; then
  echo "No authorised device. Enable USB debugging and accept the prompt." >&2
  exit 1
fi

MODEL=$("$ADB" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
echo "Model: $MODEL"
if echo "$MODEL" | grep -qiE 'sdk_gphone|emulator'; then
  echo "WARNING: this looks like an emulator, not a phone." >&2
  echo "Live-camera results from it are not phone results." >&2
fi

echo "Package: $("$ADB" shell dumpsys package $PKG 2>/dev/null | grep -m1 versionName | tr -d '\r')"
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
