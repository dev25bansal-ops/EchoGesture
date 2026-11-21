#!/bin/sh
# Read the private V5.5 diagnostics log from a locally connected phone.
set -eu

ADB="${ADB:-$HOME/AppData/ActionMesh-fork/android-toolchain/sdk/platform-tools/adb}"
PKG=org.echogesture.finalapp
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
OUT="${OUT:-$SCRIPT_DIR/phone-diagnostics-$(date +%Y%m%d-%H%M%S).txt}"

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

[ -x "$ADB" ] || fail "Set ADB to the path of an executable adb."
DEVICES=$("$ADB" devices) || fail "Could not list ADB devices."
PHONES=$(printf '%s\n' "$DEVICES" | awk '$2 == "device" && $1 !~ /^emulator-/ {print $1}')
[ -n "$PHONES" ] || fail "Attach an authorised physical phone with USB debugging enabled."

if [ -n "${SERIAL:-}" ]; then
  printf '%s\n' "$PHONES" | grep -Fxq -- "$SERIAL" ||
    fail "SERIAL is not an authorised physical phone in the ADB device list."
else
  count=$(printf '%s\n' "$PHONES" | awk 'END {print NR}')
  if [ "$count" -ne 1 ]; then
    printf 'Choose one of these phones with SERIAL:\n%s\n' "$PHONES" >&2
    exit 1
  fi
  SERIAL=$PHONES
fi

phone() {
  "$ADB" -s "$SERIAL" "$@"
}

MODEL=$(phone shell getprop ro.product.model) || fail "Could not read the phone model."
MODEL=$(printf '%s' "$MODEL" | tr -d '\r')
QEMU=$(phone shell getprop ro.kernel.qemu) || fail "Could not verify the device type."
case "$(printf '%s' "$QEMU" | tr -d '\r'):$MODEL" in
  1:*|*sdk_gphone*|*emulator*) fail "An emulator cannot supply physical phone diagnostics." ;;
esac
PACKAGE=$(phone shell dumpsys package "$PKG") || fail "Could not inspect the installed app."
VERSION=$(printf '%s\n' "$PACKAGE" | awk '/versionName=/ {sub(/^[[:space:]]+/, ""); print; exit}')
PERMISSION=$(printf '%s\n' "$PACKAGE" | awk '/android.permission.CAMERA: granted=/ {sub(/^[[:space:]]+/, ""); print; exit}')
[ -n "$VERSION" ] || fail "Install EchoGesture before collecting diagnostics."

RECORD=$(phone shell run-as "$PKG" cat files/diagnostics.txt) ||
  fail "Could not read the app's diagnostics. Enable Capture diagnostics or use Settings → Export diagnostics."
printf '%s\n' "$RECORD" | grep -q '^frames=' ||
  fail "The diagnostics log has no attempts yet. Complete a capture and retry."

TEMP=$(mktemp "$OUT.XXXXXX")
trap 'rm -f "$TEMP"' EXIT
{
  printf '# EchoGesture phone diagnostics\n# model=%s\n# %s\n' "$MODEL" "$VERSION"
  printf '# collected=%s\n# capture statistics only; no images, no landmarks\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\n' "$RECORD"
} > "$TEMP"
mv "$TEMP" "$OUT"
printf 'Model: %s\nPackage: %s\nCamera permission: %s\nWrote %s\n' "$MODEL" "$VERSION" "${PERMISSION:-unknown}" "$OUT"
