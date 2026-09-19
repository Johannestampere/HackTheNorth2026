#!/usr/bin/env bash
# One-shot setup of the projector display on a Raspberry Pi. Safe to re-run.
#
#   ./install.sh                 install and enable, but do not touch a running desktop
#   ./install.sh --now           also stop the desktop right now and start the display
#   ./install.sh --sender        also enable the sample sender service (sender.py clock)
#   ./install.sh --dry-run       show every command and render the unit files; change nothing
#   ./install.sh --port 9000     display listens on this port (default 8080)
#
# What it does: installs python3-pygame, adds you to the video/render/input/tty groups,
# installs systemd units for the display and a health watchdog, and makes the Pi boot to
# the console instead of a desktop (a desktop would own the HDMI output).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8080; NOW=0; DRY=0; SENDER=0
USERNAME="${SUDO_USER:-${USER:-$(id -un)}}"

while [ $# -gt 0 ]; do
  case "$1" in
    --port)    PORT="$2"; shift ;;
    --now)     NOW=1 ;;
    --dry-run) DRY=1 ;;
    --sender)  SENDER=1 ;;
    --user)    USERNAME="$2"; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

case "$DIR" in *" "*) echo "install path must not contain spaces: $DIR" >&2; exit 1 ;; esac
[ -f "$DIR/vec2projector.py" ] || { echo "run this from the raspi/ folder" >&2; exit 1; }
case "$PORT" in ''|*[!0-9]*) echo "--port must be a number" >&2; exit 2 ;; esac

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

render() {  # render <template> <destination>
  sed -e "s|@USER@|$USERNAME|g" -e "s|@DIR@|$DIR|g" -e "s|@PORT@|$PORT|g" "$1" >"$2"
}

echo "== user: $USERNAME   folder: $DIR   port: $PORT   dry-run: $DRY"

echo; echo "== 1/5 packages"
run $SUDO apt-get update -qq
run $SUDO apt-get install -y python3-pygame curl
if PYGAME_HIDE_SUPPORT_PROMPT=1 python3 -c 'import pygame,sys; sys.exit(0 if "/.local/" in pygame.__file__ else 1)' 2>/dev/null; then
  echo "! pygame is installed from pip in ~/.local and shadows the apt one (which has kmsdrm)."
  echo "!   fix:  pip uninstall pygame   (add --break-system-packages if it complains)"
fi

echo; echo "== 2/5 groups (needed for manual runs over SSH; the service sets its own)"
for g in video render input tty; do
  if getent group "$g" >/dev/null; then run $SUDO usermod -aG "$g" "$USERNAME"; fi
done

echo; echo "== 3/5 systemd units"
if [ "$DRY" = 1 ]; then OUT="$(mktemp -d)"; echo "(dry run: rendering into $OUT, not /etc/systemd/system)"; else OUT=""; fi
install_unit() {  # install_unit <template> <unit-name> [render]
  local tpl="$DIR/pi5/$1" name="$2"
  if [ "$DRY" = 1 ]; then
    render "$tpl" "$OUT/$name"; echo "  rendered $name"
  else
    local tmp; tmp="$(mktemp)"; render "$tpl" "$tmp"
    $SUDO install -m 644 "$tmp" "/etc/systemd/system/$name"; rm -f "$tmp"; echo "  installed $name"
  fi
}
install_unit vec2projector.service.in          vec2projector.service
install_unit vec2projector-watchdog.service.in vec2projector-watchdog.service
install_unit vec2projector-sender.service.in   vec2projector-sender.service
if [ "$DRY" = 1 ]; then cp "$DIR/pi5/vec2projector-watchdog.timer" "$OUT/"; else
  $SUDO install -m 644 "$DIR/pi5/vec2projector-watchdog.timer" /etc/systemd/system/; fi
run $SUDO systemctl daemon-reload
run $SUDO systemctl enable vec2projector.service vec2projector-watchdog.timer
[ "$SENDER" = 1 ] && run $SUDO systemctl enable vec2projector-sender.service

echo; echo "== 4/5 boot to console (a desktop would own the HDMI output)"
run $SUDO systemctl set-default multi-user.target

echo; echo "== 5/5 start"
if [ "$NOW" = 1 ]; then
  run $SUDO systemctl isolate multi-user.target
  run $SUDO systemctl restart vec2projector.service
  [ "$SENDER" = 1 ] && run $SUDO systemctl restart vec2projector-sender.service
  echo "The projector should now show the test card. Try:  python3 $DIR/sender.py clock"
else
  echo "Not started yet, so a running desktop is left alone. Either:"
  echo "  sudo reboot                                  (cleanest)"
  echo "  sudo systemctl isolate multi-user.target     (stop the desktop now, no reboot)"
  echo "  then: sudo systemctl restart vec2projector"
fi

if [ -r /boot/firmware/cmdline.txt ] && ! grep -q 'video=HDMI-A-1' /boot/firmware/cmdline.txt; then
  echo
  echo "Optional, but the usual cause of 'black wall after a power cut': the Pi only brings up"
  echo "HDMI if the projector is already on at boot. To force it on regardless, append this to"
  echo "the single line in /boot/firmware/cmdline.txt (edit by hand; a bad edit can stop boot):"
  echo "    video=HDMI-A-1:1920x1080M@60D consoleblank=0"
fi
echo; echo "Check it:   python3 $DIR/vec2projector.py --diagnose"
echo "Logs:       journalctl -u vec2projector -f"
echo "Health:     python3 $DIR/sender.py health"
