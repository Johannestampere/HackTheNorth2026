# Running vec2projector unattended on a Raspberry Pi 5

Target: Pi 5 wired straight to a projector, boots on its own, never sees a
keyboard again. Raspberry Pi OS Bookworm (64-bit).

## 0. How the two processes fit together

There is no desktop, no window, and nothing being mirrored or captured. Two
independent processes on the Pi:

```
  your server  --- HTTP on 127.0.0.1:8080 --->  vec2projector  ===HDMI==>  projector
  (vec2projector-sender.service)                (vec2projector.service)
```

`vec2projector` owns the HDMI connector from boot to power-off and never exits.
Your server pushes frames into it and is otherwise irrelevant to the picture.

Keep them as **separate systemd units**. That is the whole trick for an
unattended box: if your server crashes, redeploys, or is stopped for an update,
the projector keeps showing the last frame it received instead of going black.
One combined process, or a shell pipe, couples the picture to the server's
uptime. Verified behaviour: with the display killed the server logged dropped
frames and kept running; when the display came back it resumed instantly with
no restart on either side.

Push from your server with three stdlib calls -- see `pi5/push_example.py`, or use the
ready-made `sender.py` (clock, spin, line, image, clear, health):

```python
push_image(png_bytes)             # POST /frame
push_raw(rgb24_bytes, 1280, 720)  # POST /raw?w=&h=   (cheapest on one box)
push_vectors(["v 0 0 1 1 red 4"]) # POST /vectors
```

A push that fails returns `False` rather than raising. Treat it as a dropped
frame and carry on; never let the display take your server down.

**Frames are latest-wins.** If you push faster than the screen refreshes, stale
frames are discarded rather than queued, so the picture is always the newest one
and latency cannot creep upward over hours. Pushing 30 fps into a 60 fps display
shows every frame; pushing 200 fps shows roughly 60 of them.

**"Every now and then" is fine.** With nothing arriving, the display costs
almost nothing and simply holds the last frame indefinitely. There is no
timeout, no blanking, and no screensaver to fight.

### Install all of it

`./install.sh` does this whole section (packages, groups, units, boot target); see
README.md. Run it with `--dry-run` first to see exactly what it will change.

The watchdog timer curls `/health` once a minute and restarts the display if it
has stopped drawing -- the closest thing to a human noticing a frozen projector.

```bash
curl -s localhost:8080/health
# {"ok": true, "uptime_s": 431.2, "fps": 60.0, "vectors": 0,
#  "images_shown": 12940, "last_image_age_s": 0.03}
```

`last_image_age_s` is the useful one: it tells you whether the *server* has gone
quiet, which the display process itself cannot know.

## 1. Install

```bash
sudo apt update && sudo apt install -y python3-pygame
mkdir -p ~/vec2projector && cd ~/vec2projector
# copy vec2projector.py (and your generator) here
python3 vec2projector.py --demo          # sanity check from the desktop first
```

`python3-pygame` from apt is the right choice here — no venv, no
`--break-system-packages`, and it gets security updates with the OS.

## 2. Boot to console, not the desktop

The desktop compositor owns the HDMI connector, which blocks the direct-to-HDMI
path and wastes a chunk of GPU and RAM for a machine with no user.

```bash
sudo systemctl set-default multi-user.target
```

With no desktop running the script auto-selects SDL's `kmsdrm` driver and draws
straight onto the HDMI output — no X, no Wayland, no window manager.
(Force it either way with `--driver kmsdrm` / `--driver wayland`.)

## 3. Make HDMI come up even when the projector is off or slow

This is the one that actually bites unattended installs: if the projector is
powered off, asleep, or boots slower than the Pi, the Pi sees no EDID and may
bring up no output at all — and it will not retry on its own.

Edit `/boot/firmware/cmdline.txt` (one line, append to the end — do not add a
newline):

```
video=HDMI-A-1:1920x1080M@60D consoleblank=0
```

* `HDMI-A-1` is the HDMI port nearest the USB-C power socket (HDMI0).
* The trailing `D` forces **d**igital output on with no EDID present.
* `consoleblank=0` stops the console blanking after 10 minutes.
* Match the mode to your projector's native resolution — a projector fed its
  native mode does no internal scaling and looks noticeably sharper.

Note: `hdmi_force_hotplug=1` in `config.txt` is the Pi 4 answer and does
**nothing** on a Pi 5. Use the `video=` kernel parameter above.

## 4. Autostart as a service

`./install.sh` renders `pi5/*.service.in` with your username, folder and port and installs
them. To do it by hand, substitute `@USER@`, `@DIR@` and `@PORT@` in the `.in` files, copy
the results to `/etc/systemd/system/`, then
`sudo systemctl enable --now vec2projector vec2projector-watchdog.timer`.

`Restart=always` + `RestartSec=2` means a crash gets picked up in two seconds, forever.
`enable` covers reboots and power cuts. The sender is a separate, optional unit
(`vec2projector-sender`) so a restart of your code never blanks the projector.

## 5. Optional hardening for a box nobody will touch

```bash
sudo raspi-config nonint do_boot_wait 0     # don't block boot waiting for network
sudo apt install -y watchdog                # hardware watchdog reboots a wedged Pi
```

For a display that runs for months, consider an overlay filesystem
(`raspi-config` → Performance → Overlay File System) so a yanked power cable
can never corrupt the SD card.

## Checking it worked

```bash
systemctl status vec2projector
journalctl -u vec2projector | grep output:      # expect: output: kmsdrm  1920x1080  (or: output: fb (/dev/fb0) ...)
```

## Pushing images instead of vectors

Three ways in, all of which fill the same fullscreen output:

```bash
# 1. a file the producer keeps overwriting (simplest; any format pygame reads)
vec2projector.py --no-stdin --watch /run/frame.png

# 2. raw rgb24 frames down a pipe (fastest; no encode/decode)
my_generator.py | vec2projector.py --raw 1920x1080
ffmpeg -i clip.mp4 -f rawvideo -pix_fmt rgb24 - | vec2projector.py --raw 1920x1080

# 3. the "image <path>" protocol command, mixed in with vectors
printf 'image /run/frame.png\nv -0.9 0 0.9 0 red 6\n' | vec2projector.py
```

Images draw underneath vectors, so you can overlay one on the other.
`--fit contain|cover|stretch` controls the fill (default letterboxes).

**Writing the watched file:** write to a temp file on the same filesystem and
`os.replace()` it into place. That rename is atomic, so the display can never
catch a half-written frame. Put it on a tmpfs (`/run` or `/dev/shm`) — at any
real frame rate, writing images to the SD card will wear it out.

**Cost.** A 1080p raw frame is 6.2 MB; 60 fps of them is 373 MB/s. That is fine
down a local pipe on one box, and impossible over gigabit ethernet — if frames
must cross a network, send PNG/JPEG or encoded video, or send vectors. A vector
is ~40 bytes, so the vector path is roughly five orders of magnitude cheaper.
Prefer it whenever the source is geometry rather than pixels.
