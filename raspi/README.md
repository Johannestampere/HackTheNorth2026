# Raspberry Pi projector display (`raspi/`)

Two small programs. Neither needs a monitor, keyboard or desktop on the Pi.

| Program | Role |
| --- | --- |
| `vec2projector.py` | **The display.** Owns the Pi's HDMI output and draws what it is sent. Runs all the time. |
| `sender.py` | **The updates.** Pushes lines, images or animation to the display over HTTP. Runs whenever there is something to show. |

They are separate processes on purpose: the sender can start, crash or be redeployed and the
projector keeps showing the last frame.

```
sender.py (or your server) --HTTP, 127.0.0.1:8080--> vec2projector.py ==HDMI==> projector
```

## On the Pi, over SSH

```sh
ssh <user>@<pi>
git clone -b raspi-projector-display https://github.com/Johannestampere/HackTheNorth2026.git
cd HackTheNorth2026/raspi

sudo apt install -y python3-pygame       # the apt build; pip's pygame lacks direct-HDMI support
python3 vec2projector.py --diagnose      # tells you what is missing, with the fix for each item
```

If `--diagnose` mentions a running desktop, stop it (it owns the HDMI output):

```sh
sudo systemctl isolate multi-user.target
```

**Start the display** (in `tmux`, or it dies when your SSH session closes):

```sh
python3 vec2projector.py --http 8080 --splash
```

A test card should appear on the projector. There is nothing to click or focus: with no desktop
the display owns the output directly. It first tries SDL's `kmsdrm` driver and, if that is not
available, writes frames straight into `/dev/fb0`. The line `output: ...` says which it used.

**Send it something** from a second SSH session:

```sh
python3 sender.py clock                      # analog clock, sweeping second hand
python3 sender.py spin                       # rotating star
python3 sender.py line 0 0 1 1 red 6         # one line, then exit
python3 sender.py image photo.png
python3 sender.py clear
python3 sender.py health
```

Coordinates run -1..1 with `0 0` at the centre and +y up. `draw.py` is an interactive
version (type coordinates at a prompt; `draw.py --random` streams random strokes).

## Make it permanent

```sh
./install.sh --dry-run      # see what it will do
./install.sh --now          # install, boot to console, start the display now
./install.sh --sender       # also run `sender.py clock` as a service (replace with your own)
```

That installs systemd units so the display starts at boot and restarts if it dies, plus a
watchdog that restarts it if it stops drawing. Details, the HDMI-at-cold-boot fix and
hardening: [`pi5/SETUP-pi5.md`](pi5/SETUP-pi5.md).

## Using it from the rest of this repo

`ProjectionFrame.rgb` (`width * height * 3` RGB8 bytes) is exactly what `POST /raw?w=&h=`
accepts, so a `Projector.project(rgb, w, h)` implementation can forward frames as-is.
`pi5/push_example.py` has the client functions.

## What has and has not been tested

Tested on a laptop: drawing, the HTTP receiver, both outputs' logic, the sender, the
installer's dry run, and the systemd units under `systemd-analyze verify`. The framebuffer
output was tested against an ordinary file standing in for `/dev/fb0`, checking the actual pixel
bytes.

**Not yet run on a Pi 5 with a real HDMI output.** The first real-hardware run is the true test;
if the wall stays black, `--diagnose` and `journalctl -u vec2projector` are the places to look.
