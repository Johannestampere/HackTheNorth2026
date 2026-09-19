#!/usr/bin/env python3
"""
sender.py -- the "server" side: pushes updates to the display process over HTTP.

Run it from a second SSH session (or a systemd service) while vec2projector.py is
running. Standard library only -- nothing to install.

    python3 sender.py clock              # analog clock, sweeping second hand
    python3 sender.py spin               # rotating, colour-cycling star
    python3 sender.py random             # endless wandering trail of random strokes
    python3 sender.py line 0 0 1 1 red 6 # one line, then exit (stays on screen)
    python3 sender.py image photo.png    # show a picture (png/jpg/bmp)
    python3 sender.py clear              # erase everything
    python3 sender.py health             # is the display alive, and how stale is it?

    python3 sender.py clock --fps 30 --seconds 60     # stop after a minute
    python3 sender.py random --fps 40 --tail 100      # faster, longer trail
    python3 sender.py spin --url http://192.168.1.50:8080   # display on another machine
                                                             # (start it with --http-host 0.0.0.0)

Streaming commands send one whole frame per tick, so the display swaps each in at
once and never shows a half-drawn frame. Ctrl-C stops the stream and leaves the
last frame on screen. If the display is down, frames are dropped and it keeps
retrying: a sender must never fall over because the projector restarted.

Coordinates: x and y in [-1, 1], centre 0 0, +y up.
"""

import argparse
import colorsys
import json
import math
import os
import random
import signal
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("VEC2PROJECTOR_URL", "http://127.0.0.1:8080")


class Display:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def post(self, path, body, timeout=3.0):
        """Returns None on success, otherwise a short reason."""
        req = urllib.request.Request(self.url + path, data=body, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return None if r.status == 200 else f"HTTP {r.status}"
        except urllib.error.HTTPError as exc:
            return f"HTTP {exc.code}: {exc.read()[:80].decode(errors='replace').strip()}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return f"cannot reach {self.url} ({getattr(exc, 'reason', exc)})"

    def vectors(self, lines):
        return self.post("/vectors", ("\n".join(lines) + "\n").encode())

    def health(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return json.load(r)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return None


# ------------------------------------------------------------------ frames ----

def _rgb(h, s=0.9, v=1.0):
    return ",".join(str(int(c * 255)) for c in colorsys.hsv_to_rgb(h % 1.0, s, v))


def _seg(a, b, color, width):
    return f"v {a[0]:.4f} {a[1]:.4f} {b[0]:.4f} {b[1]:.4f} {color} {width}"


def _polar(angle, radius):
    return (math.cos(angle) * radius, math.sin(angle) * radius)


def spin_frame(t):
    n, out = 7, []
    for i in range(n):
        a = t * 0.7 + i * 2 * math.pi / n
        b = a + 3 * 2 * math.pi / n
        out.append(_seg(_polar(a, 0.9), _polar(b, 0.9), _rgb(i / n + t * 0.1), 5))
    return out


def clock_frame(_t):
    now = time.time()
    lt = time.localtime(now)
    sec = lt.tm_sec + (now % 1.0)                      # sweeping, not ticking
    minute = lt.tm_min + sec / 60.0
    hour = (lt.tm_hour % 12) + minute / 60.0

    def clockwise(fraction):                            # 0 = 12 o'clock, clockwise
        return math.pi / 2 - 2 * math.pi * fraction

    out = []
    ring = [_polar(2 * math.pi * k / 90, 0.92) for k in range(91)]
    out += [_seg(ring[k], ring[k + 1], "60,90,110", 3) for k in range(90)]
    for k in range(12):
        a = clockwise(k / 12)
        out.append(_seg(_polar(a, 0.80), _polar(a, 0.92), "230,230,230", 6 if k % 3 == 0 else 3))
    out.append(_seg((0, 0), _polar(clockwise(hour / 12), 0.50), "255,255,255", 10))
    out.append(_seg((0, 0), _polar(clockwise(minute / 60), 0.75), "200,220,255", 6))
    out.append(_seg(_polar(clockwise(sec / 60) + math.pi, 0.12),
                    _polar(clockwise(sec / 60), 0.85), "255,60,60", 3))
    return out


def random_frame_factory(tail):
    """A wandering trail: each frame adds one random stroke; the oldest fade out."""
    x, y, hue = random.uniform(-1, 1), random.uniform(-1, 1), random.random()
    trail = []                                          # (x1, y1, x2, y2, hue, width)

    def build(_t):
        nonlocal x, y, hue
        nx = max(-1.0, min(1.0, x + random.uniform(-0.45, 0.45)))
        ny = max(-1.0, min(1.0, y + random.uniform(-0.45, 0.45)))
        hue = (hue + 0.012) % 1.0
        trail.append((x, y, nx, ny, hue, random.randint(2, 7)))
        x, y = nx, ny
        del trail[:-tail]
        return [_seg((a, b), (c, d), _rgb(hu, 0.9, 0.25 + 0.75 * (i + 1) / len(trail)), w)
                for i, (a, b, c, d, hu, w) in enumerate(trail)]      # oldest dimmest
    return build


FRAMES = {"spin": spin_frame, "clock": clock_frame}


def stream(display, build, fps, seconds):
    period = 1.0 / max(1.0, fps)
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.append(1))

    t0 = time.time()
    end = t0 + seconds if seconds else None
    sent = dropped = 0
    last_report, last_err = t0, None
    nxt = time.perf_counter()
    print(f"streaming to {display.url} at {fps:g} fps -- Ctrl-C to stop", flush=True)
    while not stopping and (end is None or time.time() < end):
        err = display.vectors(["mode frame", *build(time.time() - t0), "end"])
        if err is None:
            sent += 1
            last_err = None
        else:
            dropped += 1
            if err != last_err:                        # say it once, not 30 times a second
                print(f"\n! {err} -- dropping frames, will keep trying", flush=True)
                last_err = err
            time.sleep(0.5)
            nxt = time.perf_counter()
        if time.time() - last_report >= 1.0:
            print(f"\r  sent {sent}  dropped {dropped}  ", end="", flush=True)
            last_report = time.time()
        nxt += period
        time.sleep(max(0.0, nxt - time.perf_counter()))
    print(f"\nstopped: {sent} frames sent, {dropped} dropped")
    display.vectors(["mode persist"])                  # leave it ready for one-off commands


# ---------------------------------------------------------------- one-shots ----

def cmd_line(display, a):
    if len(a.args) < 4:
        return "line needs: x1 y1 x2 y2 [color] [width]"
    return display.vectors(["mode persist", "v " + " ".join(a.args)])


def cmd_image(display, a):
    if len(a.args) != 1:
        return "image needs a file path"
    try:
        data = open(a.args[0], "rb").read()
    except OSError as exc:
        return str(exc)
    return display.post("/frame", data, timeout=10)


def cmd_clear(display, _a):
    return display.vectors(["mode persist", "clear"])


def cmd_health(display, _a):
    h = display.health()
    if h is None:
        return f"no display answering at {display.url}"
    print(json.dumps(h, indent=2))
    return None


def main():
    ap = argparse.ArgumentParser(description="Push updates to the projector display.",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["clock", "spin", "random", "line", "image", "clear", "health"])
    ap.add_argument("args", nargs="*", help="arguments for line / image")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"display address (default {DEFAULT_URL})")
    ap.add_argument("--fps", type=float, default=20.0, help="frames per second when streaming")
    ap.add_argument("--tail", type=int, default=40, help="random: strokes kept on screen")
    ap.add_argument("--seconds", type=float, default=0, help="stop streaming after this long (0 = forever)")
    a = ap.parse_args()

    display = Display(a.url)
    if a.command == "random":
        stream(display, random_frame_factory(max(2, a.tail)), a.fps, a.seconds)
        return 0
    if a.command in FRAMES:
        stream(display, FRAMES[a.command], a.fps, a.seconds)
        return 0
    err = {"line": cmd_line, "image": cmd_image, "clear": cmd_clear, "health": cmd_health}[a.command](display, a)
    if err:
        print(f"! {err}", file=sys.stderr)
        return 1
    if a.command != "health":
        print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
