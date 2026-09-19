#!/usr/bin/env python3
"""
draw.py -- type coordinates, see lines appear on the projector.

Run the display on the projector in one terminal:

    python3 vec2projector.py --display 1 --no-stdin --http 8080

Run this in another (it stays on your laptop screen, so you can see it):

    python3 draw.py

Then just type:

    vec> 0 0 1 1
    vec> -1 0.5 1 0.5 red 8
    vec> undo
    vec> clear
"""

import colorsys
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("VEC2PROJECTOR_URL", "http://127.0.0.1:8080")

HELP = """
  x1 y1 x2 y2 [color] [width]   draw a line  (e.g.  0 0 1 1   or  -1 0 1 0 red 8)
  color <c> / width <n>         defaults for later lines
  bg <color>                    background (black = invisible on a projector)
  grid on | off                 reference grid, handy while aiming
  random [strokes/sec] [tail]   stream random strokes until Ctrl-C (default 20/s, 40 kept)
  list                          show what is currently drawn
  undo                          remove the last line
  clear                         erase everything
  save <file.txt>               write the current drawing out
  load <file.txt>               read a drawing back in
  help / quit
"""


def send(lines):
    """POST protocol lines to the display. Returns None on success, else why not."""
    body = ("\n".join(lines) + "\n").encode()
    req = urllib.request.Request(URL + "/vectors", data=body, method="POST",
                                 headers={"Content-Type": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return None if r.status == 200 else f"display said HTTP {r.status}"
    except urllib.error.URLError as exc:
        return f"cannot reach the display at {URL} ({exc.reason})"
    except (OSError, TimeoutError) as exc:
        return f"cannot reach the display at {URL} ({exc})"


def fetch_strokes():
    """Ask the display what is already on screen, so undo/list survive a restart."""
    try:
        with urllib.request.urlopen(URL + "/vectors", timeout=3) as r:
            return [l for l in r.read().decode().splitlines() if l.strip()]
    except Exception:
        return []


def health():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


def redraw(strokes):
    """Rebuild the screen from scratch -- used by undo and load."""
    return send(["clear"] + strokes if strokes else ["clear"])


def random_stream(rate=20.0, tail=40):
    """Stream a wandering, fading trail of random strokes until Ctrl-C.

    Every tick sends one whole frame (mode frame ... end), so the display swaps
    it in atomically and never shows a half-drawn trail.
    """
    rate = max(0.5, min(rate, 240.0))
    tail = max(2, tail)
    period = 1.0 / rate
    x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    hue = random.random()
    trail = []                          # (x1, y1, x2, y2, hue, width)
    sent = dropped = 0
    t0 = time.time()
    print(f"streaming {rate:g} strokes/sec, {tail} kept on screen -- Ctrl-C to stop")
    try:
        nxt = time.perf_counter()
        while True:
            nx = max(-1.0, min(1.0, x + random.uniform(-0.45, 0.45)))
            ny = max(-1.0, min(1.0, y + random.uniform(-0.45, 0.45)))
            hue = (hue + 0.012) % 1.0
            trail.append((x, y, nx, ny, hue, random.randint(2, 7)))
            x, y = nx, ny
            del trail[:-tail]

            lines = ["mode frame"]
            for i, (a, b, c, d, hu, w) in enumerate(trail):
                age = (i + 1) / len(trail)          # oldest dimmest, newest brightest
                r, g, bl = (int(v * 255) for v in colorsys.hsv_to_rgb(hu, 0.9, 0.25 + 0.75 * age))
                lines.append(f"v {a:.4f} {b:.4f} {c:.4f} {d:.4f} {r},{g},{bl} {w}")
            lines.append("end")

            if send(lines) is None:
                sent += 1
            else:
                dropped += 1
                time.sleep(0.5)                     # display is down: back off, keep trying
            if (sent + dropped) % int(max(rate, 1)) == 0:
                print(f"\r  sent {sent}  dropped {dropped}  "
                      f"{sent / max(time.time() - t0, 1e-9):.1f}/s ", end="", flush=True)

            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    print(f"\nstopped after {sent} frames ({dropped} dropped)")
    send(["mode persist"])              # leave the display ready for typed lines again


def looks_like_coords(s):
    return s[:1].isdigit() or s[:1] in "+-."


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--random", "random"):
        args = sys.argv[2:4]
        try:
            random_stream(*(float(a) if i == 0 else int(a) for i, a in enumerate(args)))
        except ValueError:
            sys.exit("usage: draw.py --random [strokes/sec] [tail]")
        return 0
    try:
        import readline    # arrow-key history and line editing, free
        histfile = os.path.expanduser("~/.vec2projector_history")
        try:
            readline.read_history_file(histfile)
        except OSError:
            pass
        import atexit
        atexit.register(lambda: _save_history(readline, histfile))
    except ImportError:
        pass

    h = health()
    if h is None:
        print(f"! no display answering at {URL}")
        print("!   start it with:  python3 vec2projector.py --display 1 --no-stdin --http 8080")
        print("!   (you can keep typing -- lines will land once it is up)")
    else:
        print(f"connected to the display  ({h['fps']:.0f} fps, up {h['uptime_s']:.0f}s)")
    print("type 'help' for commands, 'quit' to exit")

    strokes = fetch_strokes()
    if strokes:
        print(f"picked up {len(strokes)} line(s) already on screen")
    while True:
        try:
            line = input("vec> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        low = line.lower()
        word = low.split()[0]

        if word in ("quit", "exit", "q"):
            return 0
        if word in ("help", "?", "h"):
            print(HELP)
            continue
        if word == "random":
            nums = line.split()[1:]
            try:
                random_stream(*(float(n) if i == 0 else int(n) for i, n in enumerate(nums[:2])))
            except ValueError:
                print("! usage: random [strokes/sec] [tail]")
            strokes = fetch_strokes()
            continue
        if word == "list":
            if not strokes:
                print("  (nothing drawn)")
            for i, s in enumerate(strokes, 1):
                print(f"  {i:3d}. {s}")
            continue
        if word == "undo":
            if not strokes:
                print("  nothing to undo")
                continue
            gone = strokes.pop()
            err = redraw(strokes)
            print(f"  removed: {gone}" if not err else f"! {err}")
            continue
        if word == "clear":
            strokes = []
            err = send(["clear"])
            print("  cleared" if not err else f"! {err}")
            continue
        if word == "save":
            parts = line.split(None, 1)
            if len(parts) < 2:
                print("! save needs a filename")
                continue
            try:
                with open(parts[1], "w") as f:
                    f.write("\n".join(strokes) + "\n")
                print(f"  wrote {len(strokes)} line(s) to {parts[1]}")
            except OSError as exc:
                print(f"! {exc}")
            continue
        if word == "load":
            parts = line.split(None, 1)
            if len(parts) < 2:
                print("! load needs a filename")
                continue
            try:
                with open(parts[1]) as f:
                    strokes = [l.strip() for l in f if l.strip()]
            except OSError as exc:
                print(f"! {exc}")
                continue
            err = redraw(strokes)
            print(f"  loaded {len(strokes)} line(s)" if not err else f"! {err}")
            continue

        # a coordinate line, or a passthrough setting like "color red" / "grid on"
        payload = ("v " + line) if looks_like_coords(line) else line
        err = send([payload])
        if err:
            print(f"! {err}")
            continue
        if looks_like_coords(line):
            strokes.append(payload)
            print(f"  drew {payload[2:]}   ({len(strokes)} on screen)")
        else:
            print(f"  {line}")


def _save_history(readline, path):
    try:
        readline.set_history_length(1000)
        readline.write_history_file(path)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
