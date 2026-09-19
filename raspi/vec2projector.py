#!/usr/bin/env python3
"""
vec2projector.py -- stream vectors to a projector over HDMI, in real time.

Opens a fullscreen pygame window on the display you pick (the HDMI output),
then draws line segments as fast as they arrive on stdin or a UDP socket.

QUICK START
    # see which display index the projector is
    python3 vec2projector.py --list-displays

    # rotating demo on display 1 (the HDMI one)
    python3 vec2projector.py --display 1 --demo

    # feed it by hand (type lines, press enter)
    python3 vec2projector.py --display 1
    0 0 1 1
    v -1 0.5 1 0.5 #ff0000 6
    clear

    # feed it from another program
    python3 my_generator.py | python3 vec2projector.py --display 1

    # feed it over the network / from another process
    python3 vec2projector.py --display 1 --udp 9000

    # headless Raspberry Pi wired to the projector (no desktop running):
    # tries SDL's kmsdrm driver, then falls back to writing /dev/fb0 directly.
    python3 vec2projector.py --http 8080 --splash
    python3 vec2projector.py --diagnose      # what is wrong, if it will not come up
    # See README.md in this folder and pi5/SETUP-pi5.md.

WIRE PROTOCOL  (one command per line, ASCII, case-insensitive keywords)
    x1 y1 x2 y2 [color] [width]     draw a vector (the "v" is optional)
    v x1 y1 x2 y2 [color] [width]   same thing, explicit
    image   <path>                  show an image file, fitted to the screen
    image   off                     drop the image again
    color   <color>                 set default color for later vectors
    width   <n>                     set default line width in pixels
    bg      <color>                 set background color
    mode    persist | frame         see below
    end                             (frame mode) commit the frame now
    clear                           erase everything
    grid    on | off                toggle the reference grid
    quit                            exit

    <color> is #rgb, #rrggbb, "r,g,b", or a name like red / cyan / white.

    persist mode (default): vectors stay on screen until "clear".
    frame mode: vectors buffer up and only appear when you send "end",
                so each frame is drawn whole -- use this for animation.

COORDINATES
    By default the world is x,y in [-1, 1] with +y UP (math convention),
    fitted to the screen with letterboxing so nothing is distorted.
    Change it with --space X0 Y0 X1 Y1, e.g. --space 0 0 1920 1080 --y-down.

KEYS
    ESC / q  quit      f  toggle fullscreen      g  grid      c  clear
    s        save a PNG snapshot next to the script
"""

import argparse
import atexit
import fcntl
import io
import json
import math
import mmap
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# A fullscreen output should stay up when the desktop moves focus elsewhere.
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
try:
    import pygame
except ImportError:
    sys.exit("pygame is required:  pip install pygame   (or: sudo apt install python3-pygame)")


# ---------------------------------------------------------------- colors ----

NAMED = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
    "green": (0, 255, 0), "blue": (0, 80, 255), "yellow": (255, 255, 0),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "orange": (255, 140, 0),
    "lime": (160, 255, 0), "grey": (128, 128, 128), "gray": (128, 128, 128),
}


def parse_color(tok, fallback=(255, 255, 255)):
    """'#0f0' / '#00ff00' / '0,255,0' / 'green' -> (r, g, b)."""
    if not tok:
        return fallback
    t = tok.strip().lower()
    if t in NAMED:
        return NAMED[t]
    if t.startswith("#"):
        h = t[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return fallback
        return fallback
    if "," in t:
        try:
            r, g, b = (int(float(p)) for p in t.split(",")[:3])
            return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        except ValueError:
            return fallback
    return fallback


# ----------------------------------------------------------------- state ----

class Latest:
    """A one-slot mailbox: a slow display drops stale frames instead of lagging."""

    def __init__(self):
        self.v = None
        self.lock = threading.Lock()

    def put(self, v):
        with self.lock:
            self.v = v

    def take(self):
        with self.lock:
            v, self.v = self.v, None
            return v


class ImageLayer:
    """Holds the current frame and its scaled-to-screen version (cached)."""

    def __init__(self, fit="contain"):
        self.src = None
        self.fit = fit
        self._key = None
        self._scaled = None
        self._pos = (0, 0)

    def set(self, surf):
        self.src = surf
        self._key = None

    def clear(self):
        self.src = None
        self._key = None
        self._scaled = None

    def blit(self, screen):
        if self.src is None:
            return
        sw, sh = screen.get_size()
        key = (id(self.src), sw, sh, self.fit)
        if key != self._key:
            self._scaled, self._pos = self._rescale(sw, sh)
            self._key = key
        if self._scaled is not None:
            screen.blit(self._scaled, self._pos)

    def _rescale(self, sw, sh):
        iw, ih = self.src.get_size()
        if not iw or not ih:
            return None, (0, 0)
        if self.fit == "stretch":
            tw, th = sw, sh
        else:
            f = min(sw / iw, sh / ih) if self.fit == "contain" else max(sw / iw, sh / ih)
            tw, th = max(1, int(iw * f)), max(1, int(ih * f))
        scaled = (self.src if (tw, th) == (iw, ih)
                  else pygame.transform.smoothscale(self.src, (tw, th)))
        return scaled.convert(), ((sw - tw) // 2, (sh - th) // 2)



class Scene:
    """Everything that is currently on (or about to be on) the screen."""

    def __init__(self, color, width, bg, cap=200000):
        self.live = []          # list of (x1, y1, x2, y2, color, width)
        self.pending = []       # frame-mode staging buffer
        self.cap = cap          # hard ceiling, so a 24/7 run cannot eat all the RAM
        self.mode = "persist"
        self.color = color
        self.width = width
        self.bg = bg
        self.grid = False
        self.dirty = True       # something changed -> redraw
        self.image_req = Latest()   # path (or None to drop); loaded on the main thread
        self.lock = threading.Lock()
        self.frames = 0         # committed frames, for the HUD

    def add(self, seg):
        with self.lock:
            buf = self.pending if self.mode == "frame" else self.live
            buf.append(seg)
            if self.cap and len(buf) > self.cap:
                del buf[:len(buf) - self.cap]      # drop the oldest; never grow forever
            if self.mode != "frame":
                self.dirty = True

    def commit(self):
        with self.lock:
            self.live, self.pending = self.pending, []
            self.frames += 1
            self.dirty = True

    def clear(self):
        with self.lock:
            self.live, self.pending = [], []
            self.dirty = True
        self.image_req.put(("drop", None))

    def snapshot(self):
        with self.lock:
            return list(self.live), self.bg, self.grid


# --------------------------------------------------------------- parsing ----

def handle_line(line, scene, log):
    """Apply one protocol line to the scene. Returns False to quit."""
    line = line.strip()
    if not line or line[0] in "#;":
        return True
    parts = line.replace(",", " ").split() if _is_bare_numbers(line) else line.split()
    head = parts[0].lower()

    if head in ("quit", "exit"):
        return False
    if head == "clear":
        scene.clear()
        return True
    if head == "end" or head == "flush":
        scene.commit()
        return True
    if head == "mode" and len(parts) > 1:
        m = parts[1].lower()
        if m in ("persist", "frame"):
            with scene.lock:
                scene.mode = m
                scene.pending = []
        return True
    if head == "image" and len(parts) > 1:
        arg = line.split(None, 1)[1].strip()
        if arg.lower() in ("off", "none", "clear"):
            scene.image_req.put(("drop", None))
        else:
            scene.image_req.put(("load", arg))
        return True
    if head == "color" and len(parts) > 1:
        scene.color = parse_color(parts[1], scene.color)
        return True
    if head == "width" and len(parts) > 1:
        try:
            scene.width = max(1, int(float(parts[1])))
        except ValueError:
            pass
        return True
    if head == "bg" and len(parts) > 1:
        scene.bg = parse_color(parts[1], scene.bg)
        scene.dirty = True
        return True
    if head == "grid" and len(parts) > 1:
        scene.grid = parts[1].lower() in ("on", "1", "true", "yes")
        scene.dirty = True
        return True

    # a vector, with or without the leading "v"
    if head in ("v", "vec", "line", "seg"):
        parts = parts[1:]
    if len(parts) < 4:
        log(f"ignored: {line!r}")
        return True
    try:
        x1, y1, x2, y2 = (float(p) for p in parts[:4])
    except ValueError:
        log(f"bad numbers: {line!r}")
        return True

    col, wid = scene.color, scene.width
    rest = parts[4:]
    if rest:
        col = parse_color(rest[0], col)
    if len(rest) > 1:
        try:
            wid = max(1, int(float(rest[1])))
        except ValueError:
            pass
    scene.add((x1, y1, x2, y2, col, wid))
    return True


def _is_bare_numbers(line):
    return line[0].isdigit() or line[0] in "+-."


# ---------------------------------------------------------------- inputs ----

def stdin_reader(q, stop):
    """Line reader over the raw fd.

    Deliberately not `for line in sys.stdin`: this is a daemon thread, and one
    parked inside Python's buffered reader at interpreter shutdown aborts the
    process with "_enter_buffered_busy" instead of exiting cleanly.
    """
    fd = sys.stdin.fileno()
    buf = b""
    while not stop.is_set():
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for line in lines:
            q.put(line.decode("utf-8", "replace"))
    if buf.strip():
        q.put(buf.decode("utf-8", "replace"))
    q.put(None)   # EOF marker: the feed ended, but keep displaying


def udp_reader(q, stop, host, port, log):
    """Rebuilds its socket after any error -- an unattended box must not go deaf."""
    while not stop.is_set():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.settimeout(0.25)
            while not stop.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                for line in data.decode("utf-8", "replace").splitlines():
                    q.put(line)
        except OSError as exc:
            if not stop.is_set():
                log(f"udp socket error ({exc}); retrying in 2s")
                time.sleep(2)
        finally:
            if sock is not None:
                sock.close()


def watch_reader(scene, stop, path, interval, log):
    """Re-show an image file whenever it changes on disk.

    Have the producer write to a temp file and os.replace() it into place --
    that swap is atomic, so a half-written frame can never be displayed.
    """
    last = None
    log(f"watching {path} for new frames")
    while not stop.is_set():
        try:
            st = os.stat(path)
            sig = (st.st_mtime_ns, st.st_size)
            if sig != last and st.st_size > 0:
                last = sig
                scene.image_req.put(("load", path))
        except FileNotFoundError:
            pass
        except OSError as exc:
            log(f"watch error: {exc}")
        time.sleep(interval)


def raw_reader(scene, stop, size, log):
    """Read raw RGB24 frames of a fixed size from stdin (ffmpeg, numpy, OpenCV).

        ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgb24 - \
          | vec2projector.py --raw 1920x1080
    """
    w, h = size
    nbytes = w * h * 3
    fd = sys.stdin.fileno()          # raw fd, not sys.stdin.buffer -- see stdin_reader
    log(f"reading raw rgb24 {w}x{h} frames from stdin ({nbytes} bytes each)")
    while not stop.is_set():
        chunks, got = [], 0
        while got < nbytes:          # a pipe hands over at most 64k at a time
            try:
                c = os.read(fd, nbytes - got)
            except OSError:
                c = b""
            if not c:
                break
            chunks.append(c)
            got += len(c)
        if got < nbytes:
            log("raw stdin ended; holding the last frame")
            return
        buf = b"".join(chunks) if len(chunks) > 1 else chunks[0]
        try:
            scene.image_req.put(("surface", pygame.image.frombytes(buf, (w, h), "RGB")))
        except (ValueError, pygame.error) as exc:
            log(f"bad raw frame: {exc}")
            return


MAX_POST = 64 * 1024 * 1024          # refuse absurd bodies rather than OOM the Pi


def http_reader(scene, q, stop, host, port, stats, log):
    """A tiny HTTP receiver, so a server process can push frames to the display.

        POST /frame              body = png/jpeg/bmp bytes
        POST /raw?w=1920&h=1080  body = raw rgb24 pixels
        POST /vectors            body = protocol lines, one per line
        GET  /health             json: is it alive, how stale is the picture

    Binds loopback by default: this is process-to-process IPC on one box, not a
    public service. Images are decoded here rather than on the draw thread, so a
    slow PNG decode cannot stutter the output.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass                          # do not spam the journal on every frame

        def _read_body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return None
            if n > MAX_POST:
                return None
            chunks, got = [], 0
            while got < n:
                c = self.rfile.read(min(65536, n - got))
                if not c:
                    return None
                chunks.append(c)
                got += len(c)
            return b"".join(chunks)

        def _reply(self, code, payload=b"ok", ctype="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except OSError:
                pass

        def do_GET(self):
            route = urlparse(self.path).path.rstrip("/")
            if route in ("/health", "/healthz"):
                self._reply(200, json.dumps(stats.snapshot()).encode(), "application/json")
            elif route in ("/vectors", "/v"):
                # what is on screen right now, as protocol lines -- lets a console
                # reconnect and still know what it is editing
                out = []
                for x1, y1, x2, y2, col, wid in scene.snapshot()[0]:
                    out.append(f"v {x1:g} {y1:g} {x2:g} {y2:g} "
                               f"{col[0]},{col[1]},{col[2]} {wid}")
                self._reply(200, ("\n".join(out) + "\n").encode() if out else b"")
            else:
                self._reply(404, b"try POST /frame, /raw, /vectors or GET /health\n")

        def do_POST(self):
            u = urlparse(self.path)
            route = u.path.rstrip("/") or "/"
            body = self._read_body()
            if body is None:
                return self._reply(400, b"empty or oversized body\n")

            if route in ("/frame", "/image", "/"):
                try:
                    surf = pygame.image.load(io.BytesIO(body))
                except (pygame.error, OSError) as exc:
                    return self._reply(415, f"undecodable image: {exc}\n".encode())
                scene.image_req.put(("surface", surf))
                return self._reply(200)

            if route == "/raw":
                qs = parse_qs(u.query)
                try:
                    w = int(qs["w"][0])
                    h = int(qs["h"][0])
                except (KeyError, IndexError, ValueError):
                    return self._reply(400, b"/raw needs ?w=<px>&h=<px>\n")
                if w * h * 3 != len(body):
                    return self._reply(
                        400, f"expected {w * h * 3} bytes, got {len(body)}\n".encode())
                try:
                    scene.image_req.put(("surface", pygame.image.frombytes(body, (w, h), "RGB")))
                except (ValueError, pygame.error) as exc:
                    return self._reply(400, f"bad raw frame: {exc}\n".encode())
                return self._reply(200)

            if route in ("/vectors", "/v"):
                for line in body.decode("utf-8", "replace").splitlines():
                    q.put(line)
                return self._reply(200)

            return self._reply(404, b"no such route\n")

        do_PUT = do_POST

    while not stop.is_set():
        srv = None
        try:
            srv = ThreadingHTTPServer((host, port), Handler)
            srv.daemon_threads = True
            srv.timeout = 0.5
            log(f"http receiver on http://{host}:{port}  (POST /frame, GET /health)")
            while not stop.is_set():
                srv.handle_request()
        except OSError as exc:
            if not stop.is_set():
                log(f"http server error ({exc}); retrying in 2s")
                time.sleep(2)
        finally:
            if srv is not None:
                srv.server_close()


class Stats:
    """What /health reports. An unattended box needs to be askable if it is OK."""

    def __init__(self):
        self.lock = threading.Lock()
        self.start = time.time()
        self.fps = 0.0
        self.vectors = 0
        self.images = 0
        self.last_image = None

    def update(self, fps, vectors, images, last_image):
        with self.lock:
            self.fps, self.vectors = fps, vectors
            self.images, self.last_image = images, last_image

    def snapshot(self):
        with self.lock:
            now = time.time()
            return {
                "ok": self.fps > 1.0,
                "pid": os.getpid(),
                "uptime_s": round(now - self.start, 1),
                "fps": round(self.fps, 1),
                "vectors": self.vectors,
                "images_shown": self.images,
                "last_image_age_s": (None if self.last_image is None
                                     else round(now - self.last_image, 2)),
            }


def demo_feed(q, stop):
    """A spinning star, so you can aim and focus the projector with no feed."""
    t0 = time.time()
    while not stop.is_set():
        t = time.time() - t0
        q.put("mode frame")
        for i in range(7):
            a = t * 0.7 + i * 2 * math.pi / 7
            b = a + 3 * 2 * math.pi / 7
            hue = (i / 7.0 + t * 0.1) % 1.0
            r, g, bl = _hsv(hue, 0.85, 1.0)
            q.put(f"v {math.cos(a):.4f} {math.sin(a):.4f} "
                  f"{math.cos(b):.4f} {math.sin(b):.4f} {r},{g},{bl} 4")
        q.put("end")
        time.sleep(1 / 60)


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, qq, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (qq, v, p), (p, v, t),
               (p, qq, v), (t, p, v), (v, p, qq)][i]
    return int(r * 255), int(g * 255), int(b * 255)


# -------------------------------------------------------------- geometry ----

class Mapper:
    """World coords -> screen pixels, aspect-preserving, letterboxed."""

    def __init__(self, space, size, y_down):
        self.set_space(space, y_down)
        self.resize(size)

    def set_space(self, space, y_down):
        self.x0, self.y0, self.x1, self.y1 = space
        self.y_down = y_down

    def resize(self, size):
        w, h = size
        self.sw, self.sh = w, h
        wx = abs(self.x1 - self.x0) or 1.0
        wy = abs(self.y1 - self.y0) or 1.0
        self.scale = min(w / wx, h / wy)
        self.ox = (w - wx * self.scale) / 2.0
        self.oy = (h - wy * self.scale) / 2.0

    def to_px(self, x, y):
        px = self.ox + (x - min(self.x0, self.x1)) * self.scale
        ty = (y - min(self.y0, self.y1)) * self.scale
        py = self.oy + ty if self.y_down else (self.sh - self.oy - ty)
        return (int(px), int(py))


def draw_grid(surf, mp, color=(38, 38, 46)):
    lo_x, hi_x = sorted((mp.x0, mp.x1))
    lo_y, hi_y = sorted((mp.y0, mp.y1))
    step = _nice_step(max(hi_x - lo_x, hi_y - lo_y))
    n = math.floor(lo_x / step)
    while n * step <= hi_x:
        x = n * step
        pygame.draw.line(surf, color, mp.to_px(x, lo_y), mp.to_px(x, hi_y), 1)
        n += 1
    n = math.floor(lo_y / step)
    while n * step <= hi_y:
        y = n * step
        pygame.draw.line(surf, color, mp.to_px(lo_x, y), mp.to_px(hi_x, y), 1)
        n += 1
    axis = (70, 70, 84)
    if lo_x <= 0 <= hi_x:
        pygame.draw.line(surf, axis, mp.to_px(0, lo_y), mp.to_px(0, hi_y), 2)
    if lo_y <= 0 <= hi_y:
        pygame.draw.line(surf, axis, mp.to_px(lo_x, 0), mp.to_px(hi_x, 0), 2)


def _nice_step(span):
    raw = span / 10.0
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def draw_arrow(surf, color, p1, p2, width):
    pygame.draw.line(surf, color, p1, p2, width)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = math.hypot(dx, dy)
    if ln < 1:
        return
    ux, uy = dx / ln, dy / ln
    size = max(8, width * 3.5)
    bx, by = p2[0] - ux * size, p2[1] - uy * size
    nx, ny = -uy * size * 0.45, ux * size * 0.45
    pygame.draw.polygon(surf, color,
                        [p2, (bx + nx, by + ny), (bx - nx, by - ny)])


# ---------------------------------------------------------------- display ----

def pick_video_driver(choice, log):
    """Choose an SDL video driver.

    On a Raspberry Pi with no desktop running there is no X and no Wayland, so
    SDL must talk to the HDMI connector directly: that is the 'kmsdrm' driver.
    It needs the user to be in the 'video' and 'render' groups, and no desktop
    compositor holding the display.
    """
    if choice and choice != "auto":
        os.environ["SDL_VIDEODRIVER"] = choice
        return
    if os.environ.get("SDL_VIDEODRIVER"):
        return                                  # caller already decided
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"):
        return                                  # a desktop session exists; let SDL pick
    os.environ["SDL_VIDEODRIVER"] = "kmsdrm"
    log("no desktop session detected -> SDL_VIDEODRIVER=kmsdrm (direct to HDMI)")


def init_display(log):
    try:
        pygame.display.init()
    except pygame.error as exc:
        if os.environ.get("SDL_VIDEODRIVER") == "kmsdrm":
            log(f"kmsdrm unavailable ({exc}); falling back to SDL's default driver")
            del os.environ["SDL_VIDEODRIVER"]
            pygame.display.init()
        else:
            raise
    pygame.font.init()          # display + font only -- no mixer to fail on a headless Pi


def open_screen(args, disp, log):
    """Get a surface, degrading gracefully instead of dying on an odd driver."""
    if args.windowed:
        attempts = [(tuple(args.size), pygame.RESIZABLE, 1),
                    (tuple(args.size), pygame.RESIZABLE, 0)]
    else:
        fs = pygame.FULLSCREEN
        attempts = [((0, 0), fs | pygame.SCALED, 1), ((0, 0), fs | pygame.SCALED, 0),
                    ((0, 0), fs, 1), ((0, 0), fs, 0),
                    (tuple(args.size), 0, 0)]
    last = None
    for size, flags, vsync in attempts:
        try:
            return pygame.display.set_mode(size, flags, display=disp, vsync=vsync)
        except pygame.error as exc:
            last = exc
    raise RuntimeError(f"could not open a display: {last}")


FBIOGET_VSCREENINFO = 0x4600
KDSETMODE, KD_TEXT, KD_GRAPHICS = 0x4B3A, 0, 1


class FbSink:
    """Write finished frames straight into a Linux framebuffer (/dev/fb0).

    The headless fallback. It needs no X, no Wayland, no OpenGL and no DRM master
    -- only write access to the device (membership of the 'video' group). We draw
    into an ordinary off-screen pygame surface and copy its bytes across.
    """

    def __init__(self, path, size_override=None):
        self.path = path
        self.fd = os.open(path, os.O_RDWR)
        try:
            self._geometry(size_override)
            self.mm = mmap.mmap(self.fd, self.stride * self.height,
                                mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception:
            os.close(self.fd)
            raise

    def _geometry(self, override):
        if override:
            self.width, self.height = override
            bpp, offs = 32, (16, 8, 0)
            self.stride = self.width * 4
        else:
            raw = fcntl.ioctl(self.fd, FBIOGET_VSCREENINFO, bytes(160))
            (xres, yres, xv, _yv, _xo, _yo, bpp, _gray,
             ro, _rl, _rm, go, _gl, _gm, bo, *_rest) = struct.unpack("20I", raw[:80])
            self.width, self.height, offs = xres, yres, (ro, go, bo)
            sysfs = f"/sys/class/graphics/{os.path.basename(self.path)}/stride"
            try:
                self.stride = int(open(sysfs).read())
            except (OSError, ValueError):
                self.stride = xv * bpp // 8
        if bpp != 32:
            raise RuntimeError(f"{self.path} is {bpp} bits per pixel; only 32 is supported")
        if offs == (16, 8, 0):
            self.fmt = "BGRA"           # memory order B,G,R,X -- the usual XRGB8888
        elif offs == (0, 8, 16):
            self.fmt = "RGBA"
        else:
            raise RuntimeError(f"{self.path} has an unsupported channel layout {offs}")

    @property
    def size(self):
        return (self.width, self.height)

    def write(self, surf):
        data = pygame.image.tobytes(surf, self.fmt)
        row = self.width * 4
        if self.stride == row:
            self.mm[0:len(data)] = data
        else:                            # padded scanlines: copy row by row
            mv = memoryview(data)
            for y in range(self.height):
                self.mm[y * self.stride:y * self.stride + row] = mv[y * row:(y + 1) * row]

    def close(self):
        try:
            self.mm.close()
            os.close(self.fd)
        except (OSError, ValueError):
            pass


class Console:
    """Put the active virtual terminal into graphics mode so the kernel console
    stops drawing its cursor and text over our frames.

    Only permitted for a process that has that tty as its controlling terminal
    (the systemd unit arranges this) or for root, so over a bare SSH login it
    warns and carries on -- a blinking cursor is then the worst that happens.
    """

    def __init__(self, log):
        self.fd = None
        try:
            try:
                tty = open("/sys/class/tty/tty0/active").read().strip() or "tty1"
            except OSError:
                tty = "tty1"
            self.fd = os.open("/dev/" + tty, os.O_RDWR | os.O_NOCTTY)
            fcntl.ioctl(self.fd, KDSETMODE, KD_GRAPHICS)
            atexit.register(self.restore)
            log(f"console {tty} switched to graphics mode (no cursor or text over the picture)")
        except OSError as exc:
            log(f"note: could not silence the text console ({exc}); a blinking cursor may show "
                f"in the picture. The systemd service avoids this. Or run: "
                f"sudo sh -c 'echo 0 > /sys/class/graphics/fbcon/cursor_blink'")
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def restore(self):
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, KDSETMODE, KD_TEXT)
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


def _open_sdl(args, driver, log):
    pygame.display.quit()
    if driver:
        os.environ["SDL_VIDEODRIVER"] = driver
    else:
        os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    n = max(1, len(pygame.display.get_desktop_sizes()))
    disp = args.display
    if disp >= n:
        log(f"display {disp} not found ({n} attached); using 0")
        disp = 0
    return open_screen(args, disp, log), None


def _open_fb(args, log):
    override = None
    if args.fb_size:
        try:
            override = tuple(int(v) for v in args.fb_size.lower().split("x"))
            assert len(override) == 2
        except (ValueError, AssertionError):
            raise RuntimeError(f"--fb-size wants WxH (got {args.fb_size!r})")
    sink = FbSink(args.fb, override)
    os.environ["SDL_VIDEODRIVER"] = "dummy"          # draw off-screen; we present the bytes
    pygame.display.quit()
    pygame.display.init()
    return pygame.display.set_mode(sink.size), sink


def open_output(args, log):
    """Bring up the best output available -> (screen, FbSink or None).

    With a desktop session, SDL's default driver. Headless, direct-to-HDMI via
    kmsdrm first, then the framebuffer, and an explicit reason for every miss.
    """
    headless = not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))
    forced = args.driver if args.driver != "auto" else os.environ.get("SDL_VIDEODRIVER")
    if forced:
        candidates = [forced]
    elif headless:
        candidates = ["kmsdrm", "fb"]
    else:
        candidates = [None]

    misses = []
    for cand in candidates:
        name = cand or "default"
        try:
            screen, sink = _open_fb(args, log) if cand == "fb" else _open_sdl(args, cand, log)
        except (RuntimeError, OSError, pygame.error) as exc:
            misses.append(f"  {name}: {exc}")
            log(f"output '{name}' unavailable: {exc}")
            continue
        log(f"output: {name}" + (f" ({args.fb})" if sink else "")
            + f"  {screen.get_size()[0]}x{screen.get_size()[1]}")
        return screen, sink
    raise SystemExit("no usable display output:\n" + "\n".join(misses)
                     + "\n\nRun  python3 " + os.path.basename(sys.argv[0])
                     + " --diagnose  to see why.")


def draw_splash(screen, font, lines):
    """A test card, so a headless install can tell you it is alive."""
    w, h = screen.get_size()
    m = int(min(w, h) * 0.03)
    pygame.draw.line(screen, (30, 50, 70), (m, m), (w - m, h - m), 2)
    pygame.draw.line(screen, (30, 50, 70), (w - m, m), (m, h - m), 2)
    pygame.draw.rect(screen, (0, 200, 140), (m, m, w - 2 * m, h - 2 * m), 4)
    rendered = [font.render(t, True, (235, 235, 235)) for t in lines]
    bw = max(r.get_width() for r in rendered) + 60
    bh = sum(r.get_height() + 10 for r in rendered) + 40
    pygame.draw.rect(screen, (0, 0, 0), ((w - bw) // 2, (h - bh) // 2, bw, bh))
    y = (h - bh) // 2 + 20
    for r in rendered:
        screen.blit(r, ((w - r.get_width()) // 2, y))
        y += r.get_height() + 10


def diagnose():
    """Print what stands between this machine and a picture on HDMI."""
    import glob
    import grp
    import pwd
    problems = []

    def ok(m):
        print(f"  [ok]   {m}")

    def warn(m):
        print(f"  [warn] {m}")

    def bad(m, fix=None):
        print(f"  [FAIL] {m}")
        problems.append(fix or m)
        if fix:
            print(f"         fix: {fix}")

    def rd(path):
        try:
            return open(path).read().strip()
        except OSError:
            return None

    user = pwd.getpwuid(os.getuid()).pw_name
    groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    print(f"vec2projector --diagnose   (user {user})\n\n1. permissions")
    for g in ("video", "render"):
        if g in groups:
            ok(f"in group '{g}'")
        else:
            bad(f"not in group '{g}'",
                f"sudo usermod -aG video,render,tty,input {user}   then log out and back in")

    print("\n2. desktop / compositor (it would own the HDMI output)")
    comps = {"Xorg", "Xwayland", "labwc", "wayfire", "weston", "sway", "gnome-shell",
             "kwin_wayland", "kwin_x11", "mutter", "lightdm", "gdm3", "sddm"}
    found = set()
    for d in glob.glob("/proc/[0-9]*/comm"):
        name = rd(d)
        if name in comps:
            found.add(name)
    if found:
        bad(f"running: {', '.join(sorted(found))}",
            "sudo systemctl isolate multi-user.target   (this boot)   "
            "and   sudo systemctl set-default multi-user.target   (permanent)")
    else:
        ok("no desktop or compositor running")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        warn("this shell has DISPLAY/WAYLAND_DISPLAY set, so 'auto' will use the desktop path; "
             "add --driver kmsdrm or --driver fb to force direct output")

    print("\n3. HDMI connectors")
    seen = False
    for st in sorted(glob.glob("/sys/class/drm/card*-*/status")):
        seen = True
        name = st.split("/")[-2]
        status = rd(st)
        mode = (rd(os.path.dirname(st) + "/modes") or "").splitlines()
        line = f"{name}: {status}" + (f", preferred {mode[0]}" if mode and status == "connected" else "")
        (ok if status == "connected" else warn)(line)
    if not seen:
        bad("no DRM connectors found", "is the KMS driver enabled? (dtoverlay=vc4-kms-v3d)")
    elif not any(rd(x) == "connected" for x in glob.glob("/sys/class/drm/card*-HDMI*/status")):
        warn("no HDMI connector is 'connected': projector off, wrong port, or no EDID at boot. "
             "See SETUP-pi5.md section 3 (video=HDMI-A-1:1920x1080M@60D).")

    print("\n4. device access")
    for node in sorted(glob.glob("/dev/dri/card*")):
        (ok if os.access(node, os.R_OK | os.W_OK) else bad)(
            f"{node} {'read/write' if os.access(node, os.R_OK | os.W_OK) else 'not accessible'}")
    fbs = sorted(glob.glob("/dev/fb[0-9]*"))
    if not fbs:
        warn("no /dev/fb0 -- the framebuffer fallback is unavailable; kmsdrm is the only path")
    for fb in fbs:
        base = os.path.basename(fb)
        info = (f"{rd(f'/sys/class/graphics/{base}/virtual_size')} px, "
                f"{rd(f'/sys/class/graphics/{base}/bits_per_pixel')} bpp")
        if os.access(fb, os.W_OK):
            ok(f"{fb} writable ({info})")
        else:
            bad(f"{fb} not writable ({info})", f"add {user} to group 'video' (see above)")

    print("\n5. pygame / SDL")
    try:
        ver = f"pygame {pygame.version.ver}, SDL {'.'.join(map(str, pygame.get_sdl_version()))}"
        ok(ver)
        if "/.local/" in (pygame.__file__ or "") or "site-packages" in (pygame.__file__ or ""):
            warn("pygame comes from pip; pip wheels usually lack the kmsdrm driver. "
                 "On a Pi prefer:  sudo apt install python3-pygame   (and pip uninstall pygame)")
    except Exception as exc:
        bad(f"pygame not importable: {exc}", "sudo apt install python3-pygame")
    probe = ("import os;os.environ['SDL_VIDEODRIVER']='kmsdrm';import pygame;"
             "pygame.display.init();print(pygame.display.get_driver())")
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            ok("SDL kmsdrm driver initialises")
        else:
            last = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
            warn(f"SDL kmsdrm unavailable ({last}) -- expected if a display service already "
                 f"holds the output; otherwise the framebuffer fallback will be used")
    except (OSError, subprocess.TimeoutExpired) as exc:
        warn(f"could not probe kmsdrm: {exc}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) above. Fix the [FAIL] lines first.")
        return 1
    print("Nothing blocking found. Start it with:  python3 vec2projector.py --http 8080 --splash")
    return 0


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(
        description="Stream vectors to an HDMI projector in real time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Protocol:  x1 y1 x2 y2 [color] [width]  |  clear | end | mode frame | quit")
    ap.add_argument("--display", type=int, default=0,
                    help="display index for the projector (see --list-displays)")
    ap.add_argument("--list-displays", action="store_true",
                    help="print attached displays and exit")
    ap.add_argument("--windowed", action="store_true", help="run in a window, not fullscreen")
    ap.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=(1280, 720),
                    help="window size when --windowed (default 1280 720)")
    ap.add_argument("--space", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=(-1, -1, 1, 1), help="world coordinate bounds")
    ap.add_argument("--y-down", action="store_true",
                    help="screen convention: +y points down (default is +y up)")
    ap.add_argument("--color", default="#00ff9c", help="default vector color")
    ap.add_argument("--bg", default="#000000", help="background color")
    ap.add_argument("--width", type=int, default=3, help="default line width in px")
    ap.add_argument("--arrows", action="store_true", help="draw an arrowhead at the end point")
    ap.add_argument("--aa", action="store_true", help="antialias 1px lines (slower, prettier)")
    ap.add_argument("--fps", type=int, default=60, help="redraw cap (default 60)")
    ap.add_argument("--udp", type=int, metavar="PORT", help="also listen for vectors on UDP")
    ap.add_argument("--udp-host", default="0.0.0.0", help="UDP bind address")
    ap.add_argument("--no-stdin", action="store_true", help="do not read stdin")
    ap.add_argument("--demo", action="store_true", help="built-in animation, no input needed")
    ap.add_argument("--watch", metavar="PATH",
                    help="show this image file and re-show it every time it changes")
    ap.add_argument("--watch-interval", type=float, default=0.05,
                    help="how often to stat the watched file, seconds (default 0.05)")
    ap.add_argument("--raw", metavar="WxH",
                    help="read raw rgb24 frames of this size from stdin (e.g. 1280x720)")
    ap.add_argument("--fit", choices=("contain", "cover", "stretch"), default="contain",
                    help="how images fill the screen (default contain = letterbox)")
    ap.add_argument("--http", type=int, metavar="PORT",
                    help="accept pushed frames: POST /frame, /raw, /vectors; GET /health")
    ap.add_argument("--http-host", default="127.0.0.1",
                    help="http bind address (default loopback -- same-box IPC only)")
    ap.add_argument("--hud", action="store_true", help="overlay fps / vector count")
    ap.add_argument("--grid", action="store_true", help="start with the reference grid on")
    ap.add_argument("--driver", default="auto",
                    help="auto (default): desktop -> SDL default; headless -> kmsdrm then fb. "
                         "Or force one: kmsdrm, fb, wayland, x11, dummy")
    ap.add_argument("--fb", default="/dev/fb0", help="framebuffer device for the fb output")
    ap.add_argument("--fb-size", metavar="WxH",
                    help="skip querying the framebuffer and assume WxH 32bpp (testing only)")
    ap.add_argument("--splash", action="store_true",
                    help="show a test card until the first frame or vector arrives")
    ap.add_argument("--diagnose", action="store_true",
                    help="check permissions, desktop, HDMI and SDL, then exit")
    ap.add_argument("--max-vectors", type=int, default=200000,
                    help="cap on retained vectors, so a long unattended run stays bounded")
    ap.add_argument("--quiet", action="store_true", help="silence parse warnings")
    args = ap.parse_args()
    log = (lambda m: None) if args.quiet else (lambda m: print(m, file=sys.stderr, flush=True))

    if args.diagnose:
        return diagnose()

    if args.list_displays:
        pick_video_driver(args.driver, log)
        init_display(log)
        sizes = pygame.display.get_desktop_sizes()
        print(f"{len(sizes)} display(s):")
        for i, (w, h) in enumerate(sizes):
            print(f"  --display {i}   {w}x{h}")
        print("\nThe projector is usually the one that is not your laptop panel.")
        print("Mirrored outputs show up as a single display; extend the desktop first.")
        pygame.quit()
        return 0

    screen, sink = open_output(args, log)
    pygame.font.init()
    console = Console(log) if sink else None
    pygame.display.set_caption("vec2projector")
    pygame.mouse.set_visible(args.windowed)

    scene = Scene(parse_color(args.color), max(1, args.width), parse_color(args.bg),
                  cap=max(0, args.max_vectors))
    scene.grid = args.grid
    mp = Mapper(args.space, screen.get_size(), args.y_down)

    stats = Stats()
    raw_size = None
    if args.raw:
        try:
            rw, rh = (int(v) for v in args.raw.lower().split("x"))
            raw_size = (rw, rh)
        except ValueError:
            raise SystemExit(f"--raw wants WxH, e.g. 1280x720 (got {args.raw!r})")

    q, stop, threads = queue.Queue(maxsize=200000), threading.Event(), []
    if args.demo:
        threads.append(threading.Thread(target=demo_feed, args=(q, stop), daemon=True))
    else:
        if raw_size:
            threads.append(threading.Thread(target=raw_reader,
                                            args=(scene, stop, raw_size, log), daemon=True))
        elif not args.no_stdin and not sys.stdin.closed:
            threads.append(threading.Thread(target=stdin_reader, args=(q, stop), daemon=True))
        if args.http:
            threads.append(threading.Thread(
                target=http_reader,
                args=(scene, q, stop, args.http_host, args.http, stats, log), daemon=True))
        if args.watch:
            threads.append(threading.Thread(
                target=watch_reader,
                args=(scene, stop, args.watch, max(0.01, args.watch_interval), log),
                daemon=True))
        if args.udp:
            threads.append(threading.Thread(target=udp_reader,
                                            args=(q, stop, args.udp_host, args.udp, log),
                                            daemon=True))
            log(f"listening for vectors on udp://{args.udp_host}:{args.udp}")
    for t in threads:
        t.start()

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16) if args.hud else None
    layer = ImageLayer(args.fit)
    frames_shown = 0
    last_image_at = None
    running = True

    quit_req = threading.Event()                 # systemd stop / Ctrl-C -> leave cleanly
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: quit_req.set())
    atexit.register(lambda: sink and sink.close())

    splash_font = pygame.font.Font(None, max(24, screen.get_height() // 16)) if args.splash else None
    splash_lines = ["vec2projector is running",
                    f"{screen.get_width()}x{screen.get_height()}   "
                    + (f"framebuffer {args.fb}" if sink else pygame.display.get_driver()),
                    (f"waiting for frames on http://{args.http_host}:{args.http}"
                     if args.http else "waiting for input")]
    splash_active = args.splash
    first_frame = True

    while running and not quit_req.is_set():
        # --- drain the input queue (bounded, so a fast feed can't stall drawing)
        for _ in range(20000):
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            if splash_active:
                splash_active = False
                scene.clear()
            if not handle_line(item, scene, log):
                running = False
                break

        # --- newest image, if any (decode on the main thread: it owns the display)
        req = scene.image_req.take()
        if req is not None:
            splash_active = False
            kind, val = req
            if kind == "drop":
                layer.clear()
            elif kind == "surface":
                layer.set(val)
                frames_shown += 1
                last_image_at = time.time()
            else:
                try:
                    layer.set(pygame.image.load(val))
                    frames_shown += 1
                    last_image_at = time.time()
                except (pygame.error, OSError) as exc:
                    log(f"could not load image {val!r}: {exc}")
            scene.dirty = True

        # --- window / keyboard events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                mp.resize((e.w, e.h))
                scene.dirty = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                    mp.resize(screen.get_size())
                    scene.dirty = True
                elif e.key == pygame.K_g:
                    scene.grid = not scene.grid
                    scene.dirty = True
                elif e.key == pygame.K_c:
                    scene.clear()
                    layer.clear()
                elif e.key == pygame.K_s:
                    name = time.strftime("vec2projector-%Y%m%d-%H%M%S.png")
                    pygame.image.save(screen, name)
                    log(f"saved {name}")

        # --- draw (framebuffer output only redraws when something changed, so an idle
        #     projector costs almost nothing; window outputs redraw every tick as before)
        with scene.lock:
            dirty, scene.dirty = scene.dirty, False
        if sink is None or dirty or args.hud or splash_active or first_frame:
            first_frame = False
            segs, bg, grid = scene.snapshot()
            screen.fill(bg)
            layer.blit(screen)
            if grid:
                draw_grid(screen, mp)
            for x1, y1, x2, y2, col, wid in segs:
                p1, p2 = mp.to_px(x1, y1), mp.to_px(x2, y2)
                if args.arrows:
                    draw_arrow(screen, col, p1, p2, wid)
                elif args.aa and wid <= 1:
                    pygame.draw.aaline(screen, col, p1, p2)
                else:
                    pygame.draw.line(screen, col, p1, p2, wid)

            if splash_active:
                draw_splash(screen, splash_font, splash_lines)
            if font:
                hud = (f"{clock.get_fps():5.1f} fps | {len(segs):6d} vec | "
                       f"{scene.mode} | frames {scene.frames} | img {frames_shown} | "
                       f"q {q.qsize()}")
                screen.blit(font.render(hud, True, (150, 150, 150)), (12, 10))

            if sink is not None:
                sink.write(screen)
            else:
                pygame.display.flip()
        else:
            segs = scene.snapshot()[0]

        stats.update(clock.get_fps(), len(segs), frames_shown, last_image_at)
        clock.tick(args.fps)

    stop.set()
    if console:
        console.restore()
    if sink:
        sink.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
