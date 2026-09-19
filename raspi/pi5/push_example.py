#!/usr/bin/env python3
"""
How your server hands frames to the display. Stdlib only -- no requests needed.

The one rule: pushing must never take your server down. The display may be
restarting, redeploying, or briefly wedged; a failed push is a dropped frame,
not an exception that kills the server.
"""

import os
import time
import urllib.error
import urllib.request

DISPLAY = os.environ.get("VEC2PROJECTOR_URL", "http://127.0.0.1:8080")


def push_image(data, timeout=2.0):
    """Send encoded image bytes (png/jpeg/bmp). Returns True if it landed."""
    return _post("/frame", data, timeout)


def push_raw(data, w, h, timeout=2.0):
    """Send raw rgb24 pixels -- no encode/decode, fastest path on one box."""
    return _post(f"/raw?w={w}&h={h}", data, timeout)


def push_vectors(lines, timeout=2.0):
    """Send protocol lines, e.g. ['mode frame', 'v 0 0 1 1 red 4', 'end']."""
    return _post("/vectors", ("\n".join(lines) + "\n").encode(), timeout)


def _post(path, data, timeout):
    req = urllib.request.Request(DISPLAY + path, data=data, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False          # display is down or busy: drop the frame, keep serving


def display_healthy(timeout=2.0):
    import json
    try:
        with urllib.request.urlopen(DISPLAY + "/health", timeout=timeout) as r:
            return json.load(r).get("ok", False)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


if __name__ == "__main__":
    # Stand-in for whatever your server actually renders.
    import io
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # we only rasterize, never display
    import pygame
    pygame.init()

    W, H = 1280, 720
    surf = pygame.Surface((W, H))
    font = pygame.font.SysFont("monospace", 64)
    n, sent, dropped = 0, 0, 0

    while True:
        n += 1
        surf.fill((10, 12, 24))
        pygame.draw.circle(surf, (0, 200, 160), (W // 2, H // 2), 120 + int(60 * (n % 40) / 40))
        surf.blit(font.render(f"frame {n}", True, (240, 240, 240)), (40, 40))

        # Raw is cheapest on one box; swap to push_image for anything networked.
        if push_raw(pygame.image.tobytes(surf, "RGB"), W, H):
            sent += 1
        else:
            dropped += 1
            time.sleep(0.5)      # display is down -- back off, do not hammer it

        if n % 100 == 0:
            print(f"sent={sent} dropped={dropped} healthy={display_healthy()}", flush=True)
        time.sleep(1 / 30)
