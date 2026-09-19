# Raspberry Pi display (`raspi/`)

Runs on the Pi that drives the projector over HDMI. Standalone: stdlib + `pygame`, nothing
from `src/companion`.

| File | Role |
| --- | --- |
| `vec2projector.py` | The display. Owns the HDMI output, listens on `127.0.0.1:8080`, holds the last frame it received. |
| `draw.py` | Type coordinates to draw lines; `draw.py --random` streams random strokes to demo real-time pushing. |
| `pi5/push_example.py` | How a server hands frames to the display (`push_image`, `push_raw`, `push_vectors`). |
| `pi5/*.service`, `*.timer` | systemd units: display, server, and a health watchdog. |
| `pi5/SETUP-pi5.md` | Pi 5 setup, including the HDMI boot setting for a cold projector. |

```sh
python3 raspi/vec2projector.py --windowed --http 8080 --grid    # display, in a window
python3 raspi/draw.py                                           # type: 0 0 1 1
```

`ProjectionFrame.rgb` (`width * height * 3` RGB8 bytes) is exactly what `POST /raw?w=&h=`
accepts, so a `Projector.project(rgb, w, h)` implementation can forward it as-is.

Tested on a laptop with the off-screen driver and a real window. Not yet run on a Pi 5.
