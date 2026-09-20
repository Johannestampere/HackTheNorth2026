# Ambient projector companion

A standalone fallback demo, independent of pool detection, planning, calibration,
Torch, and matplotlib. Point a USB camera at what you are doing; it periodically
asks OpenAI what extra information would help, then projects text and diagrams
onto a fixed wall/surface. Optional microphone clips add spoken context. Optional
image generation adds illustrations, such as a brain while studying biology.

## What actually happens

```text
optional microphone clip → OpenAI transcription ┐
blank projector → camera snapshot ──────────────┤
                                               ↓
                         OpenAI vision → validated visual-plan JSON
                                               ↓
                  Pillow diagram OR optional OpenAI-generated illustration
                                               ↓
                            RGB frame → existing Pi projector server
```

The language model does not output millions of pixel values. It returns a title,
short notes, and drawing instructions (text boxes, arrows, lines, rectangles,
ellipses). Pillow renders these deterministically. With `--illustrations`, the
model can request a separate Images API call; the resulting bitmap replaces the
diagram while our renderer retains the readable title and notes. If that call
fails, the diagram remains the fallback.

There is **no internet image retrieval** in this version. Illustrations are
AI-generated and labeled as such; they are not sourced anatomy references.
Diagram/illustration accuracy still needs human review. There is no automatic
rotation or camera-to-projector registration: aim the projector at a clear wall,
set its focus/keystone, and leave it fixed. Camera coordinates are not projector
coordinates. The camera can look at the book while the projector points elsewhere.

## Install on the Raspberry Pi

Run from the repo root. If the existing projector server works, keep using it;
do not start a second server on the same display/port.

```bash
cd /home/jack/HackTheNorth2026
sudo apt-get install -y python3-venv python3-opencv
python3 -m venv --system-site-packages ambient_projector/.venv
ambient_projector/.venv/bin/python -m pip install 'openai>=1.75,<3' 'Pillow>=10.1,<13'
```

This reuses the Pi's system OpenCV. No local language model, GPU, or Torch is
needed. For a Mac/Linux machine without system OpenCV, install the complete set:

```bash
python3 -m venv ambient_projector/.venv
ambient_projector/.venv/bin/python -m pip install -r ambient_projector/requirements.txt
```

Set `OPENAI_API_KEY` in the shell that launches the program. Never commit the key.
`.env` files are not automatically loaded. Cloud calls require an API account
with billing and access to the configured models. Defaults:

| Job | Default model | Override |
| --- | --- | --- |
| Image and context → visual plan | `gpt-4.1-mini` | `--model` |
| Optional speech → text | `gpt-4o-mini-transcribe` | `--audio-model` |
| Optional illustration | `gpt-image-1` | `--image-model` |

Overrides must support the corresponding API/features. Using the OpenAI SDK does
not make every model interchangeable.

## First test: no camera, key, or API charges

```bash
cd /home/jack/HackTheNorth2026
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --scene ambient_projector/examples/brain.json --once
```

Open `/home/jack/HackTheNorth2026/ambient_projector/output/latest.png`.
This example is a **manually authored concept map**, not evidence of a successful
OpenAI call or a generated brain illustration. `latest.json` holds its plan.

To send the example to an already-running Pi display:

```bash
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --scene ambient_projector/examples/brain.json --once \
  --projector-url http://127.0.0.1:8080
```

If needed, start the repo's display process in another Pi terminal:

```bash
cd /home/jack/HackTheNorth2026
python3 raspi/vec2projector.py --http 8080
```

See `/home/jack/HackTheNorth2026/raspi/README.md` for pygame/HDMI setup. Use a
clean dedicated display without HUD, grid, persistent vectors or other senders;
black RGB frames do not erase separate vector overlays. Use `--width` and
`--height` to match the actual HDMI resolution (default 1280×720).

## Camera → OpenAI → projector

First do one update:

```bash
cd /home/jack/HackTheNorth2026
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --camera 0 --once \
  --request 'Help me understand what I am reading. Use a useful diagram.' \
  --projector-url http://127.0.0.1:8080
```

Then remove `--once` for recurring updates:

```bash
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --camera 0 --interval 20 \
  --projector-url http://127.0.0.1:8080
```

`--interval 20` is the minimum interval **between cycle starts**, not a guaranteed
20-second response time. Cycles never overlap. Capture, audio, or image generation
can take longer; the next cycle starts after completion with at least a 1-second
pause. API timeout is 90 seconds per request, SDK retries are disabled, and errors
wait until the next cycle instead of retrying in a tight loop.

The program briefly blanks output while opening/warming the webcam. After capture,
it restores the previous image while waiting for new guidance. The first cycle
stays black until output is ready. A failed update clears the image so old advice
is not left presented as current. Ctrl+C clears the projector; `--once` leaves the
successful result visible. A disconnected projector cannot be remotely cleared.

To test a saved image, omit the camera entirely:

```bash
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --image /home/jack/table-test.jpg --once \
  --request 'Explain something useful about this scene.'
```

Omit `--projector-url` for PNG-only operation. Saved images should already be
free of projected text; the program cannot undo illumination in an existing photo.

## Add microphone context

Microphone use is **off by default**. On Pi:

```bash
sudo apt-get install -y libportaudio2
ambient_projector/.venv/bin/python -m pip install 'sounddevice>=0.5,<1'
ambient_projector/.venv/bin/python -m ambient_projector.app --list-audio-devices
```

The system NumPy installed with OpenCV is sufficient. Elsewhere use
`ambient_projector/requirements-audio.txt` and install PortAudio if your OS needs
it (on macOS, `brew install portaudio`). Allow camera/microphone access in OS
settings where required.

```bash
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --camera 0 --audio --audio-seconds 6 --audio-device 1 \
  --interval 20 --illustrations \
  --request 'Teach me a useful concept related to what I am reading aloud.' \
  --projector-url http://127.0.0.1:8080
```

Replace device `1` with the listed microphone index, or omit `--audio-device` to
use the system default. Each cycle records a short mono clip, transcribes it,
then captures an image. This is sampled audio, **not continuous listening or
synchronized video/audio streaming**. Speak during the logged recording window.
If capture/transcription fails, the cycle proceeds with camera context alone.

## Illustrations and the biology demo

Add `--illustrations` to allow image generation when the model requests it. Point
the camera at a clearly readable biology page and say what concept you are
studying, or supply it through `--request`. A successful cycle can project a brain
illustration with short supplementary notes. Without the flag, it draws diagrams
using the primitive vocabulary only.

Image generation adds latency and cost; it can take much longer than 20 seconds.
Start with `--once --illustrations` for a predictable demo. For repetitive operation,
try `--interval 60`. The app performs no caching or deduplication: each cycle can
make one vision call, one transcription call, and one image call when enabled.
There is no cross-cycle conversation memory; every cycle uses its current inputs.

## Data and outputs

- Camera JPEGs and enabled microphone audio are sent to OpenAI. Frames are
  resized to at most 1600 pixels per side and encoded without source metadata.
- Raw camera/audio inputs and transcripts stay in memory; they are not saved by
  this program. The current visual plan and PNG are saved under ignored `output/`.
- `store=False` is set for Responses calls; this does not describe all provider
  retention policies for every endpoint.
- `--outdir /absolute/path` changes the output folder. Each cycle replaces
  `latest.png` and `latest.json`, rather than filling the Pi with image history.
- Model output is validated before rendering. No generated Python, HTML, URLs,
  shell commands, or executable instructions are evaluated.

## Files and tests

| File | Responsibility |
| --- | --- |
| `app.py` | CLI, update loop, dependency composition, failure handling |
| `devices.py` | Webcam, optional microphone, projector HTTP output |
| `ai.py` | OpenAI Responses, transcription and image generation calls |
| `scene.py` | Validated visual-plan data and JSON schema |
| `render.py` | Deterministic text/shapes/illustration-to-RGB renderer |

```bash
ambient_projector/.venv/bin/python -m unittest discover -s ambient_projector/tests -v
```

Tests run offline with fake API/device responses. They check schema validation,
rendering at multiple resolutions, image preparation, audio/image API paths,
blank/capture/display ordering, fallback behavior, and exact projector RGB bytes.
They do **not** verify cloud account access, microphone support, camera hardware,
physical projection, or scientific correctness. Perform the single-update live
command before claiming a working end-to-end hardware demo.

Troubleshooting: camera errors → close other webcam processes and try `--camera 1`;
audio errors → list/select devices; HTTP display errors → check
`python3 raspi/sender.py health`; API failures → check the key, quota and access to
the selected model; invalid layout → retry with a request for a simpler diagram.
The bundled font is intended for short English text, not general multilingual typography.

Official references: [vision inputs](https://developers.openai.com/api/docs/guides/images-vision),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[transcription](https://developers.openai.com/api/docs/guides/speech-to-text),
[image generation](https://developers.openai.com/api/docs/guides/image-generation).
