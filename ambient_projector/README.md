# Ambient projector: camera context → helpful text → wall

By default, OpenAI uses the camera snapshot and optional speech to return a title
and helpful plain text. The renderer enforces only **black background, white text,
and a title at the top**. The body can contain paragraphs, lists, or steps; there
is no fixed number of sections. Text wraps and scales to fit, with a minimum
readable font size. Excessively long output fails rather than clipping or
shrinking into unreadable text.

The model should explain something useful, answer the spoken question, or ask
for missing context. Seeing a pool table alone does not establish which game or
rules apply, and this mode does not calculate shots. Use the pool planner for
shot recommendations.

Use `--web-images` to explicitly choose the previous image-search experience:
OpenAI chooses a query, searches Wikimedia Commons, selects from result metadata,
and downloads an image. It does not visually verify search results. Images keep
their aspect ratio, and source/credit/license metadata is saved separately.

## Offline text preview

```bash
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --scene ambient_projector/examples/helpful-text.json --once
```

This uses a fixture, without a camera, network, or API key. `latest.png` contains
the rendered output; `latest.json` contains the title and body. Add
`--projector-url http://127.0.0.1:8080` to display it on the Pi.

## Install

On the Pi, reuse system OpenCV:

```bash
cd /home/jack/HackTheNorth2026
sudo apt-get install -y python3-venv python3-opencv
python3 -m venv --system-site-packages ambient_projector/.venv
ambient_projector/.venv/bin/python -m pip install 'openai>=1.75,<3' 'Pillow>=10.1,<13'
```

On Mac or a machine without system OpenCV:

```bash
python3 -m venv ambient_projector/.venv
ambient_projector/.venv/bin/python -m pip install -r ambient_projector/requirements.txt
```

Set `OPENAI_API_KEY` locally before camera/context mode. Do not commit it. `.env`
is not loaded automatically. Default vision model is `gpt-4.1-mini` (`--model`),
transcription is `gpt-4o-mini-transcribe` (`--audio-model`). No Torch, matplotlib,
local language model or image-generation API is required.

## Test web images without a camera or OpenAI key

```bash
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --query 'human brain anatomy diagram' --once
```

Direct-query mode uses the first usable search result; it does not call OpenAI.
On the Pi, the output is:

- `/home/jack/HackTheNorth2026/ambient_projector/output/latest.png`
- `/home/jack/HackTheNorth2026/ambient_projector/output/latest.json`
- `/home/jack/HackTheNorth2026/ambient_projector/output/source.txt`

`latest.json` records the query, chosen image URL, source page, selection method,
author, credit, license and attribution metadata. `source.txt` is a readable copy.
Check the linked source page for its complete reuse requirements and provide
attribution when sharing or publicly presenting the image. We retain metadata
but do not draw it over the image. Files are replaced each cycle, not accumulated.
Use `--outdir /absolute/path` to choose another output location.

## Camera → helpful text → projector

Start the existing Pi display if it is not already running (see
`/home/jack/HackTheNorth2026/raspi/README.md` for pygame/HDMI setup):

```bash
cd /home/jack/HackTheNorth2026
python3 raspi/vec2projector.py --http 8080
```

In another terminal:

```bash
cd /home/jack/HackTheNorth2026
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --camera 0 --once \
  --request 'Give useful, concise guidance about what I am looking at.' \
  --projector-url http://127.0.0.1:8080
```

Remove `--once` to repeat at the default `--interval 20`. Set `--width` and
`--height` to match the projector mode (default 1280×720). Omit `--projector-url`
for PNG-only previews. Use `--image /absolute/path/to/photo.jpg` instead of a live
camera to test on a saved image.

Use a dedicated display without other senders, HUD, grid, or persistent vector
overlays. RGB blanking cannot erase a separate vector layer. Aim/focus/keystone
are configured physically. No robot rotation or camera-to-projector registration
is performed; the camera can face the book while the projector faces the wall.

## Optional audio

Audio is off by default. On Pi:

```bash
sudo apt-get install -y libportaudio2
ambient_projector/.venv/bin/python -m pip install 'sounddevice>=0.5,<1'
ambient_projector/.venv/bin/python -m ambient_projector.app --list-audio-devices
```

Then choose the listed microphone index:

```bash
ambient_projector/.venv/bin/python -m ambient_projector.app \
  --camera 0 --audio --audio-seconds 6 --audio-device 1 \
  --interval 20 --projector-url http://127.0.0.1:8080
```

Omit `--audio-device` for the default microphone. On other platforms install
`requirements-audio.txt` and PortAudio as needed. Microphone errors fall back to
camera-only context. Speech is sampled in short windows, not continuously; speak
when the recording message appears. Audio is transcribed before the snapshot.

## Timing, failures, and data

- Text mode uses one vision/text request plus optional transcription. Web-image
  mode uses up to two requests, Commons search and an image download. Cycles do
  not overlap; 20 seconds is a minimum start-to-start interval, not a latency SLA.
- OpenAI requests have a 90-second timeout and no SDK retries. Web requests have
  a 20-second timeout; at most three candidate images are attempted. No image
  generation is used by the live command.
- Projection is briefly blank during camera capture. Previous output is restored
  while processing. A failed update clears it. Ctrl+C clears; a successful
  `--once` leaves its image visible. Network failures can prevent clearing.
- Camera images and enabled microphone audio go to OpenAI. Raw inputs/transcripts
  are not saved. The search query goes to Commons. Responses uses `store=False`;
  this is not a claim about all provider retention policies.
- Only Wikimedia HTTPS download hosts are accepted, including redirects. Downloads
  are size-limited and decoded as images; no model-generated code/HTML is executed.
- Each cycle is independent; no conversational memory or search caching yet.
- The old `--scene examples/brain.json --once` remains as an explicit offline
  renderer test. It is not the default live experience. `--illustrations` and
  `--image-model` are no longer CLI options.

## Tests and files

```bash
ambient_projector/.venv/bin/python -m unittest discover -s ambient_projector/tests -v
```

Offline tests cover context→query→search→download→RGB, source metadata, invalid
selection, URL restrictions, failed-download fallback, transparency, aspect ratio,
API shapes, audio fallback and display bytes. The direct-query brain example has
also been run against live Commons search and download. Hardware and cloud calls
still depend on your device/network/key; offline tests do not prove those work.

`app.py` composes the loop, `ai.py` handles model calls, `web_images.py` searches,
downloads and fits images, and `devices.py` handles capture/display. `scene.py`
and `render.py` validate and render text pages as well as the retained offline
diagram example.

References: [Commons image metadata API](https://www.mediawiki.org/wiki/API:Imageinfo),
[OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision),
[structured output](https://developers.openai.com/api/docs/guides/structured-outputs).
