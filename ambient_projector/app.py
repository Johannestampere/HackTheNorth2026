"""Camera → optional speech → OpenAI visual plan → RGB projector, on a timer."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import time
from PIL import Image
from .ai import SceneAI
from .devices import Camera, Microphone, Projector
from .render import Renderer
from .scene import Scene

LOG=logging.getLogger('ambient_projector')


class Companion:
    """One bounded update at a time. Dependencies are injected for offline tests."""
    def __init__(self,ai,renderer,outdir,projector=None,camera=None,microphone=None,illustrations=False):
        self.ai,self.renderer,self.outdir=ai,renderer,Path(outdir)
        self.projector,self.camera,self.microphone=projector,camera,microphone
        self.illustrations=illustrations
        self.previous=None

    def cycle(self,image_path=None,scene_path=None,request=''):
        """Blank during capture, analyze, render, save, then display a complete frame."""
        transcript=''
        if scene_path:
            scene=Scene.from_dict(json.loads(Path(scene_path).read_text()))
        else:
            if self.microphone:
                LOG.info('Recording microphone for %.1f seconds',self.microphone.seconds)
                try: transcript=self.ai.transcribe(self.microphone.capture())
                except Exception as error:
                    LOG.warning('Audio unavailable (%s); continuing with camera only',type(error).__name__)
            if image_path:
                with Image.open(image_path) as source: image=source.copy()
            else:
                if self.projector: self.projector.blank()
                image=self.camera.capture()
                if self.projector and self.previous is not None:
                    self.projector.show(self.previous)
            LOG.info('Asking OpenAI for visual guidance')
            scene=self.ai.understand(image,transcript,request,self.illustrations)
        illustration=None
        if self.illustrations and scene.illustration_prompt and self.ai is not None:
            try:
                LOG.info('Generating illustration; this can take longer than the update interval')
                illustration=self.ai.illustrate(scene.illustration_prompt)
            except Exception as error:
                LOG.warning('Illustration unavailable (%s); using the diagram',type(error).__name__)
        frame=self.renderer.render(scene,illustration)
        self.outdir.mkdir(parents=True,exist_ok=True)
        # Replace latest files instead of accumulating camera/audio history.
        temporary=self.outdir/'next.png'; frame.save(temporary,format='PNG')
        temporary.replace(self.outdir/'latest.png')
        (self.outdir/'latest.json').write_text(json.dumps(scene.as_dict(),indent=2)+'\n')
        if self.projector: self.projector.show(frame)
        self.previous=frame
        LOG.info('Saved %s',(self.outdir/'latest.png').resolve())
        return scene

    def clear(self):
        if self.projector:
            try: self.projector.blank()
            except Exception: LOG.warning('Could not clear display; check its connection')
        self.previous=None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--image',type=Path,help='Use a saved photo instead of a webcam')
    source.add_argument('--scene',type=Path,help='Offline drawing JSON; no OpenAI calls')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--audio',action='store_true',help='Enable microphone recording and cloud transcription')
    parser.add_argument('--audio-seconds',type=float,default=6)
    parser.add_argument('--audio-device',type=int)
    parser.add_argument('--list-audio-devices',action='store_true')
    parser.add_argument('--interval',type=float,default=20,help='Minimum seconds between cycle starts; no overlapping requests')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--illustrations',action='store_true',help='Allow additional paid image-generation requests')
    parser.add_argument('--request',default='',help='Optional goal, e.g. help me understand what I am reading')
    parser.add_argument('--model',default='gpt-4.1-mini')
    parser.add_argument('--audio-model',default='gpt-4o-mini-transcribe')
    parser.add_argument('--image-model',default='gpt-image-1')
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--projector-url',help='Example: http://127.0.0.1:8080; omit for PNG-only preview')
    parser.add_argument('--outdir',type=Path,default=Path(__file__).parent/'output')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if args.list_audio_devices:
        import sounddevice as sd
        print(sd.query_devices()); return 0
    if not math.isfinite(args.interval) or args.interval<1: parser.error('--interval must be at least 1 second')
    if args.scene and (args.audio or args.illustrations): parser.error('--scene is offline; omit --audio and --illustrations')
    if not math.isfinite(args.audio_seconds) or not 1<=args.audio_seconds<=20: parser.error('--audio-seconds must be 1–20')
    renderer=Renderer(args.width,args.height)
    # Do not let a user overwrite their input with generated output.
    for source_path in (args.image,args.scene):
        if source_path and source_path.resolve() in {(args.outdir/name).resolve() for name in ('latest.png','latest.json','next.png')}:
            parser.error('Input must be outside the output file paths')
    ai=None
    if not args.scene:
        if not os.environ.get('OPENAI_API_KEY','').strip(): parser.error('Set OPENAI_API_KEY in this terminal (or use --scene for an offline test)')
        from openai import OpenAI
        client=OpenAI(timeout=90,max_retries=0)
        ai=SceneAI(client,args.model,args.audio_model,args.image_model)
    projector=Projector(args.projector_url,args.width,args.height) if args.projector_url else None
    companion=Companion(ai,renderer,args.outdir,projector,Camera(args.camera),
                        Microphone(args.audio_seconds,args.audio_device) if args.audio else None,args.illustrations)
    try:
        while True:
            started=time.monotonic()
            try:
                companion.cycle(args.image,args.scene,args.request)
            except Exception as error:
                # Avoid dumping SDK request bodies, credentials or private transcripts.
                LOG.error('Update failed: %s. Check device connection, model access and visual layout.',type(error).__name__)
                companion.clear()
                if args.once: return 1
            if args.once: return 0
            time.sleep(max(1,args.interval-(time.monotonic()-started)))
    except KeyboardInterrupt:
        companion.clear()
        LOG.info('Stopped and cleared projector')
        return 0


if __name__=='__main__':
    raise SystemExit(main())
