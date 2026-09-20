"""Camera → optional speech → OpenAI helpful text (or optional web image) → projector."""
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
from .render import Renderer, render_text_page
from .scene import Scene, TextPage
from .web_images import WebImages, fullscreen

LOG=logging.getLogger('ambient_projector')


class Companion:
    """One bounded update at a time. Dependencies are injected for offline tests."""
    def __init__(self,ai,renderer,outdir,projector=None,camera=None,microphone=None,illustrations=False,image_search=None,text_mode=False):
        self.ai,self.renderer,self.outdir=ai,renderer,Path(outdir)
        self.projector,self.camera,self.microphone=projector,camera,microphone
        self.illustrations=illustrations
        self.text_mode=text_mode
        self.image_search=image_search
        self.previous=None

    def cycle(self,image_path=None,scene_path=None,request='',query=None):
        """Blank during capture, analyze, render, save, then display a complete frame."""
        transcript=''
        if query:
            return self.project_search(query)
        if scene_path:
            data=json.loads(Path(scene_path).read_text())
            scene=TextPage.from_dict(data) if 'body' in data else Scene.from_dict(data)
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
            if self.image_search is not None:
                query=self.ai.image_query(image,transcript,request)
                return self.project_search(query)
            LOG.info('Asking OpenAI for visual guidance')
            scene=(self.ai.helpful_text(image,transcript,request) if self.text_mode
                   else self.ai.understand(image,transcript,request,self.illustrations))
        illustration=None
        if not isinstance(scene,TextPage) and self.illustrations and scene.illustration_prompt and self.ai is not None:
            try:
                LOG.info('Generating illustration; this can take longer than the update interval')
                illustration=self.ai.illustrate(scene.illustration_prompt)
            except Exception as error:
                LOG.warning('Illustration unavailable (%s); using the diagram',type(error).__name__)
        frame=(render_text_page(scene,self.renderer.width,self.renderer.height)
               if isinstance(scene,TextPage) else self.renderer.render(scene,illustration))
        self.outdir.mkdir(parents=True,exist_ok=True)
        # Replace latest files instead of accumulating camera/audio history.
        temporary=self.outdir/'next.png'; frame.save(temporary,format='PNG')
        temporary.replace(self.outdir/'latest.png')
        (self.outdir/'latest.json').write_text(json.dumps(scene.as_dict(),indent=2)+'\n')
        if self.projector: self.projector.show(frame)
        self.previous=frame
        LOG.info('Saved %s',(self.outdir/'latest.png').resolve())
        return scene

    def project_search(self,query):
        """Project a real downloaded image, retaining its source and license metadata."""
        LOG.info('Searching Commons: %s',query)
        candidates=self.image_search.search(query)
        selected=self.ai.choose_image(query,candidates) if self.ai else 0
        # If the chosen thumbnail is unavailable, try at most two other real results.
        order=[selected]+[i for i in range(len(candidates)) if i!=selected]
        image=None
        for index in order[:3]:
            try:
                image=self.image_search.download(candidates[index])
                metadata={**candidates[index], 'selection': 'model_metadata' if self.ai and index==selected else 'search_rank_fallback'}
                break
            except Exception as error:
                LOG.warning('Image download failed (%s); trying another result',type(error).__name__)
        if image is None: raise RuntimeError('No search result could be downloaded')
        frame=fullscreen(image,self.renderer.width,self.renderer.height)
        self.outdir.mkdir(parents=True,exist_ok=True)
        frame.save(self.outdir/'next.png',format='PNG')
        (self.outdir/'next.png').replace(self.outdir/'latest.png')
        (self.outdir/'latest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (self.outdir/'source.txt').write_text('\n'.join(f'{key}: {metadata.get(key, "")}' for key in
            ('title','source_url','image_url','author','credit','license','license_url','attribution'))+'\n')
        if self.projector: self.projector.show(frame)
        self.previous=frame
        LOG.info('Saved %s',(self.outdir/'latest.png').resolve())
        LOG.info('Source: %s',metadata['source_url'])
        return metadata

    def clear(self):
        if self.projector:
            try: self.projector.blank()
            except Exception: LOG.warning('Could not clear display; check its connection')
        self.previous=None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--image',type=Path,help='Use a saved photo instead of a webcam')
    source.add_argument('--query',help='Search directly without camera or OpenAI; uses first usable result')
    source.add_argument('--scene',type=Path,help='Offline title/body or drawing JSON; no OpenAI calls')
    parser.add_argument('--web-images',action='store_true',help='Search web images instead of the default helpful text')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--audio',action='store_true',help='Enable microphone recording and cloud transcription')
    parser.add_argument('--audio-seconds',type=float,default=6)
    parser.add_argument('--audio-device',type=int)
    parser.add_argument('--list-audio-devices',action='store_true')
    parser.add_argument('--interval',type=float,default=20,help='Minimum seconds between cycle starts; no overlapping requests')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--request',default='',help='Optional goal, e.g. help me understand what I am reading')
    parser.add_argument('--model',default='gpt-4.1-mini')
    parser.add_argument('--audio-model',default='gpt-4o-mini-transcribe')
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
    if (args.scene or args.query) and args.audio: parser.error('--audio requires camera or image context')
    if not math.isfinite(args.audio_seconds) or not 1<=args.audio_seconds<=20: parser.error('--audio-seconds must be 1–20')
    renderer=Renderer(args.width,args.height)
    # Do not let a user overwrite their input with generated output.
    for source_path in (args.image,args.scene):
        if source_path and source_path.resolve() in {(args.outdir/name).resolve() for name in ('latest.png','latest.json','next.png')}:
            parser.error('Input must be outside the output file paths')
    ai=None
    if not args.scene and not args.query:
        if not os.environ.get('OPENAI_API_KEY','').strip(): parser.error('Set OPENAI_API_KEY in this terminal (or use --scene for an offline test)')
        from openai import OpenAI
        client=OpenAI(timeout=90,max_retries=0)
        ai=SceneAI(client,args.model,args.audio_model)
    projector=Projector(args.projector_url,args.width,args.height) if args.projector_url else None
    companion=Companion(ai,renderer,args.outdir,projector,Camera(args.camera),
                        Microphone(args.audio_seconds,args.audio_device) if args.audio else None,
                        image_search=WebImages() if (args.web_images or args.query) and not args.scene else None,
                        text_mode=True)
    try:
        while True:
            started=time.monotonic()
            try:
                companion.cycle(args.image,args.scene,args.request,args.query)
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
