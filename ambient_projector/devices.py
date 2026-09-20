"""USB camera, optional microphone, and existing Pi HTTP display transport."""
from io import BytesIO
import time
import wave
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from PIL import Image


class Camera:
    """Open the webcam per capture to avoid stale buffered frames between cycles."""
    def __init__(self,index=0): self.index=index

    def capture(self):
        import cv2
        cap=cv2.VideoCapture(self.index)
        try:
            if not cap.isOpened(): raise RuntimeError(f'Cannot open camera {self.index}')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            deadline=time.monotonic()+1.5
            while True:
                ok,frame=cap.read()
                if not ok: raise RuntimeError('Camera returned no frame')
                if time.monotonic()>=deadline: break
            return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally: cap.release()


class Microphone:
    """Record a short mono PCM WAV; opt-in, raw audio stays in memory."""
    def __init__(self,seconds=6,device=None):
        if not 1<=seconds<=20: raise ValueError('Audio duration must be 1–20 seconds')
        self.seconds,self.device=seconds,device

    def capture(self):
        import sounddevice as sd
        rate=16000
        samples=sd.rec(int(self.seconds*rate),samplerate=rate,channels=1,dtype='int16',device=self.device)
        try: sd.wait()
        finally: sd.stop()
        data=BytesIO()
        with wave.open(data,'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
            stream.writeframes(samples.tobytes())
        return data.getvalue()


class Projector:
    """Send full RGB frames to raspi/vec2projector.py; no calibration is implied."""
    def __init__(self,url,width,height):
        parsed=urlsplit(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError('Projector URL must be an HTTP(S) base URL')
        self.url,self.width,self.height=url.rstrip('/'),width,height

    def show(self,image):
        if image.mode!='RGB' or image.size!=(self.width,self.height):
            raise ValueError('Projector needs RGB at its configured resolution')
        request=Request(f'{self.url}/raw?w={self.width}&h={self.height}',data=image.tobytes(),
                        headers={'Content-Type':'application/octet-stream'},method='POST')
        with urlopen(request,timeout=5) as response:
            if response.status!=200: raise RuntimeError('Projector rejected the frame')

    def blank(self): self.show(Image.new('RGB',(self.width,self.height),'black'))
