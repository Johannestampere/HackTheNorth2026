"""Explicit HTTP output adapter for raspi/vec2projector.py's raw RGB endpoint."""

from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class HttpProjector:
    """Send native-resolution RGB; raise on failure so stale output is not called fresh.

    Construct with the calibrated physical display resolution. The receiver can
    scale images and does not expose that resolution in its health API, so the
    operator must configure/verify the actual HDMI mode. Pose is likewise external.
    """

    def __init__(self, url: str, width_px: int, height_px: int, timeout: float = 3):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError('Display URL must be an HTTP(S) base URL without query or fragment')
        if any(type(v) is not int or v <= 0 for v in (width_px,height_px)) or timeout <= 0:
            raise ValueError('Display dimensions and timeout must be positive')
        if width_px*height_px*3 > 64*1024*1024:
            raise ValueError('Frame exceeds the Pi receiver 64 MiB limit')
        self.url,self.width_px,self.height_px,self.timeout = url.rstrip('/'),width_px,height_px,timeout

    def project(self, rgb: bytes, width_px: int, height_px: int) -> None:
        """POST one full frame. HTTP acceptance is not proof of physical alignment."""
        if (width_px,height_px) != (self.width_px,self.height_px):
            raise ValueError('Frame resolution differs from configured projector resolution')
        if not isinstance(rgb,bytes) or len(rgb) != width_px*height_px*3:
            raise ValueError('Expected width × height × 3 RGB bytes')
        request = Request(f'{self.url}/raw?w={width_px}&h={height_px}',data=rgb,
                          headers={'Content-Type':'application/octet-stream'},method='POST')
        with urlopen(request,timeout=self.timeout) as response:
            if response.status != 200:
                raise OSError(f'Display rejected frame: HTTP {response.status}')

    def blank(self) -> None:
        """Replace the image with black at the same native resolution."""
        self.project(bytes(self.width_px*self.height_px*3),self.width_px,self.height_px)
