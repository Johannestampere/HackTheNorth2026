"""Lossless PNG export of the exact RGB frame, without imaging dependencies."""
import struct
import zlib
from pathlib import Path

from .models import ProjectionFrame


def save_png(frame: ProjectionFrame, path: Path) -> None:
    """Save native-resolution pixels; no rescaling, rotation, or extra annotations."""
    def chunk(kind, data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data))
    stride = frame.width_px*3
    rows = b''.join(b'\0'+frame.rgb[i:i+stride] for i in range(0,len(frame.rgb),stride))
    payload = (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',frame.width_px,frame.height_px,8,2,0,0,0))
               +chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(payload)
