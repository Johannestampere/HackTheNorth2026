"""Verify flexible text output and the exact frame sent to the projector."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from PIL import Image
from ambient_projector.ai import SceneAI
from ambient_projector.app import Companion
from ambient_projector.render import Renderer, render_text_page
from ambient_projector.scene import TextPage

class TextTests(unittest.TestCase):
    def test_freeform_body_and_monochrome_output(self):
        page=TextPage('A useful title','One paragraph.\n\n1. First step\n2. Second step')
        self.assertEqual(TextPage.from_dict(page.as_dict()),page)
        for size in ((640,480),(1280,720),(1920,1080)):
            frame=render_text_page(page,*size)
            self.assertEqual(frame.size,size)
            self.assertEqual(frame.getpixel((0,0)),(0,0,0))
            r,g,b=frame.split()
            self.assertEqual(r.tobytes(),g.tobytes())
            self.assertEqual(g.tobytes(),b.tobytes())
            self.assertIsNotNone(frame.getbbox())

    def test_unreadable_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            render_text_page(TextPage('Title','Long content '*300),640,480)

    def test_camera_to_text_to_projector(self):
        page=TextPage('Practice','Ask about aiming or a rule.')
        ai=Mock(); ai.helpful_text.return_value=page
        camera=Mock(); camera.capture.return_value=Image.new('RGB',(100,100))
        projector=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            Companion(ai,Renderer(),Path(tmp),projector,camera,text_mode=True).cycle()
            ai.helpful_text.assert_called_once()
            ai.understand.assert_not_called()
            self.assertEqual(json.loads((Path(tmp)/'latest.json').read_text()),page.as_dict())
            with Image.open(Path(tmp)/'latest.png') as saved:
                self.assertEqual(saved.tobytes(),projector.show.call_args.args[0].tobytes())

    def test_model_contract_only_requires_title_and_body(self):
        client=Mock()
        client.responses.create.return_value=SimpleNamespace(status='completed',output_text=json.dumps({'title':'Title','body':'Helpful answer.'}))
        page=SceneAI(client).helpful_text(Image.new('RGB',(100,100)),'How do I aim?')
        self.assertEqual(page.body,'Helpful answer.')
        args=client.responses.create.call_args.kwargs
        self.assertEqual(set(args['text']['format']['schema']['required']),{'title','body'})
        self.assertIn('How do I aim?',str(args['input']))
