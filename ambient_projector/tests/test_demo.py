"""Offline tests for rendering, provider contracts, orchestration and hardware bytes."""
import base64
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from ambient_projector.ai import SceneAI
from ambient_projector.app import Companion, main
from ambient_projector.devices import Projector
from ambient_projector.render import Renderer
from ambient_projector.scene import Scene

EXAMPLE=Path(__file__).resolve().parents[1]/'examples/brain.json'

class DemoTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads(EXAMPLE.read_text())
        self.scene=Scene.from_dict(self.data)
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)

    def test_example_and_every_primitive_render_at_supported_sizes(self):
        self.data['elements'].append(dict(kind='line',x=.1,y=.8,x2=.9,y2=.8,text='',color='green'))
        scene=Scene.from_dict(self.data)
        for size in ((1280,720),(640,480),(1920,1080)):
            frame=Renderer(*size).render(scene)
            self.assertEqual(frame.size,size)
            self.assertEqual(frame.mode,'RGB')
            self.assertIsNotNone(frame.getbbox())
            self.assertEqual(frame.getpixel((0,0)),(0,0,0))

    def test_bad_primitives_and_boxes_rejected(self):
        for changes in ({'x':float('nan')},{'x':-1},{'x':True},{'kind':'python'},
                        {'x2':0},{'text':'a'*101},{'color':'url(https://example.com)'}):
            data=json.loads(EXAMPLE.read_text()); data['elements'][0].update(changes)
            with self.subTest(changes=changes),self.assertRaises((ValueError,TypeError)):
                Scene.from_dict(data)

    def test_api_image_schema_and_context(self):
        client=Mock()
        client.responses.create.return_value=SimpleNamespace(status='completed',output_text=json.dumps(self.data))
        scene=SceneAI(client).understand(Image.new('RGB',(2000,1000)), 'Reading about the brain','Explain',True)
        self.assertEqual(scene,self.scene)
        args=client.responses.create.call_args.kwargs
        self.assertFalse(args['store']); self.assertTrue(args['text']['format']['strict'])
        content=args['input'][0]['content']
        self.assertIn('Reading about the brain',content[0]['text'])
        with Image.open(BytesIO(base64.b64decode(content[1]['image_url'].split(',')[1]))) as photo:
            self.assertEqual(photo.size,(1600,800))

    def test_refusal_and_invalid_schema_fail(self):
        client=Mock(); ai=SceneAI(client)
        for status,text in (('incomplete',''),('completed',''),('completed','{}')):
            client.responses.create.return_value=SimpleNamespace(status=status,output_text=text)
            with self.assertRaises(RuntimeError): ai.understand(Image.new('RGB',(10,10)))

    def test_audio_and_image_api_paths(self):
        client=Mock(); ai=SceneAI(client)
        client.audio.transcriptions.create.return_value=SimpleNamespace(text='brain')
        self.assertEqual(ai.transcribe(b'wav-data'),'brain')
        self.assertEqual(client.audio.transcriptions.create.call_args.kwargs['file'].getvalue(),b'wav-data')
        buffer=BytesIO(); Image.new('RGB',(32,32),'green').save(buffer,format='PNG')
        client.images.generate.return_value=SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(buffer.getvalue()).decode())])
        self.assertEqual(ai.illustrate('brain').size,(32,32))

    def test_capture_is_blanked_and_output_matches_preview(self):
        ai=Mock(); ai.understand.return_value=self.scene
        devices=Mock(); devices.camera.capture.return_value=Image.new('RGB',(100,100))
        app=Companion(ai,Renderer(),self.out,devices.projector,devices.camera)
        app.cycle()
        names=[c[0] for c in devices.mock_calls]
        self.assertLess(names.index('projector.blank'),names.index('camera.capture'))
        with Image.open(self.out/'latest.png') as saved:
            self.assertEqual(saved.tobytes(),devices.projector.show.call_args.args[0].tobytes())
        self.assertEqual(json.loads((self.out/'latest.json').read_text()),self.data)

    def test_audio_failure_continues_and_illustration_failure_uses_diagram(self):
        ai=Mock(); ai.understand.return_value=Scene.from_dict({**self.data,'illustration_prompt':'brain'})
        ai.illustrate.side_effect=RuntimeError('not available')
        mic=Mock(seconds=6); mic.capture.side_effect=RuntimeError('missing microphone')
        camera=Mock(); camera.capture.return_value=Image.new('RGB',(100,100))
        app=Companion(ai,Renderer(),self.out,camera=camera,microphone=mic,illustrations=True)
        app.cycle()
        self.assertEqual(ai.understand.call_args.args[1],'')
        self.assertTrue((self.out/'latest.png').exists())

    def test_offline_cli_needs_no_key_or_hardware(self):
        with patch.dict('os.environ',{},clear=True),patch('ambient_projector.app.Camera') as camera:
            self.assertEqual(main(['--scene',str(EXAMPLE),'--once','--outdir',str(self.out)]),0)
            camera.return_value.capture.assert_not_called()
        self.assertTrue((self.out/'latest.png').exists())

    def test_failure_clears_projector(self):
        with patch('ambient_projector.app.Projector') as projector:
            code=main(['--scene',str(self.out/'missing.json'),'--once',
                       '--projector-url','http://pi:8080','--outdir',str(self.out)])
        self.assertEqual(code,1); projector.return_value.blank.assert_called_once()

    def test_http_frame_bytes(self):
        response=Mock(); response.__enter__=Mock(return_value=SimpleNamespace(status=200))
        response.__exit__=Mock(return_value=False)
        frame=Image.new('RGB',(640,480),'cyan')
        with patch('ambient_projector.devices.urlopen',return_value=response) as send:
            Projector('http://pi:8080',640,480).show(frame)
            request=send.call_args.args[0]
            self.assertEqual(request.data,frame.tobytes())
            self.assertEqual(request.full_url,'http://pi:8080/raw?w=640&h=480')

if __name__=='__main__': unittest.main()
