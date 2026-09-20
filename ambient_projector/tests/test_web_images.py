from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from ambient_projector.ai import SceneAI
from ambient_projector.app import Companion
from ambient_projector.render import Renderer
from ambient_projector.web_images import WebImages, fullscreen, check_url


class WebImageTests(unittest.TestCase):
    def test_empty_specific_query_broadens_before_giving_up(self):
        search=WebImages()
        with patch.object(search,'_search_once',side_effect=[[],[],[{'title':'Pool'}]]) as call:
            result=search.search('pool table billiard balls arrangement diagram')
        self.assertEqual([c.args[0] for c in call.call_args_list],
                         ['pool table billiard balls arrangement diagram','pool table billiard balls','pool table'])
        self.assertEqual(result[0]['requested_query'],'pool table billiard balls arrangement diagram')

    def test_fit_does_not_crop_and_flattens_to_rgb(self):
        frame=fullscreen(Image.new('RGB',(100,200),'red'),640,480)
        self.assertEqual(frame.size,(640,480))
        self.assertEqual(frame.getpixel((0,0)),(0,0,0))
        self.assertEqual(frame.getpixel((320,0)),(255,0,0))
        self.assertEqual(frame.getpixel((320,479)),(255,0,0))

    def test_search_uses_real_results_and_preserves_metadata(self):
        page={'title':'File:Brain.svg','index':1,'imageinfo':[{
            'mime':'image/svg+xml','thumburl':'https://upload.wikimedia.org/brain.png',
            'descriptionurl':'https://commons.wikimedia.org/wiki/File:Brain.svg',
            'thumbwidth':1000,'thumbheight':800,'extmetadata':{
                'Artist':{'value':'<b>Author</b>'},'LicenseShortName':{'value':'CC BY 4.0'}}}]}
        with patch('ambient_projector.web_images.fetch',return_value=json.dumps({'query':{'pages':{'1':page}}}).encode()):
            result=WebImages().search('brain')
        self.assertEqual(result[0]['author'],'Author')
        self.assertEqual(result[0]['license'],'CC BY 4.0')
        self.assertTrue(result[0]['image_url'].endswith('.png'))

    def test_empty_search_and_untrusted_download_hosts_rejected(self):
        with patch('ambient_projector.web_images.fetch',return_value=b'{}'),self.assertRaises(RuntimeError):
            WebImages().search('nothing')
        for url in ('http://upload.wikimedia.org/a','https://127.0.0.1/a',
                    'https://upload.wikimedia.org.evil.com/a','file:///etc/passwd'):
            with self.assertRaises(ValueError): check_url(url)

    def test_transparency_is_white_and_image_data_validated(self):
        data=BytesIO();Image.new('RGBA',(32,32),(0,0,0,0)).save(data,format='PNG')
        with patch('ambient_projector.web_images.fetch',return_value=data.getvalue()):
            image=WebImages().download({'image_url':'https://upload.wikimedia.org/test.png'})
        self.assertEqual(image.getpixel((0,0)),(255,255,255))
        with patch('ambient_projector.web_images.fetch',return_value=b'not an image'),self.assertRaises(OSError):
            WebImages().download({'image_url':'https://upload.wikimedia.org/test.png'})

    def test_camera_to_query_to_search_to_project(self):
        ai=Mock();ai.image_query.return_value='brain';ai.choose_image.return_value=1
        search=Mock(); search.search.return_value=[{'title':'bad'},{'title':'Brain','source_url':'https://commons.wikimedia.org/wiki/File:Brain'}]
        search.download.return_value=Image.new('RGB',(1000,800),'green')
        camera=Mock();camera.capture.return_value=Image.new('RGB',(50,50))
        projector=Mock()
        with tempfile.TemporaryDirectory() as path:
            app=Companion(ai,Renderer(),path,projector,camera,image_search=search)
            result=app.cycle()
            self.assertEqual(result['title'],'Brain')
            self.assertTrue((Path(path)/'source.txt').exists())
            with Image.open(Path(path)/'latest.png') as image:
                self.assertEqual(image.tobytes(),projector.show.call_args.args[0].tobytes())
        ai.understand.assert_not_called()
        search.download.assert_called_once_with(search.search.return_value[1])

    def test_failed_download_tries_another_actual_result(self):
        search=Mock();search.search.return_value=[{'title':'bad'},{'title':'good','source_url':'source'}]
        search.download.side_effect=[OSError('404'),Image.new('RGB',(400,300))]
        with tempfile.TemporaryDirectory() as path:
            result=Companion(None,Renderer(),path,image_search=search).project_search('brain')
            self.assertEqual(result['title'],'good')
            self.assertEqual(result['selection'],'search_rank_fallback')

    def test_ai_returns_only_query_and_valid_result_index(self):
        client=Mock();ai=SceneAI(client)
        client.responses.create.return_value=SimpleNamespace(status='completed',output_text='{"query":"brain anatomy"}')
        self.assertEqual(ai.image_query(Image.new('RGB',(20,20))),'brain anatomy')
        candidates=[dict(title='Brain',description='Diagram',width=1000,height=800)]
        client.responses.create.return_value=SimpleNamespace(status='completed',output_text='{"index":0}')
        self.assertEqual(ai.choose_image('brain',candidates),0)
        client.responses.create.return_value=SimpleNamespace(status='completed',output_text='{"index":99}')
        with self.assertRaises(ValueError): ai.choose_image('brain',candidates)
