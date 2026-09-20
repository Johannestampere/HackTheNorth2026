"""Calibration, real rendering and Pi wire format, without operating hardware."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from companion.app import main
from companion.pool.contracts.serialization import load_shot_plan
from companion.pool.projection.calibration import from_corners, map_point
from companion.pool.projection.models import load_projection_target
from companion.pool.projection.service import ProjectionService, clip_segment
from companion.hardware.http_projector import HttpProjector

ROOT = Path(__file__).resolve().parents[2]

class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.target = load_projection_target(ROOT/'fixtures/projection_targets/synthetic.json')
        self.plan = load_shot_plan(ROOT/'fixtures/shot_plans/direct_shot.json')

    def test_four_corners_and_interior_round_trip(self):
        fitted = from_corners(self.target.geometry.width, self.target.corners_px)
        for x,y in ((0,0),(1,0),(1,.5),(0,.5),(.3,.2)):
            expected = map_point(self.target.table_to_pixel,x,y)
            actual = map_point(fitted,x,y)
            self.assertAlmostEqual(actual[0],expected[0])
            self.assertAlmostEqual(actual[1],expected[1])

    def test_invalid_corner_order_and_degenerate_quad(self):
        for corners in (((0,0),(10,10),(10,0),(0,10)), ((0,0),(1,0),(2,0),(3,0))):
            with self.assertRaises(ValueError):
                from_corners(.5,corners)

    def test_real_renderer_hits_calibrated_cue_and_preserves_provenance(self):
        frame = ProjectionService().render(self.plan,self.target)
        p = self.plan.cue_aim.origin
        x,y = map_point(self.target.table_to_pixel,p.x,p.y)
        offset = (round(y)*frame.width_px+round(x))*3
        self.assertEqual(frame.rgb[offset:offset+3],bytes((255,255,255)))
        self.assertEqual(frame.calibration_id,self.target.calibration_id)
        self.assertEqual(frame.rgb[:3],bytes(3))
        self.assertEqual(len(frame.rgb),frame.width_px*frame.height_px*3)

    def test_aim_only_is_visible_and_wrong_frame_rejected(self):
        plan = load_shot_plan(ROOT/'fixtures/shot_plans/aim_only.json')
        self.assertTrue(any(ProjectionService().render(plan,self.target).rgb))
        with self.assertRaises(ValueError):
            ProjectionService().render(replace(plan,table_id='other'),self.target)

    def test_horizon_crossing_rejected(self):
        target = replace(self.target,table_to_pixel=((1,0,0),(0,1,0),(2,0,-1)))
        with self.assertRaises(ValueError):
            ProjectionService().render(self.plan,target)

    def test_offscreen_lines_clipped(self):
        segment = clip_segment((-100,5),(100,5),10,10)
        for actual, expected in zip(segment, ((0,5),(10,5))):
            for a,b in zip(actual,expected):
                self.assertAlmostEqual(a,b)
        self.assertIsNone(clip_segment((-10,-5),(-1,-5),10,10))

    def test_png_and_cli_preview_never_send_without_explicit_url(self):
        with tempfile.TemporaryDirectory() as directory, patch('companion.app.HttpProjector') as sender:
            path = Path(directory)/'preview.png'
            self.assertEqual(main(['render','--plan',str(ROOT/'fixtures/shot_plans/direct_shot.json'),
                '--target',str(ROOT/'fixtures/projection_targets/synthetic.json'),'--output',str(path)]),0)
            self.assertEqual(path.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            sender.assert_not_called()

    def test_http_payload_blank_and_failures(self):
        projector = HttpProjector('http://pi:8080',2,1)
        response = MagicMock()
        response.__enter__.return_value.status = 200
        with patch('companion.hardware.http_projector.urlopen',return_value=response) as send:
            projector.project(bytes([1,2,3,4,5,6]),2,1)
            request = send.call_args.args[0]
            self.assertEqual(request.full_url,'http://pi:8080/raw?w=2&h=1')
            self.assertEqual(request.data,bytes([1,2,3,4,5,6]))
            projector.blank()
            self.assertEqual(send.call_args.args[0].data,bytes(6))
        with self.assertRaises(ValueError):
            projector.project(bytes(6),1,2)
        with patch('companion.hardware.http_projector.urlopen',side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                projector.project(bytes(6),2,1)

    def test_explicit_cli_send_uses_rendered_frame(self):
        with tempfile.TemporaryDirectory() as directory, patch('companion.app.HttpProjector') as sender:
            main(['render','--plan',str(ROOT/'fixtures/shot_plans/direct_shot.json'),
                  '--target',str(ROOT/'fixtures/projection_targets/synthetic.json'),
                  '--output',str(Path(directory)/'preview.png'),'--display-url','http://pi:8080'])
            sender.assert_called_once_with('http://pi:8080',800,400)
            frame = ProjectionService().render(self.plan,self.target)
            sender.return_value.project.assert_called_once_with(frame.rgb,800,400)

    def test_calibration_cli_retains_frame_identity(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            corners, output = Path(directory)/'corners.json', Path(directory)/'target.json'
            corners.write_text(json.dumps(self.target.corners_px))
            main(['calibrate','--geometry',str(ROOT/'fixtures/table_geometry.json'),
                  '--corners',str(corners),'--width-px','800','--height-px','400',
                  '--calibration-id','test','--pose-id','fixed','--output',str(output)])
            target = load_projection_target(output)
            self.assertEqual(target.calibration_id,'test')
            self.assertEqual(target.pose_id,'fixed')
            for a,b in zip(target.corners_px,self.target.corners_px):
                self.assertAlmostEqual(a[0],b[0])
                self.assertAlmostEqual(a[1],b[1])
