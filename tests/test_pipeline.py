"""Engineering checks on synthetic images, never presented as model accuracy."""
import json
import tempfile
import unittest
import socket
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import torch
from carvision.core import Pipeline, heuristic_colour, Classifier, ROOT,detection_tiles,restore_tile_boxes,save_prediction
from carvision.data import load_records,split_records
from carvision.scoring import photo_score

class Checks(unittest.TestCase):
    def test_assignment_scoring(self):
        self.assertEqual(photo_score([('Chevrolet','white')],[('Chevrolet','white')]),5)
        self.assertEqual(photo_score([('Chevrolet','white')],[('Chevrolet','black')]),3)
        self.assertEqual(photo_score([('Chevrolet','white')],[('Kia','white')]),2)
        self.assertEqual(photo_score([('Chevrolet','white')],[('Chevrolet','white'),('Kia','grey')]),4)
        self.assertEqual(photo_score([('Chevrolet','white'),('Chevrolet','white'),('BYD','blue')],[('Chevrolet','white'),('Chevrolet','white'),('BYD','black')]),4.3)
        self.assertEqual(photo_score([('BYD','blue'),('Kia','red')],[('Kia','red'),('BYD','blue')]),5)
        self.assertEqual(photo_score([('Mercedes-Benz','white'),('Volkswagen','grey'),('Li Auto','black')],[('Mercedes','white'),('VW','grey'),('Lixiang','black')]),5)

    def test_brand_and_colour_are_inferred_without_manual_brand_input(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'new-photo.jpg';Image.new('RGB',(400,300),'white').save(path)
            pipeline=object.__new__(Pipeline);pipeline.mode='fast';pipeline.detection=.5;pipeline.brand_threshold=.65;pipeline.colour_mode='trained';pipeline.brands=['Kia','BYD']
            pipeline.detector=lambda images:[dict(boxes=torch.tensor([[20.,30,320,260]]),scores=torch.tensor([.95]),labels=torch.tensor([1]))]
            pipeline.classifier=lambda x:(torch.tensor([[0.,10.]]),torch.tensor([[0.,0.,0.,0.,10.,0.,0.,0.]]))
            result=pipeline.predict(path)
            self.assertEqual([(r['brand'],r['colour']) for r in result['cars']],[('BYD','blue')]);self.assertEqual(result['cars'][0]['colour_source'],'trained')

    def test_colour_palette(self):
        for name,rgb in [('white',(245,245,245)),('black',(10,10,10)),('grey',(110,110,110)),('red',(220,20,20)),('blue',(20,40,230)),('green',(20,160,30)),('brown',(120,70,30)),('yellow',(240,200,20))]:
            self.assertEqual(heuristic_colour(Image.new('RGB',(100,100),rgb))[0],name)

    def test_inference_has_no_network_and_no_untrained_brands(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'blank.png';Image.new('RGB',(640,480),'white').save(path)
            pipeline=object.__new__(Pipeline);pipeline.mode='balanced';pipeline.detection=.5;pipeline.classifier=None;pipeline.brand_threshold=.65;pipeline.colour_mode='trained'
            pipeline.detector=lambda images:[dict(boxes=torch.empty((0,4)),scores=torch.empty(0),labels=torch.empty(0,dtype=torch.long)) for _ in images]
            with patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden')):result=pipeline.predict(path)
            self.assertFalse(result['classifier_trained']);self.assertEqual(result['cars'],[])

    def test_team_detector_required(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):Pipeline(temp)

    def test_tile_mapping_and_seam_rejection(self):
        d=dict(boxes=torch.tensor([[30.,40,160,140],[0.,50,60,110]]),scores=torch.tensor([.8,.9]),labels=torch.tensor([1,1]))
        boxes,scores=restore_tile_boxes(d,(500,0,1268,768),1600,1000,.5)
        self.assertEqual(boxes.tolist(),[[530,40,660,140]])

    def test_tiled_predictions_deduplicate_and_keep_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'street.png';Image.new('RGB',(1600,900),'white').save(path);tiles=detection_tiles(1600,900);remaining=iter(tiles)
            def detect(images):
                results=[]
                for image in images:
                    x,y,r,b=next(remaining);global_box=[800,300,1100,500]
                    if x<=800 and y<=300 and r>=1100 and b>=500:results.append(dict(boxes=torch.tensor([[800-x,300-y,1100-x,500-y]],dtype=torch.float32),scores=torch.tensor([.85]),labels=torch.tensor([1])))
                    else:results.append(dict(boxes=torch.empty((0,4)),scores=torch.empty(0),labels=torch.empty(0,dtype=torch.long)))
                return results
            p=object.__new__(Pipeline);p.mode='balanced';p.detection=.5;p.brand_threshold=.65;p.classifier=None;p.colour_mode='trained';p.detector=detect
            result=p.predict(path);self.assertEqual(len(result['cars']),1);self.assertEqual(result['cars'][0]['box'],[800,300,1100,500]);self.assertGreater(result['detector_views'],1)

    def test_exports_do_not_overwrite_same_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);photo=root/'same.png';Image.new('RGB',(32,32),'white').save(photo)
            result=dict(cars=[],classifier_trained=False)
            a=save_prediction(photo,result,root/'out');b=save_prediction(photo,result,root/'out')
            self.assertNotEqual(a,b);self.assertTrue(a.exists() and b.exists())

    def test_duplicate_images_stay_together(self):
        rows=[dict(brand='BYD' if i%2 else 'Kia',group=str(i//2),hash=str(i//2),image=str(i)) for i in range(20)]
        tr,va=split_records(rows)
        self.assertFalse({r['hash'] for r in tr}&{r['hash'] for r in va})

    def test_invalid_box_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);Image.new('RGB',(10,10)).save(d/'car.png')
            row=dict(image='car.png',box=[0,0,20,20],brand='BYD',colour='white',source='synthetic test',group='one')
            (d/'labels.jsonl').write_text(json.dumps(row),encoding='utf-8')
            with self.assertRaises(ValueError):load_records(d/'labels.jsonl')

if __name__=='__main__':unittest.main()
