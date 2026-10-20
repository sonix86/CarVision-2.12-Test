"""Engineering checks on synthetic images, never presented as model accuracy."""
import json
import tempfile
import unittest
import socket
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from carvision.core import Pipeline, heuristic_colour, Classifier, ROOT
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

    def test_colour_palette(self):
        for name,rgb in [('white',(245,245,245)),('black',(10,10,10)),('grey',(110,110,110)),('red',(220,20,20)),('blue',(20,40,230)),('green',(20,160,30)),('brown',(120,70,30)),('yellow',(240,200,20))]:
            self.assertEqual(heuristic_colour(Image.new('RGB',(100,100),rgb))[0],name)

    def test_inference_has_no_network_and_no_untrained_brands(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'blank.png';Image.new('RGB',(640,480),'white').save(path)
            with patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden')):
                pipeline=Pipeline(); result=pipeline.predict(path)
            self.assertFalse(result['classifier_trained']);self.assertEqual(result['cars'],[])

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
