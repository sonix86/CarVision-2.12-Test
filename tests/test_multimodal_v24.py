import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from carvision.multimodal_make import make_inputs,MultimodalMakeExpert
from carvision.vehicle_text import text_features,VehicleTextReader
from carvision.illumination_colour import features,scene_gains


class MultimodalChecks(unittest.TestCase):
    def test_runtime_and_training_preprocessing_match(self):
        from tools.cache_multimodal_v24 import inputs
        a=np.random.default_rng(24).integers(0,256,(90,170,3),dtype=np.uint8)
        mask=np.zeros((90,170),bool);mask[10:80,20:150]=True
        row=dict(body_rgb=torch.from_numpy(a),body_mask=torch.from_numpy(mask))
        for kind in ('whole','phone'):
            self.assertTrue(torch.equal(inputs(row,kind),make_inputs(Image.fromarray(a),mask,kind)))

    def test_plate_and_watermark_are_not_make_features(self):
        for word in ('01Z294JB','https://cars.com','copyright','telegram123'):
            self.assertFalse(text_features([dict(text=word,confidence=.99)]).any())
        a=text_features([dict(text='BUILD YOUR DREAMS',confidence=.9)])
        self.assertTrue(a.any());np.testing.assert_array_equal(a,text_features([dict(text='build your dreams',confidence=.9)]))

    def test_ocr_excludes_neighbour_text_and_plate(self):
        reader=object.__new__(VehicleTextReader)
        reader.engine=lambda a:([([[0,0],[20,0],[20,20],[0,20]],'BYD',.99),([[90,90],[180,90],[180,150],[90,150]],'01Z294JB',.99),([[90,90],[180,90],[180,150],[90,150]],'SONG',.99)],None)
        mask=np.zeros((100,100),bool);mask[25:80,25:80]=True
        result=reader.read(Image.new('RGB',(100,100)),mask)
        self.assertEqual([r['text'] for r in result],['SONG'])

    def test_neutral_gains_require_background_and_are_bounded(self):
        image=Image.new('RGB',(100,100),(150,140,125))
        np.testing.assert_array_equal(scene_gains(image,np.ones((100,100),bool)),np.ones(3))
        gains=scene_gains(image,np.zeros((100,100),bool))
        self.assertTrue(np.isfinite(gains).all());self.assertTrue(((gains>=.75)&(gains<=1.33)).all());self.assertLess(gains[0],gains[2])

    def test_paint_features_use_tight_foreground_with_explicit_scene_gains(self):
        image=Image.new('RGB',(120,90),(80,90,100));mask=np.zeros((90,120),bool);mask[20:70,30:100]=True
        full=features(image,mask,np.ones(3,dtype=np.float32))
        crop=features(image.crop((30,20,100,70)),np.ones((50,70),bool),np.ones(3,dtype=np.float32))
        self.assertTrue(torch.equal(full,crop))

    def test_disabled_threshold_does_not_confirm_softmax_one(self):
        expert=object.__new__(MultimodalMakeExpert);expert.encoder=lambda x:torch.ones((2,768));expert.mean=torch.zeros(1536);expert.std=torch.ones(1536)
        expert.head=lambda x:torch.tensor([100.,-100.]);expert.temperature=1.;expert.thresholds={'BYD':1.,'Zeekr':1.};expert.brands=['BYD','Zeekr'];expert.margin=.06;expert.text_reader=None
        image=Image.new('RGB',(90,90));mask=np.ones((90,90),bool)
        self.assertFalse(expert.predict(image,mask)['accepted']);expert.thresholds['BYD']=.8
        self.assertTrue(expert.predict(image,mask)['accepted'])
        with self.assertRaises(ValueError):expert.predict(image,mask,float('nan'))

if __name__=='__main__':unittest.main()
