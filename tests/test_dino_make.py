"""Acceptance and taxonomy safety for our optional DINO decision head."""
import unittest
import numpy as np
import torch
from PIL import Image
from carvision.dino_make import DinoMakeExpert


class DinoMakeChecks(unittest.TestCase):
    def expert(self):
        e=object.__new__(DinoMakeExpert);e.brands=['BYD','Zeekr'];e.temperature=1.;e.margin=.06;e.clip=None
        e.thresholds={'BYD':1.,'Zeekr':1.};e.feature=lambda *a:torch.tensor([1.,0.]);e.head=lambda *a:torch.tensor([100.,-100.])
        e.retrieval=dict(keys=torch.tensor([[[1.,0.]],[[1.,0.]]]),labels=torch.tensor([0]),groups=torch.tensor([0]),k=1,alpha=.5,beta=50.,mode='abstention-only',fallback_thresholds={'BYD':1.,'Zeekr':1.})
        return e

    def test_uncalibrated_cannot_be_confirmed_even_when_score_is_one(self):
        e=self.expert();r=e.predict(Image.new('RGB',(100,80)),np.ones((80,100),bool))
        self.assertEqual(r['confidence'],1.);self.assertFalse(r['accepted']);self.assertEqual(r['brand'],'unknown')

    def test_primary_confirmation_retains_own_label(self):
        e=self.expert();e.thresholds['BYD']=.9
        r=e.predict(Image.new('RGB',(100,80)),np.ones((80,100),bool))
        self.assertTrue(r['accepted']);self.assertEqual(r['brand'],'BYD')

    def test_minimum_and_geometry_still_gate_a_new_backbone(self):
        e=self.expert();e.thresholds['BYD']=.9
        self.assertFalse(e.predict(Image.new('RGB',(30,30)),np.ones((30,30),bool))['accepted'])
        with self.assertRaises(ValueError):e.predict(Image.new('RGB',(100,80)),np.ones((80,100),bool),float('nan'))

    def test_foreign_classifier_checkpoint_is_rejected(self):
        with self.assertRaises(ValueError):DinoMakeExpert(dict(origin='ready-made-car-classifier'))
