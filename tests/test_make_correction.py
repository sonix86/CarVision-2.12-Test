import unittest
import numpy as np
import torch
from carvision.make_correction import MakeCorrection
from carvision.semantic_make import CosineMakeHead

class CorrectionChecks(unittest.TestCase):
    def make(self):
        h=CosineMakeHead(torch.eye(2))
        return MakeCorrection(dict(origin='project-trained-phone-make-correction-v32',brands=['Leapmotor','Li Auto'],head=h.state_dict(),scope=['Leapmotor'],thresholds={'Leapmotor':.95,'Li Auto':1.},temperature=1.,margin=.06,minimum_size=64),['Leapmotor','Li Auto'],2)
    def test_scoped_head_corrects_from_global_probability(self):
        r=self.make().predict(torch.tensor([1.,0.]),(160,120),np.ones((120,160),bool),.5)
        self.assertEqual(r['brand'],'Leapmotor')
    def test_other_top_class_never_enables_correction(self):
        self.assertIsNone(self.make().predict(torch.tensor([0.,1.]),(160,120),np.ones((120,160),bool),.5))
    def test_user_threshold_one_cannot_be_overridden(self):
        self.assertIsNone(self.make().predict(torch.tensor([1.,0.]),(160,120),np.ones((120,160),bool),1.))
    def test_geometry_rejects_occluded_or_small_fragments(self):
        h=self.make()
        self.assertIsNone(h.predict(torch.tensor([1.,0.]),(160,120),np.zeros((120,160),bool),.5))
        self.assertIsNone(h.predict(torch.tensor([1.,0.]),(30,30),np.ones((30,30),bool),.5))
    def test_disabled_gate_beats_high_score(self):
        h=self.make();h.thresholds['Leapmotor']=1.
        self.assertIsNone(h.predict(torch.tensor([1.,0.]),(160,120),np.ones((120,160),bool),.5))
