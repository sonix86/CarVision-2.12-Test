import unittest
import numpy as np
import torch
from carvision.family_head import FamilyHead
from carvision.semantic_make import CosineMakeHead
from carvision.scene import model_visibility_limited

class FamilyHeadTests(unittest.TestCase):
    def head(self):
        h=CosineMakeHead(torch.eye(2));ck=dict(origin='project-trained-family-head-v30',pairs=[('BYD','Song Plus'),('Dongfeng','Aeolus E70')],head=h.state_dict(),thresholds={'BYD|Song Plus':.95,'Dongfeng|Aeolus E70':.7},temperature=1.,margin=.06,clipped_validation_passed=True,global_margin_required=True,direct_manufacturer_fallback=True,similarity_thresholds={'BYD|Song Plus':.4,'Dongfeng|Aeolus E70':.3})
        return FamilyHead(ck,2)
    def test_high_score_does_not_enable_an_uncalibrated_class(self):
        h=self.head();h.thresholds['BYD|Song Plus']=1.;r=h.predict(torch.tensor([1.,0.]),'BYD',(200,150));self.assertEqual(r['model'],'unknown')
    def test_model_cannot_change_a_different_accepted_manufacturer(self):
        h=self.head();r=h.predict(torch.tensor([1.,0.]),'Dongfeng',(200,150));self.assertEqual(r['model'],'unknown');self.assertEqual(r['model_candidate_brand'],'Dongfeng')
    def test_global_competitor_prevents_false_conditional_certainty(self):
        h=self.head();h.thresholds['BYD|Song Plus']=.5;r=h.describe(torch.tensor([.51,.49]),'BYD',(200,150),torch.ones(2));self.assertEqual(r['model'],'unknown')
    def test_absolute_feature_similarity_is_required(self):
        h=self.head();r=h.describe(torch.tensor([.99,.01]),'BYD',(200,150),torch.tensor([.1,.1]));self.assertEqual(r['model'],'unknown')
    def test_validated_family_can_establish_manufacturer_on_abstention(self):
        r=self.head().predict_make(torch.tensor([0.,1.]),['BYD','Dongfeng'],(128,128),np.ones((128,128),bool),.5);self.assertEqual(r['brand'],'Dongfeng')
    def test_user_threshold_one_still_forbids_confirmation(self):
        self.assertIsNone(self.head().predict_make(torch.tensor([0.,1.]),['BYD','Dongfeng'],(128,128),np.ones((128,128),bool),1.))
    def test_targeted_repair_does_not_enable_other_manufacturer_fallbacks(self):
        h=self.head();h.fallback_brands=['Dongfeng'];self.assertIsNone(h.predict_make(torch.tensor([1.,0.]),['BYD','Dongfeng'],(128,128),np.ones((128,128),bool),.5))
    def test_bad_fragment_never_uses_family_make_fallback(self):
        self.assertIsNone(self.head().predict_make(torch.tensor([0.,1.]),['BYD','Dongfeng'],(128,128),np.zeros((128,128),bool),.5))
    def test_border_exception_requires_a_validated_accepted_family(self):
        self.assertTrue(model_visibility_limited({},[],True));self.assertTrue(model_visibility_limited(dict(model_clipping_validated=True,model_family_accepted=False),[],True));self.assertFalse(model_visibility_limited(dict(model_clipping_validated=True,model_family_accepted=True),[],True))
    def test_border_exception_never_hides_poor_detail_warning(self):
        self.assertTrue(model_visibility_limited(dict(model_clipping_validated=True,model_family_accepted=True),['too small'],True))
