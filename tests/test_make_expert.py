import unittest
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from carvision.make_expert import model_details_for_make
from carvision.make_expert import MakeExpert
from carvision.vehicle_model import vehicle_name


class MakeFirstChecks(unittest.TestCase):
    def test_export_uses_make_variants_when_make_is_uncertain(self):
        from carvision.vehicle_model import detailed_answer
        row=dict(make_first=True,brand='unknown',colour='white',make_evidence=dict(candidate='BYD',accepted=False,alternatives=[dict(brand='BYD',confidence=.6),dict(brand='Zeekr',confidence=.3)]),model_alternatives=[dict(brand='Chevrolet',model='Captiva',confidence=.9)])
        text=detailed_answer(row)
        self.assertIn('BYD 60%',text);self.assertNotIn('Captiva',text)

    def test_fragment_limits_model_without_erasing_confirmed_make(self):
        import time
        from unittest.mock import patch
        from carvision.scene import predict_instances
        mask=np.ones((180,240),bool)
        p=SimpleNamespace(segmenter=SimpleNamespace(instances=lambda *a:([dict(mask=mask,score=.99)],1)),detection=.5,mode='fast',classifier=object(),brands=['Chevrolet'],make_expert=SimpleNamespace(brands=['Zeekr']))
        predicted=dict(make_first=True,brand='Zeekr',model='001',colour='white',make_evidence=dict(accepted=True,candidate='Zeekr'))
        with patch('carvision.scene.classify',return_value=predicted):
            result=predict_instances(p,Image.new('RGB',(240,180)),'boundary.jpg',None,False,time.perf_counter())
        row=result['cars'][0]
        self.assertEqual(row['brand'],'Zeekr');self.assertEqual(row['model'],'unknown');self.assertTrue(row['model_limited'])

    def test_uncalibrated_make_cannot_be_confirmed_even_if_softmax_rounds_to_one(self):
        expert=object.__new__(MakeExpert);expert.brands=['BYD','Zeekr'];expert.temperature=1.;expert.margin=.1
        expert.thresholds={'BYD':1.,'Zeekr':1.};expert.feature=lambda *a:torch.zeros(512)
        expert.head=lambda x:torch.tensor([100.,-100.])
        result=expert.predict(Image.new('RGB',(100,80)))
        self.assertEqual(result['confidence'],1.);self.assertFalse(result['accepted']);self.assertEqual(result['brand'],'unknown')
        self.assertIn('независимых',result['acceptance_reason'])
        expert.thresholds['BYD']=.8
        self.assertTrue(expert.predict(Image.new('RGB',(100,80)),minimum=.9)['accepted'])
        self.assertFalse(expert.predict(Image.new('RGB',(100,80)),minimum=1.)['accepted'])
        with self.assertRaises(ValueError):expert.predict(Image.new('RGB',(100,80)),minimum=float('nan'))

    def test_same_make_models_keep_global_confidence(self):
        row=dict(model='unknown',model_candidate='Captiva',model_candidate_brand='Chevrolet',model_confidence=.4,
                 model_ranking=[dict(brand='Chevrolet',model='Captiva',confidence=.4),dict(brand='BYD',model='Seal',confidence=.3),dict(brand='BYD',model='Han',confidence=.2)])
        result=model_details_for_make(row,'BYD')
        self.assertEqual(result['model'],'unknown')
        self.assertEqual(result['model_candidate'],'Seal')
        self.assertEqual(result['model_confidence'],.3)
        self.assertEqual([r['brand'] for r in result['model_alternatives']],['BYD','BYD'])
        self.assertEqual(row['model_candidate'],'Captiva')

    def test_new_make_does_not_borrow_model_from_other_make(self):
        row=dict(model='Cobalt',model_candidate='Cobalt',model_candidate_brand='Chevrolet',model_ranking=[dict(brand='Chevrolet',model='Cobalt',confidence=.99)])
        result=model_details_for_make(row,'Xiaomi')
        self.assertEqual(result['model'],'unknown');self.assertEqual(result['model_alternatives'],[])
        self.assertIn('model_reason',result)

    def test_primary_text_preserves_confirmed_make_when_model_missing(self):
        row=dict(make_first=True,brand='Li Auto',model='unknown',model_candidate='Captiva',model_candidate_brand='Chevrolet',model_limited=True)
        self.assertEqual(vehicle_name(row),'Li Auto · модель не определена')
        self.assertEqual(vehicle_name(dict(make_first=True,brand='unknown',make_evidence=dict(candidate='Zeekr'))),'Возможно Zeekr · марка не подтверждена')

    def test_independent_make_survives_conflicting_model(self):
        from carvision.scene import classify
        class Model:
            references=None;threshold=.8;body_expert=None
            def probabilities(self,*args):return torch.tensor([.85,.15])
            def describe(self,prob,brand):
                return dict(model='Captiva' if brand=='Chevrolet' else 'unknown',model_candidate='Captiva',model_candidate_brand='Chevrolet',model_confidence=.85,
                            model_ranking=[dict(brand='Chevrolet',model='Captiva',confidence=.85),dict(brand='Zeekr',model='001',confidence=.15)],model_alternatives=[])
        evidence=dict(brand='Zeekr',candidate='Zeekr',accepted=True,confidence=.97,source='test independent make',alternatives=[dict(brand='Zeekr',confidence=.97)])
        p=SimpleNamespace(classifier=None,vehicle_classifier=Model(),brands=[],brand_threshold=.5,preprocess=None,
                          make_expert=SimpleNamespace(predict=lambda *a:evidence),mask_fusion=dict(model_mask_weight=.25),
                          colour_classifier=None,colour_reliability_threshold=.5)
        row=classify(p,Image.new('RGB',(80,60),'white'),np.ones((60,80),bool))
        self.assertEqual(row['brand'],'Zeekr');self.assertEqual(row['model'],'unknown')
        self.assertEqual(row['model_candidate'],'001')
        self.assertEqual(vehicle_name(row),'Zeekr · модель не определена')
        evidence.update(brand='unknown',candidate='Zeekr',accepted=False,confidence=.5)
        row=classify(p,Image.new('RGB',(80,60),'white'),np.ones((60,80),bool))
        self.assertEqual(row['brand'],'unknown');self.assertEqual(row['model'],'unknown')
        self.assertIn('Возможно Zeekr',vehicle_name(row))
