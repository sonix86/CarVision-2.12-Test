import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from carvision.body_expert import body_canvas,body_tensor,BodyExpert
from carvision.vehicle_model import VehicleModelClassifier


class BodyExpertTests(unittest.TestCase):
    def test_geometry_and_small_images_share_full_canvas_scale(self):
        for size in [(300,100),(30,10)]:
            a=np.asarray(body_canvas(Image.new('RGB',size,'white')))
            visible=(a==255).all(2);y,x=np.where(visible)
            self.assertEqual(x.max()-x.min()+1,224)
            self.assertAlmostEqual((x.max()-x.min()+1)/(y.max()-y.min()+1),3.,delta=.03)
            self.assertLessEqual(abs(y.min()-(223-y.max())),1)

    def test_preprocessing_never_reads_a_file_or_network(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('network')):
            t=body_tensor(Image.new('RGB',(40,80),'red'))
        self.assertEqual(tuple(t.shape),(3,224,224));self.assertTrue(torch.isfinite(t).all())

    def test_reject_wrong_taxonomy_before_loading_network(self):
        with self.assertRaisesRegex(ValueError,'taxonomy'):
            BodyExpert(dict(origin='project-trained-body-expert-v1',pairs=[['BMW','X']]),[('Chevrolet','Cobalt')])

    def test_fusion_keeps_competing_manufacturers_and_normalization(self):
        m=object.__new__(VehicleModelClassifier)
        class Expert:
            weight=.5
            def probabilities(self,image):return torch.tensor([.2,.6,.2])
        m.body_expert=Expert();p=m._body_blend(torch.tensor([.8,.1,.1]),Image.new('RGB',(20,20)))
        self.assertTrue(torch.allclose(p,torch.tensor([.5,.35,.15])))
        self.assertAlmostEqual(float(p.sum()),1.)

    def classify_make_conflict(self,old_make,confidence):
        from types import SimpleNamespace
        from carvision.scene import classify
        class Make:
            def __call__(self,x):return torch.tensor([[9.,0.]]).repeat(len(x),1),None
        class Vehicle:
            body_expert=object();brands=['Chevrolet','BMW'];references=SimpleNamespace(threshold=.8)
            def probabilities(self,*args):return torch.tensor([confidence,1-confidence])
            def describe(self,p,brand):
                return dict(model='Cobalt' if brand=='Chevrolet' else 'unknown',model_candidate='Cobalt',model_candidate_brand='Chevrolet',model_confidence=confidence,model_ranking=[dict(brand='Chevrolet',model='Cobalt',confidence=confidence),dict(brand='BMW',model='3 Series',confidence=1-confidence)])
        p=SimpleNamespace(classifier=Make(),brands=[old_make,'Chevrolet'],preprocess=lambda im:torch.zeros(3,10,10),mask_fusion=dict(brand_mask_weight=1.,model_mask_weight=.25),brand_temperature=1.,brand_threshold=.5,vehicle_classifier=Vehicle(),colour_reliability_threshold=.5,colour_classifier=None)
        return classify(p,Image.new('RGB',(120,80),'white'),np.ones((80,120),bool))

    def test_only_strong_expert_corrects_older_supported_make(self):
        row=self.classify_make_conflict('BMW',.95)
        self.assertEqual(row['brand'],'Chevrolet');self.assertEqual(row['model'],'Cobalt')
        self.assertIn('calibrated correction',row['brand_source'])
        self.assertEqual(self.classify_make_conflict('BMW',.89)['brand'],'unknown')

    def test_closed_model_vocabulary_does_not_erase_unrepresented_make(self):
        row=self.classify_make_conflict('Audi',.95)
        self.assertEqual(row['brand'],'unknown')
        self.assertEqual(row['classifier_brand'],'Audi')
        self.assertTrue(row['brand_model_conflict'])

    def test_make_can_be_confident_while_exact_chevrolet_model_is_uncertain(self):
        from types import SimpleNamespace
        from carvision.scene import classify
        class Make:
            def __call__(self,x):return torch.tensor([[9.,0.]]).repeat(len(x),1),None
        vehicle=object.__new__(VehicleModelClassifier)
        vehicle.body_expert=object();vehicle.labels=['Cobalt','Spark','Captiva','3 Series'];vehicle.brands=['Chevrolet','Chevrolet','Chevrolet','BMW']
        vehicle.references=SimpleNamespace(threshold=.8,calibrated=lambda p:p)
        vehicle.probabilities=lambda *args:torch.tensor([.45,.35,.15,.05])
        p=SimpleNamespace(classifier=Make(),brands=['BMW','Chevrolet'],preprocess=lambda im:torch.zeros(3,10,10),mask_fusion=dict(brand_mask_weight=1.,model_mask_weight=.25),brand_temperature=1.,brand_threshold=.5,vehicle_classifier=vehicle,colour_reliability_threshold=.5,colour_classifier=None)
        row=classify(p,Image.new('RGB',(120,80),'white'),np.ones((80,120),bool))
        self.assertEqual(row['brand'],'Chevrolet');self.assertAlmostEqual(row['brand_confidence'],.95,places=5)
        self.assertEqual(row['model'],'unknown');self.assertEqual(row['model_candidate'],'Cobalt')
        self.assertEqual(row['brand_alternatives'][0]['brand'],'Chevrolet')

if __name__=='__main__':unittest.main()
