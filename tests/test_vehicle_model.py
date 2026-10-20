import unittest
import torch
from torch import nn
from PIL import Image
from carvision.vehicle_model import VehicleModelClassifier,vehicle_name

class ModelChecks(unittest.TestCase):
    def checkpoint(self):
        head=nn.Linear(2,2)
        with torch.no_grad():head.weight.zero_();head.bias.copy_(torch.tensor([5.,0.]))
        return dict(origin='project-trained-regional-model-head',labels=['Cobalt','Matiz'],brands=['Chevrolet','Daewoo'],mean=torch.zeros(2),std=torch.ones(2),state=head.state_dict(),temperature=1.,threshold=.5)
    def test_reject_foreign_checkpoint(self):
        c=self.checkpoint();c['origin']='ready-made-vehicle-classifier'
        with self.assertRaises(ValueError):VehicleModelClassifier(c)
    def test_brand_consistency_and_unknown(self):
        m=VehicleModelClassifier(self.checkpoint())
        class Backbone:
            def features(self,x):return torch.ones(len(x),1,2,2)
        image=Image.new('RGB',(64,48));pre=lambda im:torch.zeros(3,32,32)
        self.assertEqual(m.predict(image,'Chevrolet',Backbone(),pre)[0],'Cobalt')
        self.assertEqual(m.predict(image,'Daewoo',Backbone(),pre)[0],'unknown')
        self.assertEqual(m.predict(image,'unknown',Backbone(),pre),('unknown',0.))
    def test_ambiguous_models_do_not_become_certain_by_brand_filtering(self):
        c=self.checkpoint();c['state']['bias'].zero_();c['threshold']=.6
        m=VehicleModelClassifier(c)
        class Backbone:
            def features(self,x):return torch.ones(len(x),1,2,2)
        result=m.predict(Image.new('RGB',(64,48)),'Chevrolet',Backbone(),lambda im:torch.zeros(3,32,32))
        self.assertEqual(result[0],'unknown');self.assertAlmostEqual(result[1],.5)
    def test_full_name_and_old_result_compatibility(self):
        self.assertEqual(vehicle_name(dict(brand='Chevrolet',model='Cobalt')),'Chevrolet Cobalt')
        self.assertEqual(vehicle_name(dict(brand='Chevrolet')),'Chevrolet')

    def test_uncertain_candidate_is_marked_in_all_text_names(self):
        self.assertEqual(vehicle_name(dict(brand='Chevrolet',model='unknown',model_candidate='Cobalt')),'Chevrolet Cobalt?')
