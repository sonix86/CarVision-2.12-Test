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
        self.assertEqual(m.predict(image,'unknown',Backbone(),pre)[0],'unknown')
        self.assertTrue(m.predict_details(image,'unknown',Backbone(),pre)['model_alternatives'])
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

    def test_unsupported_brand_keeps_comparison_candidates_without_confirmation(self):
        m=VehicleModelClassifier(self.checkpoint());result=m.describe(torch.tensor([.7,.3]),'Mercedes-Benz')
        self.assertEqual(result['model'],'unknown');self.assertEqual(len(result['model_alternatives']),2)
        self.assertEqual(result['model_candidate'],'Cobalt');self.assertTrue(result['model_reason'])

    def test_new_trained_manufacturer_is_not_forced_into_old_make_vocabulary(self):
        from types import SimpleNamespace
        import numpy as np
        from carvision.scene import classify
        c=self.checkpoint();c.update(brands=['Zeekr','Geely'],labels=['001','Galaxy E5'])
        class OldMake:
            def __call__(self,x):return torch.ones(len(x),1),torch.zeros(len(x),8)
            def features(self,x):return torch.ones(len(x),1,2,2)
        p=SimpleNamespace(classifier=OldMake(),brands=['Audi'],preprocess=lambda im:torch.zeros(3,32,32),brand_temperature=1.,brand_threshold=.5,vehicle_classifier=VehicleModelClassifier(c),mask_fusion={'brand_mask_weight':1.,'model_mask_weight':.25},colour_classifier=None,masked_colour=None,colour_reliability_threshold=.5)
        row=classify(p,Image.new('RGB',(80,60),'white'),np.ones((60,80),bool))
        self.assertEqual(row['brand'],'Zeekr');self.assertEqual(row['model'],'001');self.assertTrue(row['brand_source'].startswith('own model head'))

    def test_reference_expansion_uses_real_learned_bank_for_added_classes(self):
        from types import SimpleNamespace
        from carvision.core import Classifier
        c=dict(origin='project-trained-regional-model-head',architecture='reference-expanded-model-network-v1',state=Classifier(2).state_dict(),labels=['Cobalt','Matiz','Galaxy E5'],brands=['Chevrolet','Daewoo','Geely'],neural_labels=['Cobalt','Matiz'],neural_brands=['Chevrolet','Daewoo'],temperature=1.,threshold=.5)
        m=VehicleModelClassifier(c);image=Image.new('RGB',(100,80),'white')
        p=m.probabilities(image,None,None);self.assertEqual(len(p),3);self.assertEqual(float(p[2]),0.);self.assertAlmostEqual(float(p.sum()),1.,places=5)
        m.references=SimpleNamespace(weight=1.,probabilities=lambda *a:torch.tensor([.1,.1,.8]))
        self.assertAlmostEqual(float(m.probabilities(image,None,None)[2]),.8,places=5)

    def test_hybrid_preserves_prefix_and_blends_both_own_heads(self):
        from carvision.core import Classifier
        def checkpoint(labels,brands,p):
            net=Classifier(len(labels))
            with torch.no_grad():net.brand.weight.zero_();net.brand.bias.copy_(torch.tensor(p).log())
            return dict(origin='project-trained-regional-model-head',architecture='finetuned-model-network-v1',state=net.state_dict(),labels=labels,brands=brands,temperature=1.,threshold=.5)
        old=checkpoint(['Cobalt','Matiz'],['Chevrolet','Daewoo'],[.8,.2]);new=checkpoint(['Cobalt','Matiz','Galaxy E5'],['Chevrolet','Daewoo','Geely'],[.1,.1,.8])
        reference=dict(origin='project-trained-visible-part-references',features=torch.eye(2,256),targets=torch.tensor([0,1]),ratios=torch.ones(2),labels=old['labels'],brands=old['brands'],temperature=.1,aspect_penalty=0.,neighbours=1,weight=0.,probability_temperature=1.,confidence_threshold=.5,source_model_sha256='a'*64)
        new.update(legacy_model=old,legacy_references=reference,legacy_model_sha256='a'*64,legacy_weight=.25)
        m=VehicleModelClassifier(new);p=m.probabilities(Image.new('RGB',(100,80),'white'),None,None)
        self.assertTrue(torch.allclose(p,torch.tensor([.275,.125,.6]),atol=1e-5));self.assertAlmostEqual(float(p.sum()),1.,places=5)

    def test_uncertain_candidate_is_marked_in_all_text_names(self):
        self.assertEqual(vehicle_name(dict(brand='Chevrolet',model='unknown',model_candidate='Cobalt')),'Chevrolet Cobalt?')
