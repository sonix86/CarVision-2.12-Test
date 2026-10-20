import unittest
import numpy as np
import torch
from PIL import Image
from carvision.visual_parts import VisualReferences,quality_of_fragment
from carvision.colour_model import JointColourClassifier

class VisualPartChecks(unittest.TestCase):
    def checkpoint(self):
        return dict(origin='project-trained-visible-part-references',features=torch.tensor([[1.,0.],[0.,1.]]),targets=torch.tensor([0,1]),ratios=torch.tensor([2.,4.]),labels=['Cobalt','K5'],brands=['Chevrolet','Kia'],temperature=.1,aspect_penalty=.05,neighbours=3,weight=1.,probability_temperature=1.,confidence_threshold=.6)
    def test_descriptor_comparison_uses_appearance_not_filename(self):
        r=VisualReferences(self.checkpoint());p=r.probabilities(torch.tensor([1.,0.]),2.)
        self.assertGreater(float(p[0]),.99);self.assertAlmostEqual(float(r.calibrated(p).sum()),1.)
    def test_generic_or_downloaded_classifier_cannot_be_used_as_our_references(self):
        c=self.checkpoint();c['origin']='ready-made-model-head'
        with self.assertRaises(ValueError):VisualReferences(c)
    def test_roof_strip_and_tiny_car_need_more_evidence(self):
        self.assertTrue(quality_of_fragment(np.ones((65,400),bool)))
        self.assertTrue(quality_of_fragment(np.ones((30,50),bool)))
        self.assertFalse(quality_of_fragment(np.ones((150,250),bool)))
    def test_personal_training_colour_head_is_used(self):
        class Joint:
            def __call__(self,x):
                logits=torch.zeros(len(x),8);logits[:,2]=12
                return torch.zeros(len(x),2),logits
        r=JointColourClassifier(Joint(),lambda im:torch.zeros(3,32,32));self.assertEqual(int(r.probabilities(Image.new('RGB',(40,30))).argmax()),2)
    def test_pipeline_connects_the_new_personal_colour_head(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from carvision.core import Classifier,Pipeline
        model=Classifier(2).eval()
        with torch.no_grad():model.colour.weight.zero_();model.colour.bias.zero_();model.colour.bias[5]=12
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'detector.pt').touch();torch.save({},d/'segmenter.pt')
            torch.save(dict(origin='team-trained-from-scratch',state=model.state_dict(),brands=['Chevrolet','Kia'],joint_colour_head=True),d/'classifier.pt')
            with patch('carvision.segmentation.InstanceSegmenter',return_value=object()):p=Pipeline(d)
            self.assertIsInstance(p.colour_classifier,JointColourClassifier);self.assertIsNone(p.masked_colour)
            self.assertEqual(int(p.colour_classifier.probabilities(Image.new('RGB',(100,80),'white')).argmax()),5)
    def test_thin_roof_keeps_candidates_but_never_confirms_a_model(self):
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        from carvision.scene import predict_instances
        mask=np.zeros((400,600),bool);mask[30:65,100:500]=True
        p=SimpleNamespace(segmenter=SimpleNamespace(instances=lambda *a:([dict(mask=mask,score=.99)],1)),detection=.5,mode='fast',classifier=object(),brands=['Chevrolet'])
        predicted=dict(brand='Chevrolet',model='Cobalt',model_alternatives=[dict(brand='Chevrolet',model='Cobalt',confidence=.99)],colour='white')
        with patch('carvision.scene.classify',return_value=predicted):r=predict_instances(p,Image.new('RGB',(600,400)),'roof.jpg',None,False,time.perf_counter())['cars'][0]
        self.assertEqual(r['model'],'unknown');self.assertEqual(r['brand'],'unknown');self.assertTrue(r['model_limited']);self.assertEqual(r['model_alternatives'][0]['model'],'Cobalt')
