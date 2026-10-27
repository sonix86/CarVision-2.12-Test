import unittest
from types import SimpleNamespace
from PIL import Image
import numpy as np
from carvision.scene import classify

class MakeFirstSpeedTests(unittest.TestCase):
    def test_abstention_skips_unused_legacy_networks_but_keeps_make_and_colour(self):
        calls=[]
        def obsolete(*a):raise AssertionError('Discarded legacy inference must not run')
        evidence=dict(brand='unknown',candidate='BYD',accepted=False,confidence=.4,source='own head',alternatives=[dict(brand='BYD',confidence=.4)])
        def make(*a):calls.append(1);return evidence
        p=SimpleNamespace(classifier=obsolete,vehicle_classifier=SimpleNamespace(probabilities=obsolete),make_expert=SimpleNamespace(predict=make),brand_threshold=.5,colour_classifier=None,colour_reliability_threshold=.5)
        r=classify(p,Image.new('RGB',(120,80),'white'),np.ones((80,120),bool))
        self.assertEqual(calls,[1]);self.assertEqual(r['brand'],'unknown');self.assertEqual(r['colour'],'white');self.assertEqual(r['model'],'unknown');self.assertEqual(r['make_evidence']['candidate'],'BYD')
    def test_known_make_without_trained_model_families_does_not_run_wrong_decoder(self):
        def obsolete(*a):raise AssertionError('Unsupported exact-model decoder must not run')
        evidence=dict(brand='GAC Aion',candidate='GAC Aion',accepted=True,confidence=.97,source='own head',alternatives=[dict(brand='GAC Aion',confidence=.97)])
        p=SimpleNamespace(classifier=obsolete,vehicle_classifier=SimpleNamespace(brands=['Chevrolet'],probabilities=obsolete),make_expert=SimpleNamespace(predict=lambda *a:evidence),brand_threshold=.5,colour_classifier=None,colour_reliability_threshold=.5)
        r=classify(p,Image.new('RGB',(120,80),'white'),np.ones((80,120),bool))
        self.assertEqual(r['brand'],'GAC Aion');self.assertEqual(r['model'],'unknown');self.assertIn('not trained',r['model_reason'])

if __name__=='__main__':unittest.main()
