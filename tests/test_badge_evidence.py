import unittest
import torch
from PIL import Image
from carvision.badge_evidence import BadgeEvidence,fuse_badge

class BadgeEvidenceTests(unittest.TestCase):
    def test_unaccepted_cannot_change_answer(self):
        p=torch.tensor([.1,.9]);q=fuse_badge(p,[('Chevrolet','Cobalt'),('BMW','3')],dict(brand='Chevrolet',accepted=False))
        self.assertTrue(torch.equal(p,q))
    def test_soft_evidence_does_not_delete_other_brands(self):
        p=torch.tensor([.1,.9]);q=fuse_badge(p,[('Chevrolet','Cobalt'),('BMW','3')],dict(brand='Chevrolet',accepted=True))
        self.assertAlmostEqual(float(q.sum()),1.,places=6);self.assertGreater(q[0],p[0]);self.assertGreater(q[1],0)
    def test_unrepresented_emblem_does_not_force_supported_model(self):
        p=torch.tensor([.1,.9]);q=fuse_badge(p,[('Chevrolet','Cobalt'),('BMW','3')],dict(brand='Kia',accepted=True))
        self.assertTrue(torch.equal(p,q))
    def test_no_pixels_no_badge(self):
        detector=BadgeEvidence(dict(templates=[],enabled=False,threshold=1))
        self.assertIsNone(detector.predict(Image.new('RGB',(20,20))))

if __name__=='__main__':unittest.main()
