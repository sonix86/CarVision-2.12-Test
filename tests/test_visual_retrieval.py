import unittest
import torch
from carvision.visual_retrieval import retrieval_probabilities,validate_bank,routed_probabilities

class RetrievalTests(unittest.TestCase):
    def bank(self):
        return dict(keys=torch.tensor([[[1.,0.],[1.,0.],[.8,.6]],[[1.,0.],[1.,0.],[.8,.6]]]),labels=torch.tensor([0,0,1]),groups=torch.tensor([0,0,1]),k=2,beta=1.,alpha=.5,mode='abstention-only',fallback_thresholds={'A':1.,'B':.7})
    def test_duplicate_image_views_and_source_groups_do_not_duplicate_votes(self):
        b=self.bank();p=retrieval_probabilities(torch.tensor([1.,0.]),b['keys'],b['labels'],b['groups'],2,2,1.)
        self.assertAlmostEqual(float(p.sum()),1.,6);self.assertAlmostEqual(float(p[0]),float(torch.tensor([1.,.8]).softmax(0)[0]),6)
    def test_primary_accepted_is_exactly_preserved(self):
        p=torch.tensor([.9,.1]);out,gate,route=routed_probabilities(torch.tensor([1.,0.]),p,self.bank(),['A','B'],{'A':.8,'B':.8},.06,True)
        self.assertIs(out,p);self.assertEqual(route,'primary');self.assertEqual(gate,.8)
    def test_abstention_uses_separate_gate(self):
        out,gate,route=routed_probabilities(torch.tensor([1.,0.]),torch.tensor([.55,.45]),self.bank(),['A','B'],{'A':.8,'B':.8},.06,True)
        self.assertEqual(route,'reference-fallback');self.assertEqual(gate,1.);self.assertAlmostEqual(float(out.sum()),1.,6)
    def test_malformed_bank_rejected(self):
        b=self.bank();validate_bank(b,2,2);b['labels']=torch.tensor([0,0,2])
        with self.assertRaises(ValueError):validate_bank(b,2,2)

if __name__=='__main__':unittest.main()
