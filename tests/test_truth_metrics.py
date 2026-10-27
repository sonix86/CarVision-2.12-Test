import unittest
import torch
from tools.train_truth_v30 import evidence_correct

class TruthMetricTests(unittest.TestCase):
    def test_abstention_cannot_match_class_zero(self):
        self.assertFalse(bool(evidence_correct(torch.zeros(1,3),torch.tensor([0]))[0]))
    def test_invalid_evidence_is_not_a_correct_answer(self):
        self.assertFalse(bool(evidence_correct(torch.tensor([[float('nan'),.2]]),torch.tensor([0]))[0]))
    def test_evidence_must_match_independent_truth(self):
        p=torch.tensor([[.1,.8,.1],[.8,.1,.1]])
        self.assertEqual(evidence_correct(p,torch.tensor([1,1])).tolist(),[True,False])
