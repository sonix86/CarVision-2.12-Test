import unittest
import torch
from carvision.semantic_make import MultiPrototypeMakeHead,load_make_head

class MultiPrototypeTests(unittest.TestCase):
    def test_opposite_views_of_same_brand_can_share_label(self):
        p=torch.tensor([[[1.,0.],[-1.,0.]],[[0.,1.],[0.,-1.]]]);head=MultiPrototypeMakeHead(p)
        self.assertEqual(head(torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])).argmax(1).tolist(),[0,0,1])
    def test_gradients_reach_learned_visual_modes(self):
        head=MultiPrototypeMakeHead(torch.randn(3,4,512));loss=torch.nn.functional.cross_entropy(head(torch.nn.functional.normalize(torch.randn(5,512),dim=1)),torch.tensor([0,1,2,0,1]));loss.backward();self.assertTrue(torch.isfinite(head.weight.grad).all());self.assertGreater(float(head.weight.grad.abs().sum()),0.)
    def test_checkpoint_factory_roundtrip(self):
        head=MultiPrototypeMakeHead(torch.randn(3,4,512));ck=dict(brands=['A','B','C'],two_views=False,head_kind='multi-prototype',prototype_count=4,head=head.state_dict());loaded=load_make_head(ck);x=torch.randn(2,512);self.assertTrue(torch.equal(head(x),loaded(x)))

if __name__=='__main__':unittest.main()
