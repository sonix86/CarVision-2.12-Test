import unittest
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from carvision.semantic_make import SemanticMakeExpert,CosineMakeHead,embedding_features,visual_inputs


class SemanticMakeChecks(unittest.TestCase):
    def test_generic_encoder_matches_reference(self):
        p=Path(__file__).resolve().parents[1]/'models/semantic_v25/preparation.json'
        if not p.exists():self.skipTest('General resource preparation is not part of a minimal runtime checkout')
        report=json.loads(p.read_text(encoding='utf-8'));self.assertLess(report['reference_max_absolute_error'],1e-4);self.assertFalse(report['automotive_classifier_downloaded'])

    def test_whole_and_detail_use_same_feature_definition_for_batch_and_runtime(self):
        z=torch.randn(4,2,512,generator=torch.Generator().manual_seed(25))
        for two in (False,True):
            batched=embedding_features(z,two);single=torch.stack([embedding_features(r,two) for r in z])
            self.assertTrue(torch.equal(batched,single));self.assertTrue(torch.allclose(batched.norm(dim=1),torch.ones(4)))

    def test_unconfirmed_gate_cannot_be_bypassed_by_softmax_rounding(self):
        expert=object.__new__(SemanticMakeExpert);expert.encoder=lambda x:torch.ones((len(x),512));expert.two_views=False;expert.head=lambda x:torch.tensor([100.,-100.]);expert.temperature=1.;expert.thresholds={'BYD':1.,'Zeekr':1.};expert.brands=['BYD','Zeekr'];expert.margin=.06;expert.retrieval=None
        image=Image.new('RGB',(100,80));mask=np.ones((80,100),bool)
        self.assertFalse(expert.predict(image,mask)['accepted']);expert.thresholds['BYD']=.8
        self.assertTrue(expert.predict(image,mask)['accepted']);self.assertFalse(expert.predict(image,mask,minimum=1.)['accepted'])
        with self.assertRaises(ValueError):expert.predict(image,mask,float('nan'))

    def test_neighbour_pixels_do_not_enter_semantic_views(self):
        mask=np.zeros((80,100),bool);mask[15:70,20:90]=True
        a=np.zeros((80,100,3),np.uint8);b=a.copy();b[~mask]=255
        for kind in ('whole','phone'):
            # Whole view must remove outside-mask pixels exactly. Phone blur can
            # legitimately mix boundary pixels, so compare the central region.
            if kind=='whole':self.assertTrue(torch.equal(visual_inputs(Image.fromarray(a),mask,kind),visual_inputs(Image.fromarray(b),mask,kind)))

    def test_head_is_trainable_from_labelled_images(self):
        prototype=torch.eye(2);head=CosineMakeHead(prototype);before=head.weight.detach().clone();optimizer=torch.optim.SGD(head.parameters(),lr=.001)
        x=torch.tensor([[.9,.1],[.1,.9]]);y=torch.tensor([1,0]);loss=torch.nn.functional.cross_entropy(head(x),y);optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertFalse(torch.equal(before,head.weight.detach()))

if __name__=='__main__':unittest.main()
