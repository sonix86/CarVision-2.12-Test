"""Synthetic control-flow check; these are not car photos or accuracy evidence."""
import tempfile,unittest
from pathlib import Path
import torch
from carvision.make_expert import make_head
from tools.make_scope_features_v23 import key,TRAIN_VIEWS,EVAL_VIEWS
from tools.finetune_make_scope_v23 import refine


class RefinementChecks(unittest.TestCase):
    def test_rejected_second_round_restores_original_encoder_and_head(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layer4=torch.nn.Linear(3,512)
            def forward(self,x):return self.layer4(x.mean((2,3)))
        torch.manual_seed(9);encoder=TinyEncoder();head=make_head(2)
        saved_encoder={k:v.clone() for k,v in encoder.state_dict().items()};saved_head={k:v.clone() for k,v in head.state_dict().items()}
        rows=[dict(body_rgb=torch.full((24,32,3),30+i*40,dtype=torch.uint8),body_mask=torch.ones(24,32,dtype=torch.bool),y=i%2,historical=True,split='train' if i<2 else 'validation') for i in range(4)]
        cache=dict(items={key(r,k):dict(feature=torch.nn.functional.normalize(torch.ones(512),dim=0)) for r in rows for k in TRAIN_VIEWS})
        val={k:(torch.zeros(2,512),torch.tensor([0,1]),rows[2:]) for k in EVAL_VIEWS}
        def measure(p,y,idx=None):return dict(top1=float((p.argmax(1)==y).float().mean()),macro=float((p.argmax(1)==y).float().mean()))
        baseline={k:dict(top1=0.) for k in EVAL_VIEWS}
        with tempfile.TemporaryDirectory() as temp:
            encoder,head,_,report=refine(encoder,head,rows,cache,val,baseline,measure,torch.ones(2),2.,Path(temp))
        self.assertFalse(report['selected']);self.assertEqual(len(report['history']),6)
        self.assertTrue(all(torch.equal(v,saved_encoder[k]) for k,v in encoder.state_dict().items()))
        self.assertTrue(all(torch.equal(v,saved_head[k]) for k,v in head.state_dict().items()))
