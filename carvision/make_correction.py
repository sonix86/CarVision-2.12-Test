"""Scoped, validation-gated correction by our supervised phone-domain head."""
import math
import torch
from .semantic_make import CosineMakeHead

class MakeCorrection:
    def __init__(self,checkpoint,brands,dimension):
        if checkpoint.get('origin')!='project-trained-phone-make-correction-v32':raise ValueError('Expected our supervised correction head')
        if checkpoint['brands']!=brands:raise ValueError('Correction taxonomy mismatch')
        self.brands=brands;self.scope=set(checkpoint['scope']);self.thresholds=checkpoint['thresholds'];self.temperature=float(checkpoint['temperature']);self.margin=float(checkpoint['margin']);self.minimum_size=int(checkpoint['minimum_size'])
        if not self.scope.issubset(brands) or set(self.thresholds)!=set(brands):raise ValueError('Invalid correction scope')
        if not math.isfinite(self.temperature) or self.temperature<=0 or not 0<=self.margin<=1 or self.minimum_size<64:raise ValueError('Invalid correction calibration')
        if any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in self.thresholds.values()):raise ValueError('Invalid correction gate')
        if any(self.thresholds[b]!=1. for b in brands if b not in self.scope):raise ValueError('Unsupported correction class enabled')
        self.head=CosineMakeHead(torch.zeros(len(brands),dimension)).eval();self.head.load_state_dict(checkpoint['head'])

    @torch.inference_mode()
    def predict(self,z,size,mask,minimum):
        if min(size)<self.minimum_size or size[0]/size[1]>3.2 or int(mask.sum())<1200 or float(mask.mean())<.35:return None
        p=(self.head(z)/self.temperature).softmax(0);v,ids=p.topk(2);brand=self.brands[int(ids[0])];gate=max(float(minimum),float(self.thresholds[brand]));margin=float(v[0]-v[1])
        if brand not in self.scope or not bool(torch.isfinite(p).all()) or gate>=1 or float(v[0])<gate or margin<self.margin:return None
        ranking=[dict(brand=self.brands[int(i)],confidence=float(p[i])) for i in p.argsort(descending=True)]
        return dict(brand=brand,candidate=brand,display_name=brand,accepted=True,confidence=float(v[0]),threshold=gate,margin=margin,ranking=ranking,alternatives=ranking[:3],source='own supervised phone-domain correction; validation-gated scoped makes',acceptance_reason='Марка подтверждена дополнительной собственной моделью; ошибки возможны.',text_evidence=[])
