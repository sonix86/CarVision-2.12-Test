"""Our source-labelled family decision over shared general image features."""
import math
import torch
from .semantic_make import CosineMakeHead

class FamilyHead:
    def __init__(self,checkpoint,dimension):
        if checkpoint.get('origin')!='project-trained-family-head-v30':raise ValueError('Expected our independently source-labelled family head')
        self.pairs=[tuple(p) for p in checkpoint['pairs']];self.keys=[b+'|'+m for b,m in self.pairs];self.thresholds=checkpoint['thresholds'];self.temperature=float(checkpoint['temperature']);self.margin=float(checkpoint.get('margin',.06));self.clipped=bool(checkpoint.get('clipped_validation_passed',False))
        self.similarity_thresholds=checkpoint.get('similarity_thresholds',{k:-1. for k in self.keys});self.global_margin=bool(checkpoint.get('global_margin_required',False));self.direct=bool(checkpoint.get('direct_manufacturer_fallback',False))
        self.fallback_brands=checkpoint.get('manufacturer_fallback_brands')
        if not self.pairs or len(set(self.pairs))!=len(self.pairs) or set(self.keys)!=set(self.thresholds):raise ValueError('Invalid family taxonomy/calibration')
        if not math.isfinite(self.temperature) or self.temperature<=0 or not 0<=self.margin<=1 or not all(math.isfinite(float(v)) and 0<=v<=1 for v in self.thresholds.values()):raise ValueError('Invalid family thresholds')
        if set(self.similarity_thresholds)!=set(self.keys) or not all(math.isfinite(float(v)) and -1<=v<=1 for v in self.similarity_thresholds.values()):raise ValueError('Invalid family similarity gates')
        self.head=CosineMakeHead(torch.zeros(len(self.pairs),dimension)).eval();self.head.load_state_dict(checkpoint['head'])

    @torch.inference_mode()
    def predict(self,feature,brand,size):
        probabilities=(self.head(feature)/self.temperature).softmax(0)
        similarity=feature@torch.nn.functional.normalize(self.head.weight,dim=1).T
        return self.describe(probabilities,brand,size,similarity)

    def describe(self,probabilities,brand,size,similarities=None):
        ids=[i for i,(b,m) in enumerate(self.pairs) if b==brand]
        empty=dict(model='unknown',model_candidate='unknown',model_candidate_brand=brand,model_confidence=0.,model_alternatives=[],model_ranking=[],model_source='own shared-feature family head',model_clipping_validated=self.clipped)
        if not ids:
            return dict(empty,model_reason='Марка не подтверждена.' if brand=='unknown' else 'Для этой марки нет обученных семейств модели.')
        order=sorted(ids,key=lambda i:float(probabilities[i]),reverse=True);index=order[0];confidence=float(probabilities[index]);second=float(probabilities[order[1]]) if len(order)>1 else 0.
        if self.global_margin:second=max(float(probabilities[i]) for i in range(len(probabilities)) if i!=index)
        threshold=float(self.thresholds[self.keys[index]]);finite=bool(torch.isfinite(probabilities).all());similarity_ok=similarities is not None and float(similarities[index])>=self.similarity_thresholds[self.keys[index]] if self.global_margin else True;accepted=finite and similarity_ok and threshold<1 and confidence>=threshold and confidence-second>=self.margin and min(size)>=40
        ranking=[dict(brand=self.pairs[i][0],model=self.pairs[i][1],confidence=float(probabilities[i])) for i in order]
        return dict(empty,model=self.pairs[index][1] if accepted else 'unknown',model_candidate=self.pairs[index][1],model_confidence=confidence,model_threshold=threshold,model_margin=confidence-second,model_alternatives=ranking[:3],model_ranking=ranking,model_family_accepted=accepted)

    @torch.inference_mode()
    def predict_make(self,feature,brands,size,mask,minimum):
        if not self.direct or min(size)<64 or size[0]/size[1]>3.2 or int(mask.sum())<1200 or float(mask.mean())<.35:return None
        p=(self.head(feature)/self.temperature).softmax(0);v,ids=p.topk(2);index=int(ids[0]);gate=max(float(minimum),float(self.thresholds[self.keys[index]]));similarity=float(feature@torch.nn.functional.normalize(self.head.weight[index],dim=0))
        if self.fallback_brands is not None and self.pairs[index][0] not in self.fallback_brands:return None
        if not bool(torch.isfinite(p).all()) or gate>=1 or float(v[0])<gate or float(v[0]-v[1])<self.margin or similarity<self.similarity_thresholds[self.keys[index]]:return None
        scores=torch.zeros(len(brands));mapping=torch.tensor([brands.index(b) for b,m in self.pairs]);scores.scatter_add_(0,mapping,p);brand=self.pairs[index][0];ranking=[dict(brand=brands[int(j)],confidence=float(scores[j])) for j in scores.argsort(descending=True)]
        return dict(brand=brand,candidate=brand,display_name=brand,confidence=float(scores[brands.index(brand)]),accepted=True,threshold=gate,margin=float(v[0]-v[1]),ranking=ranking,alternatives=ranking[:3],source='own source-labelled family head; independently validated manufacturer fallback',family_score=float(v[0]),family_similarity=similarity,acceptance_reason='Производитель подтверждён собственной моделью семейства; ошибки возможны.',text_evidence=[])
