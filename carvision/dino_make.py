"""Our trained make decision over general DINO and optional CLIP features."""
import math
import torch
from .dino_features import DinoEncoder,dino_input,descriptor
from .semantic_make import CosineMakeHead,GeneralVisionEncoder,visual_inputs
from .visual_retrieval import validate_bank,routed_probabilities


class DinoMakeExpert:
    def __init__(self,checkpoint,clip_checkpoint=None):
        if checkpoint.get('origin')!='project-trained-dino-make-v28':raise ValueError('Expected our supervised DINO make head')
        self.kind=checkpoint['kind']
        if self.kind not in ('dino','joint'):raise ValueError('Unsupported supervised feature combination')
        self.brands=checkpoint['brands'];self.thresholds=checkpoint['thresholds'];self.temperature=float(checkpoint['temperature']);self.margin=float(checkpoint['margin'])
        if not self.brands or len(set(self.brands))!=len(self.brands) or set(self.thresholds)!=set(self.brands):raise ValueError('Invalid make taxonomy')
        if not math.isfinite(self.temperature) or self.temperature<=0 or not 0<=self.margin<=1 or not all(math.isfinite(float(v)) and 0<=v<=1 for v in self.thresholds.values()):raise ValueError('Invalid make calibration')
        self.encoder=DinoEncoder().eval();self.encoder.load_state_dict(checkpoint['encoder_state']);self.clip=None
        if self.kind=='joint':
            if clip_checkpoint is None:raise ValueError('Joint head requires its paired generic CLIP image encoder')
            self.clip=GeneralVisionEncoder().eval();self.clip.load_state_dict(clip_checkpoint['encoder_state'])
        dimension=768 if self.kind=='dino' else 1280
        self.head=CosineMakeHead(torch.zeros(len(self.brands),dimension)).eval();self.head.load_state_dict(checkpoint['head'])
        self.family=None
        if checkpoint.get('model_head'):
            from .family_head import FamilyHead
            self.family=FamilyHead(checkpoint['model_head'],dimension)
        self.correction=None
        if checkpoint.get('make_correction_head'):
            from .make_correction import MakeCorrection
            self.correction=MakeCorrection(checkpoint['make_correction_head'],self.brands,dimension)
        self.retrieval=checkpoint['retrieval'];validate_bank(self.retrieval,len(self.brands),dimension)
        gates=self.retrieval['fallback_thresholds']
        if set(gates)!=set(self.brands) or not all(math.isfinite(float(v)) and 0<=v<=1 for v in gates.values()):raise ValueError('Invalid fallback calibration')

    @torch.inference_mode()
    def feature(self,image,mask):
        z=descriptor(self.encoder(dino_input(image,mask).unsqueeze(0)))[0]
        if self.clip is not None:
            clip=torch.nn.functional.normalize(self.clip(visual_inputs(image,mask)[:1])[0],dim=-1)
            z=torch.nn.functional.normalize(torch.cat([z,clip]),dim=-1)
        return z

    @torch.inference_mode()
    def predict(self,image,mask,minimum=.5):
        if not math.isfinite(float(minimum)) or not 0<=minimum<=1:raise ValueError('Make threshold must be between zero and one')
        z=self.feature(image,mask);p=(self.head(z)/self.temperature).softmax(0)
        p,gate,route=routed_probabilities(z,p,self.retrieval,self.brands,self.thresholds,self.margin,min(image.size)>=40,float(minimum))
        v,ids=p.topk(2);candidate=self.brands[int(ids[0])];confidence=float(v[0]);margin=float(v[0]-v[1]);accepted=bool(torch.isfinite(p).all()) and gate<1 and confidence>=gate and margin>=self.margin and min(image.size)>=40
        ranking=[dict(brand=self.brands[int(j)],confidence=float(p[j])) for j in p.argsort(descending=True)]
        result=dict(brand=candidate if accepted else 'unknown',candidate=candidate,display_name=candidate,confidence=confidence,accepted=accepted,threshold=gate,margin=margin,ranking=ranking,alternatives=ranking[:3],source='own supervised make head over general '+('DINO and CLIP' if self.clip is not None else 'DINO')+' image features'+('; grouped training-reference fallback' if route=='reference-fallback' else ''),acceptance_reason='Марка прошла проверочный порог; ошибки возможны.' if accepted else 'Марка не подтверждена: показаны варианты по изображению.',text_evidence=[])
        family=getattr(self,'family',None)
        if family is not None and not result['accepted']:
            fallback=family.predict_make(z,self.brands,image.size,mask,minimum)
            if fallback is not None:
                fallback['primary_make_candidate']=result['candidate'];fallback['primary_make_confidence']=result['confidence'];result=fallback
        correction=getattr(self,'correction',None)
        if correction is not None:
            corrected=correction.predict(z,image.size,mask,minimum)
            if corrected is not None:
                corrected['prior_make_brand']=result['brand'];corrected['prior_make_candidate']=result['candidate'];result=corrected
        if family is not None:result['model_evidence']=family.predict(z,result['brand'],image.size)
        return result
