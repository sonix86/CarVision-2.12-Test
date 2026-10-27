"""Own make classifier over a general CLIP image backbone; fully local inference.

Backbone: standard ViT-B/32, OpenCLIP/LAION (MIT). No automotive head is
downloaded. Runtime carries only the image encoder and our supervised head.
"""
from collections import OrderedDict
import math
import numpy as np
import torch
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
from torchvision.transforms.functional import to_tensor,normalize
from .segmentation import isolate

MEAN=(.48145466,.4578275,.40821073);STD=(.26862954,.26130258,.27577711)

class VisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.ln_1=torch.nn.LayerNorm(768);self.attn=torch.nn.MultiheadAttention(768,12,batch_first=True)
        self.ln_2=torch.nn.LayerNorm(768);self.mlp=torch.nn.Sequential(OrderedDict(c_fc=torch.nn.Linear(768,3072),gelu=torch.nn.GELU(),c_proj=torch.nn.Linear(3072,768)))
    def forward(self,x):
        q=self.ln_1(x);x=x+self.attn(q,q,q,need_weights=False)[0]
        return x+self.mlp(self.ln_2(x))

class GeneralVisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv1=torch.nn.Conv2d(3,768,32,stride=32,bias=False)
        self.class_embedding=torch.nn.Parameter(torch.zeros(768));self.positional_embedding=torch.nn.Parameter(torch.zeros(50,768));self.ln_pre=torch.nn.LayerNorm(768)
        self.transformer=torch.nn.Module();self.transformer.resblocks=torch.nn.Sequential(*[VisionBlock() for _ in range(12)])
        self.ln_post=torch.nn.LayerNorm(768);self.proj=torch.nn.Parameter(torch.zeros(768,512))
    def forward(self,x):
        x=self.conv1(x).flatten(2).transpose(1,2)
        x=torch.cat([self.class_embedding.expand(x.shape[0],1,-1),x],dim=1)
        x=self.transformer.resblocks(self.ln_pre(x+self.positional_embedding))
        return self.ln_post(x[:,0])@self.proj

def image_tensor(image):
    canvas=Image.new('RGB',(224,224),tuple(round(v*255) for v in MEAN));im=ImageOps.contain(image.convert('RGB'),(224,224),Image.Resampling.BICUBIC)
    canvas.paste(im,((224-im.width)//2,(224-im.height)//2));return normalize(to_tensor(canvas),MEAN,STD)

def visual_inputs(image,mask,kind='whole'):
    image=image.convert('RGB').copy();mask=np.asarray(mask)
    if kind=='phone':
        image.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST));image=ImageEnhance.Brightness(image).enhance(.55).filter(ImageFilter.GaussianBlur(.35))
    full=isolate(image,mask);w,h=full.size
    detail=full.crop((round(w*.1),round(h*.2),max(1,round(w*.9)),max(1,round(h*.9))))
    return torch.stack([image_tensor(full),image_tensor(detail)])

def embedding_features(z,two_views):
    z=torch.nn.functional.normalize(z,dim=-1)
    return torch.nn.functional.normalize(torch.cat([z[...,0,:]*.7,z[...,1,:]*.3],dim=-1),dim=-1) if two_views else z[...,0,:]

class CosineMakeHead(torch.nn.Module):
    def __init__(self,prototype):
        super().__init__();self.weight=torch.nn.Parameter(prototype.clone());self.bias=torch.nn.Parameter(torch.zeros(len(prototype)));self.log_scale=torch.nn.Parameter(torch.tensor(math.log(30.)))
    def forward(self,x):return self.log_scale.exp().clamp(max=100)*(x@torch.nn.functional.normalize(self.weight,dim=1).T)+self.bias

class MultiPrototypeMakeHead(torch.nn.Module):
    """Several learned visual modes per make (e.g. front/rear/body families)."""
    def __init__(self,prototype):
        super().__init__();self.weight=torch.nn.Parameter(prototype.clone());self.bias=torch.nn.Parameter(torch.zeros(len(prototype)));self.log_scale=torch.nn.Parameter(torch.tensor(math.log(30.)))
    def forward(self,x):
        scores=self.log_scale.exp().clamp(max=100)*torch.einsum('...d,ckd->...ck',x,torch.nn.functional.normalize(self.weight,dim=-1))
        return torch.logsumexp(scores,dim=-1)-math.log(self.weight.shape[1])+self.bias

def load_make_head(checkpoint):
    dimension=1024 if checkpoint['two_views'] else 512;classes=len(checkpoint['brands']);kind=checkpoint.get('head_kind','cosine')
    if kind=='cosine':head=CosineMakeHead(torch.zeros(classes,dimension))
    elif kind=='multi-prototype':
        k=checkpoint['prototype_count']
        if not isinstance(k,int) or not 1<=k<=8:raise ValueError('Invalid learned prototype count')
        head=MultiPrototypeMakeHead(torch.zeros(classes,k,dimension))
    else:raise ValueError('Unsupported own make head')
    head.load_state_dict(checkpoint['head']);return head.eval()

class SemanticMakeExpert:
    def __init__(self,checkpoint):
        if checkpoint.get('origin') not in ('project-trained-semantic-make-v25','project-trained-semantic-make-v26'):raise ValueError('Expected our supervised make head')
        self.brands=checkpoint['brands'];self.two_views=checkpoint['two_views'];self.thresholds=checkpoint['thresholds'];self.temperature=float(checkpoint['temperature']);self.margin=float(checkpoint['margin'])
        if not self.brands or len(set(self.brands))!=len(self.brands) or set(self.thresholds)!=set(self.brands):raise ValueError('Invalid semantic make taxonomy')
        if not math.isfinite(self.temperature) or self.temperature<=0 or not 0<=self.margin<=1 or not all(math.isfinite(float(v)) and 0<=v<=1 for v in self.thresholds.values()):raise ValueError('Invalid semantic make calibration')
        self.encoder=GeneralVisionEncoder().eval();self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.head=load_make_head(checkpoint)
        self.retrieval=checkpoint.get('retrieval')
        if self.retrieval is not None:
            from .visual_retrieval import validate_bank
            validate_bank(self.retrieval,len(self.brands),1024 if self.two_views else 512)
            gates=self.retrieval['fallback_thresholds']
            if set(gates)!=set(self.brands) or not all(math.isfinite(float(v)) and 0<=v<=1 for v in gates.values()):raise ValueError('Invalid reference confirmation thresholds')
    @torch.inference_mode()
    def predict(self,image,mask,minimum=.5):
        if not math.isfinite(float(minimum)) or not 0<=minimum<=1:raise ValueError('Make threshold must be between zero and one')
        inputs=visual_inputs(image,mask)
        z=self.encoder(inputs if self.two_views else inputs[:1])
        if not self.two_views:z=torch.cat([z,z],0)
        features=embedding_features(z,self.two_views);p=(self.head(features)/self.temperature).softmax(0);route='primary'
        if self.retrieval is not None:
            from .visual_retrieval import routed_probabilities
            p,threshold,route=routed_probabilities(features,p,self.retrieval,self.brands,self.thresholds,self.margin,min(image.size)>=40,float(minimum))
        else:threshold=max(float(self.thresholds[self.brands[int(p.argmax())]]),float(minimum))
        values,ids=p.topk(2);candidate=self.brands[int(ids[0])];confidence=float(values[0]);margin=float(values[0]-values[1])
        accepted=bool(torch.isfinite(p).all()) and threshold<1 and confidence>=threshold and margin>=self.margin and min(image.size)>=40
        ranking=[dict(brand=self.brands[int(j)],confidence=float(p[j])) for j in p.argsort(descending=True)]
        return dict(brand=candidate if accepted else 'unknown',candidate=candidate,display_name=candidate,confidence=confidence,accepted=accepted,threshold=threshold,margin=margin,alternatives=ranking[:3],ranking=ranking,source='own supervised make head; general CLIP image encoder'+('; grouped training-reference fallback' if route=='reference-fallback' else ''),acceptance_reason='Марка прошла проверочный порог; ошибки возможны.' if accepted else 'Марка не подтверждена: показаны варианты по изображению.',text_evidence=[])
