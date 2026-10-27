"""Own supervised make head over generic ImageNet and general OCR features."""
import math
import numpy as np,torch
from PIL import Image,ImageEnhance,ImageFilter
from torchvision import models
from .body_expert import body_tensor
from .segmentation import isolate

def make_inputs(image,mask,kind='whole'):
    im=image.copy();mask=np.asarray(mask)
    if kind=='phone':
        im.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));im=ImageEnhance.Brightness(im).enhance(.55).filter(ImageFilter.GaussianBlur(.35))
    foreground=isolate(im,mask);w,h=foreground.size
    detail=foreground.crop((round(w*.12),round(h*.25),max(round(w*.12)+1,round(w*.88)),max(round(h*.25)+1,round(h*.85))))
    return torch.stack([body_tensor(foreground),body_tensor(detail)])

def make_encoder(state):
    encoder=models.convnext_tiny(weights=None);encoder.classifier[2]=torch.nn.Identity()
    encoder.load_state_dict(state);return encoder.eval()

class MultimodalMakeExpert:
    def __init__(self,checkpoint,directory):
        if checkpoint.get('origin')!='project-trained-multimodal-make-v24':raise ValueError('Expected own supervised make head')
        self.brands=checkpoint['brands'];self.encoder=make_encoder(checkpoint['encoder_state'])
        self.mean=checkpoint['mean'];self.std=checkpoint['std'];self.head=torch.nn.Linear(len(self.mean),len(self.brands)).eval();self.head.load_state_dict(checkpoint['head'])
        self.temperature=checkpoint['temperature'];self.thresholds=checkpoint['thresholds'];self.margin=checkpoint['margin'];self.text_reader=None
        if self.temperature<=0 or not 0<=self.margin<=1 or set(self.thresholds)!=set(self.brands):raise ValueError('Invalid make calibration')
        if not all(math.isfinite(float(t)) and 0<=t<=1 for t in self.thresholds.values()):raise ValueError('Invalid make thresholds')
        if checkpoint['with_text']:
            from .vehicle_text import VehicleTextReader
            self.text_reader=VehicleTextReader(directory/'ocr')

    @torch.inference_mode()
    def predict(self,image,mask,minimum=.5):
        if not math.isfinite(float(minimum)) or not 0<=minimum<=1:raise ValueError('Make threshold must be between zero and one')
        z=self.encoder(make_inputs(image,mask));feature=torch.nn.functional.normalize(z,dim=1).reshape(-1);records=[]
        if self.text_reader is not None:
            from .vehicle_text import text_features
            records=self.text_reader.read(image,mask);feature=torch.cat([feature,torch.from_numpy(text_features(records))])
        p=(self.head((feature-self.mean)/self.std)/self.temperature).softmax(0);values,indices=p.topk(2);candidate=self.brands[int(indices[0])]
        threshold=max(float(self.thresholds[candidate]),float(minimum));confidence=float(values[0]);margin=float(values[0]-values[1])
        accepted=bool(torch.isfinite(p).all()) and threshold<1 and confidence>=threshold and margin>=self.margin and min(image.size)>=40
        ranking=[dict(brand=self.brands[int(j)],confidence=float(p[j])) for j in p.argsort(descending=True)]
        return dict(brand=candidate if accepted else 'unknown',candidate=candidate,display_name=candidate,confidence=confidence,accepted=accepted,threshold=threshold,margin=margin,alternatives=ranking[:3],ranking=ranking,source='own supervised ImageNet-detail'+(' + general OCR feature head' if self.text_reader else ' feature head'),acceptance_reason='Марка подтверждена по проверенным признакам; ошибки возможны.' if accepted else 'Недостаточно согласованных признаков или независимых примеров для подтверждения.',text_evidence=records)
