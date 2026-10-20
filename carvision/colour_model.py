"""Independently trained eight-colour head over general ImageNet + original colour statistics."""
import numpy as np
import torch
from torch import nn
from torchvision import models,transforms
from .core import COLOURS

PREPROCESS=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def colour_statistics(image):
    image=image.convert('RGB').resize((128,128));features=[]
    for box in [(.05,.15,.95,.9),(.15,.35,.85,.85),(.1,.55,.9,.85)]:
        crop=image.crop(tuple(round(v*128) for v in box));rgb=np.asarray(crop).astype(np.float32)/255;hsv=np.asarray(crop.convert('HSV')).astype(np.float32)/255
        values=np.concatenate([rgb,hsv],axis=2).reshape(-1,6)
        features.extend(values.mean(0));features.extend(values.std(0))
        for q in [.1,.25,.5,.75,.9]:features.extend(np.quantile(values,q,axis=0))
        for channel,bins in [(0,18),(1,8),(2,10)]:
            counts=np.histogram(hsv[:,:,channel],bins=bins,range=(0,1))[0];features.extend(counts/max(1,counts.sum()))
    a=np.asarray(image).astype(np.float32)/255
    for row in range(3):
        for col in range(3):features.extend(np.median(a[row*42:(row+1)*42,col*42:(col+1)*42].reshape(-1,3),axis=0))
    return torch.tensor(np.asarray(features,dtype=np.float32))

def colour_head(dimension):return nn.Sequential(nn.Linear(dimension,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,8))

def paint_statistics(image):
    """Joint colour/brightness evidence on several possible body panels.

    Independent hue/saturation averages lose the distinction between blue glass,
    dark paint and pale neutral paint. Joint histograms retain that distinction;
    their interpretation is learned from training labels rather than hard-coded.
    """
    import cv2
    image=image.convert('RGB').resize((128,128));rgb=np.asarray(image);hsv=np.asarray(image.convert('HSV')).astype(np.float32)/255;lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)/255;values=[]
    for box in [(.08,.32,.92,.88),(.15,.5,.85,.85),(.03,.3,.3,.85),(.7,.3,.97,.85)]:
        x,y,r,b=[round(v*128) for v in box];a=hsv[y:b,x:r].reshape(-1,3);l=lab[y:b,x:r].reshape(-1,3)
        hist=np.histogram2d(a[:,1],a[:,2],bins=(8,10),range=((0,1),(0,1)))[0];values.extend((hist/hist.sum()).ravel())
        chrom=(a[:,1]>.25)&(a[:,2]>.22);neutral=a[:,1]<.2
        hist=np.histogram(a[chrom,0],bins=18,range=(0,1))[0];values.extend(hist/max(1,len(a)))
        hist=np.histogram(a[neutral,2],bins=12,range=(0,1))[0];values.extend(hist/max(1,len(a)))
        for q in [.1,.25,.5,.75,.9]:values.extend(np.quantile(l,q,axis=0))
        values.extend([neutral.mean(),(a[:,2]<.2).mean()])
    for row in range(4):
        for col in range(4):
            x,r=round((.05+col*.225)*128),round((.05+(col+1)*.225)*128);y,b=round((.28+row*.155)*128),round((.28+(row+1)*.155)*128)
            values.extend(np.median(rgb[y:b,x:r].reshape(-1,3),axis=0)/255);values.extend(np.median(hsv[y:b,x:r].reshape(-1,3),axis=0));values.extend(np.median(lab[y:b,x:r].reshape(-1,3),axis=0));values.append(float((hsv[y:b,x:r,1]<.2).mean()))
    return torch.cat([colour_statistics(image),torch.tensor(values,dtype=torch.float32)])

class ColourCNN(nn.Module):
    """Own colour decision, with trainable general-purpose image features."""
    def __init__(self):
        super().__init__();self.features=models.mobilenet_v3_small(weights=None).features;self.head=colour_head(1152)
    def forward(self,x):
        f=self.features(x);return self.head(torch.cat([f.mean((2,3)),f.amax((2,3))],1))

class ColourClassifier:
    def __init__(self,checkpoint):
        if checkpoint.get('origin') not in ('team-trained-colour-head-imagenet-backbone','team-trained-colour-statistics','team-trained-colour-ensemble'):raise ValueError('Expected the project-trained colour decision head')
        if checkpoint['classes']!=COLOURS:raise ValueError('Colour checkpoint uses a different eight-class taxonomy')
        self.network=None
        self.components=None
        if checkpoint.get('architecture')=='colour_evidence_ensemble_v3':
            self.components=[ColourClassifier(c) for c in checkpoint['components']];self.mixture=checkpoint['mixture'];self.confidence_power=float(checkpoint.get('confidence_power',0.));return
        if checkpoint.get('architecture')=='colour_cnn_avgmax_v2':
            self.network=ColourCNN().eval();self.network.load_state_dict(checkpoint['state']);return
        self.feature_mode=checkpoint.get('feature_mode','combined')
        self.features=None
        if self.feature_mode!='paint_statistics':
            self.features=models.mobilenet_v3_small(weights=None).features.eval();self.features.load_state_dict(checkpoint['backbone'])
        self.mean=checkpoint['mean'];self.std=checkpoint['std'];self.head=colour_head(len(self.mean)).eval();self.head.load_state_dict(checkpoint['head'])
    @torch.inference_mode()
    def probabilities(self,image):
        if self.components is not None:
            prob=torch.zeros(8);cache={}
            for entry in self.mixture:
                key=(entry['component'],tuple(entry['region']) if entry['region'] else None)
                if key not in cache:
                    view=image.crop(tuple(round(v*(image.width if i%2==0 else image.height)) for i,v in enumerate(entry['region']))) if entry['region'] else image
                    cache[key]=self.components[entry['component']].probabilities(view)
                prob+=cache[key]*entry['weight']*cache[key].max().pow(getattr(self,'confidence_power',0.))
            return prob/prob.sum().clamp_min(1e-8)
        if self.network is not None:
            return self.network(PREPROCESS(image.convert('RGB')).unsqueeze(0)).softmax(1)[0]
        if self.feature_mode=='paint_statistics':values=paint_statistics(image)
        elif self.feature_mode=='statistics':values=colour_statistics(image)
        else:
            f=self.features(PREPROCESS(image.convert('RGB')).unsqueeze(0));pooled=torch.cat([f.mean((2,3)),f.amax((2,3))],1)[0]
            values=torch.cat([pooled,colour_statistics(image)])
        return self.head(((values-self.mean)/self.std).unsqueeze(0)).softmax(1)[0]
    def predict(self,image):
        prob=self.probabilities(image);index=int(prob.argmax())
        return COLOURS[index],float(prob[index])
