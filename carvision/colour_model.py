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

class ColourClassifier:
    def __init__(self,checkpoint):
        if checkpoint.get('origin')!='team-trained-colour-head-imagenet-backbone':raise ValueError('Expected the project-trained colour decision head')
        if checkpoint['classes']!=COLOURS:raise ValueError('Colour checkpoint uses a different eight-class taxonomy')
        self.features=models.mobilenet_v3_small(weights=None).features.eval();self.features.load_state_dict(checkpoint['backbone'])
        self.feature_mode=checkpoint.get('feature_mode','combined')
        self.mean=checkpoint['mean'];self.std=checkpoint['std'];self.head=colour_head(len(self.mean)).eval();self.head.load_state_dict(checkpoint['head'])
    @torch.inference_mode()
    def predict(self,image):
        if self.feature_mode=='statistics':values=colour_statistics(image)
        else:
            f=self.features(PREPROCESS(image.convert('RGB')).unsqueeze(0));pooled=torch.cat([f.mean((2,3)),f.amax((2,3))],1)[0]
            values=torch.cat([pooled,colour_statistics(image)])
        prob=self.head(((values-self.mean)/self.std).unsqueeze(0)).softmax(1)[0];index=int(prob.argmax())
        return COLOURS[index],float(prob[index])
