"""Own paint-colour head over visible foreground pixels; neighbours are excluded."""
import numpy as np
import torch
from torch import nn
from PIL import Image
from .core import COLOURS

def features(image,mask):
    import cv2
    rgb=np.asarray(image.convert('RGB').resize((128,128),Image.Resampling.BILINEAR));m=np.asarray(Image.fromarray(np.asarray(mask,dtype=np.uint8)*255).resize((128,128),Image.Resampling.NEAREST))>127
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).astype(np.float32)/np.array([180,255,255],dtype=np.float32);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)/255;out=[]
    for x,y,r,b in [(0,0,128,128),(6,58,122,117),(3,42,51,112),(77,42,125,112)]:
        valid=m[y:b,x:r];a=hsv[y:b,x:r][valid];l=lab[y:b,x:r][valid]
        out.append(float(valid.mean()))
        if len(a)<3:a=np.zeros((3,3),dtype=np.float32);l=np.zeros((3,3),dtype=np.float32)
        for v in [a,l]:
            out.extend(v.mean(0));out.extend(v.std(0))
            for q in [.05,.15,.3,.5,.7,.85,.95]:out.extend(np.quantile(v,q,axis=0))
        joint=np.histogram2d(a[:,1],a[:,2],bins=(6,8),range=((0,1),(0,1)))[0];out.extend((joint/max(1,joint.sum())).ravel())
        chromatic=a[(a[:,1]>.2)&(a[:,2]>.18)];hist=np.histogram(chromatic[:,0],bins=24,range=(0,1))[0];out.extend(hist/max(1,len(a)))
        neutral=a[a[:,1]<.2];hist=np.histogram(neutral[:,2],bins=16,range=(0,1))[0];out.extend(hist/max(1,len(a)))
    return torch.tensor(np.asarray(out,dtype=np.float32))

def head(dimension):return nn.Sequential(nn.Linear(dimension,96),nn.ReLU(),nn.Dropout(.15),nn.Linear(96,8))

class MaskedColourClassifier:
    def __init__(self,checkpoint):
        if checkpoint.get('origin')!='project-trained-masked-paint-head':raise ValueError('Expected independently trained visible-paint classifier')
        if checkpoint['classes']!=COLOURS:raise ValueError('Incompatible colour names')
        self.mean=checkpoint['mean'];self.std=checkpoint['std'];self.trees=checkpoint.get('trees');self.head=None
        if self.trees is None:self.head=head(len(self.mean)).eval();self.head.load_state_dict(checkpoint['state'])
        self.temperature=checkpoint.get('temperature',1.);self.weight=checkpoint.get('weight',1.);self.threshold=checkpoint.get('threshold',.65);self.gate=checkpoint.get('gate')
    @torch.inference_mode()
    def probabilities(self,image,mask):return self.feature_probabilities(features(image,mask).unsqueeze(0))[0]

    @torch.inference_mode()
    def feature_probabilities(self,values):
        x=(values-self.mean)/self.std
        if self.trees is None:return (self.head(x)/self.temperature).softmax(1)
        result=np.zeros((len(x),8),dtype=np.float32);a=x.numpy()
        for tree in self.trees:
            for i,row in enumerate(a):
                node=0
                while tree['left'][node]>=0:node=tree['left'][node] if row[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
                result[i]+=tree['probability'][node]
        return torch.from_numpy(result/len(self.trees))

    def blend(self,base,paint):
        weight=self.weight
        if self.gate and float(base.max())<self.gate['base_ceiling'] and float(paint.max())>=self.gate['paint_floor']:weight=self.gate['weight']
        return base*(1-weight)+paint*weight
