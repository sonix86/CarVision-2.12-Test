"""Original car detector: fixed HOG features + team-trained linear decision functions.

OpenCV supplies the algorithm only. No bundled people/car detector coefficients are used.
"""
import cv2
import numpy as np
import torch
from torchvision.ops import nms

def descriptor(window):return cv2.HOGDescriptor(tuple(window),(16,16),(8,8),(8,8),9)

class HOGCarDetector:
    def __init__(self,checkpoint):
        self.models=[];self.raw_threshold=float(checkpoint.get('raw_threshold',1.5))
        for row in checkpoint['models']:
            hog=descriptor(row['window']);weights=row['weights'].cpu().numpy().astype(np.float32)
            hog.setSVMDetector(weights);self.models.append(hog)
        cv2.setNumThreads(2)
    def eval(self):return self
    def __call__(self,images):
        results=[]
        for tensor in images:
            rgb=(tensor.detach().cpu().permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
            original_h,original_w=rgb.shape[:2];scale=min(1.,960/max(original_h,original_w))
            image=cv2.cvtColor(cv2.resize(rgb,(max(80,round(original_w*scale)),max(80,round(original_h*scale)))),cv2.COLOR_RGB2BGR)
            sx,sy=original_w/image.shape[1],original_h/image.shape[0];boxes=[];scores=[]
            for hog in self.models:
                rects,weights=hog.detectMultiScale(image,self.raw_threshold,(8,8),(0,0),1.12,0,False)
                for (x,y,w,h),raw in zip(rects,np.asarray(weights).reshape(-1)):
                    boxes.append([x*sx,y*sy,(x+w)*sx,(y+h)*sy]);scores.append(float(1/(1+np.exp(-np.clip(raw,-20,20)))))
            b=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4);s=torch.tensor(scores,dtype=torch.float32)
            keep=nms(b,s,.4)[:100] if len(b) else torch.empty(0,dtype=torch.long)
            results.append(dict(boxes=b[keep],scores=s[keep],labels=torch.ones(len(keep),dtype=torch.long)))
        return results
