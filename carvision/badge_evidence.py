"""Local emblem template evidence learned from manually annotated badge crops.

No automotive pretrained classifier. Similarity is not a probability; activation
is gated on held-out make labels. Only supported emblems can add soft evidence.
"""
import cv2
import numpy as np
import torch

def descriptor(image):
    a=np.asarray(image.convert('RGB'))
    grey=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    return grey-cv2.GaussianBlur(grey,(0,0),2)

class BadgeEvidence:
    def __init__(self,checkpoint):
        self.templates=checkpoint['templates'];self.threshold=float(checkpoint['threshold'])
        self.enabled=bool(checkpoint.get('enabled',False))
        self.brand_thresholds=checkpoint.get('brand_thresholds',{})

    def scores(self,image,mask=None):
        # Keep original pixels: upscaling cannot restore an unreadable emblem.
        if min(image.size)<40:return {}
        a=descriptor(image);h,w=a.shape;results={}
        for template in self.templates:
            raw=template['descriptor'].numpy();brand=template['brand'];best=results.get(brand,dict(similarity=-1.))
            for width in sorted({round(w*f) for f in [.035,.05,.07,.10,.14,.19]}):
                height=round(width*raw.shape[0]/raw.shape[1])
                if width<12 or height<6 or height>=h*.5:continue
                t=cv2.resize(raw,(width,height));y0=int(h*.22);y1=int(h*.90)
                search=a[y0:y1]
                if search.shape[0]<height:continue
                response=cv2.matchTemplate(search,t,cv2.TM_CCOEFF_NORMED)
                if mask is not None:
                    m=np.asarray(mask,dtype=np.uint8);coverage=cv2.boxFilter(m.astype(np.float32),-1,(width,height),normalize=True)
                    yy=np.arange(response.shape[0])+y0+height//2;xx=np.arange(response.shape[1])+width//2
                    response[coverage[np.ix_(yy,xx)]<.8]=-1
                _,score,_,location=cv2.minMaxLoc(response)
                if score>best['similarity']:
                    x,y=location;best=dict(similarity=float(score),box=[x,y+y0,x+width,y+y0+height],brand=brand)
            if best.get('brand') and best['similarity']>-1:
                results[brand]=best
        return results

    def predict(self,image,mask=None):
        if not self.enabled:return None
        ranked=sorted(self.scores(image,mask).values(),key=lambda r:r['similarity'],reverse=True)
        if not ranked:return None
        best=dict(ranked[0]);margin=best['similarity']-ranked[1]['similarity'] if len(ranked)>1 else 1.
        threshold=self.brand_thresholds.get(best['brand'],1.)
        best.update(accepted=self.enabled and best['similarity']>=threshold and margin>=.08,margin=margin)
        return best

def fuse_badge(probability,pairs,evidence):
    """Reserve at most30% of model mass for a validated emblem; never hard-filter."""
    if not evidence or not evidence.get('accepted'):return probability
    support=torch.tensor([b==evidence['brand'] for b,m in pairs],dtype=probability.dtype,device=probability.device)
    if not support.any():return probability
    selected=probability*support
    if selected.sum()<=0:return probability
    return probability*.7+selected/selected.sum()*.3
