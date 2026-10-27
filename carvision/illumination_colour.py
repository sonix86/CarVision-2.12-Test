"""Our foreground paint statistics with scene-derived neutral illumination evidence."""
import numpy as np,torch
from PIL import Image
from .masked_colour import features as paint_features

def scene_gains(image,mask):
    rgb=np.asarray(image.convert('RGB').resize((160,160)),dtype=np.float32)/255
    fg=np.asarray(Image.fromarray(np.asarray(mask,dtype=np.uint8)*255).resize((160,160),Image.Resampling.NEAREST))>127
    hi=rgb.max(2);lo=rgb.min(2);sat=(hi-lo)/np.maximum(hi,.001)
    valid=(~fg)&(sat<.3)&(hi>.18)&(hi<.95)
    if valid.sum()<100:return np.ones(3,dtype=np.float32)
    neutral=np.median(rgb[valid],axis=0).clip(.05,1);target=float(np.prod(neutral)**(1/3))
    return np.clip(target/neutral,.75,1.33).astype(np.float32)

def features(image,mask,gains=None):
    gains=scene_gains(image,mask) if gains is None else np.asarray(gains,dtype=np.float32)
    from .segmentation import tight_box
    box=tight_box(np.asarray(mask,dtype=bool))
    if box:
        image=image.crop(box);mask=np.asarray(mask)[box[1]:box[3],box[0]:box[2]]
    corrected=Image.fromarray(np.clip(np.asarray(image.convert('RGB'),dtype=np.float32)*gains,0,255).astype(np.uint8))
    return torch.cat([paint_features(image,mask),paint_features(corrected,mask),torch.from_numpy(gains.copy())])

def head(dimension):return torch.nn.Sequential(torch.nn.Linear(dimension,96),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(96,8))

class IlluminationColour:
    def __init__(self,checkpoint):
        if checkpoint.get('origin')!='project-trained-illumination-paint-v24':raise ValueError('Expected own paint classifier')
        from .core import COLOURS
        if checkpoint.get('classes')!=COLOURS:raise ValueError('Unexpected paint label order')
        if not 0<=checkpoint['weight']<=1 or checkpoint['temperature']<=0:raise ValueError('Invalid paint calibration')
        self.mean=checkpoint['mean'];self.std=checkpoint['std'];self.network=head(len(self.mean)).eval();self.network.load_state_dict(checkpoint['head'])
        self.temperature=checkpoint['temperature'];self.weight=checkpoint['weight'];self.threshold=checkpoint.get('threshold',.8)
    @torch.inference_mode()
    def probabilities(self,image,mask,gains=None):return (self.network((features(image,mask,gains)-self.mean)/self.std)/self.temperature).softmax(0)
    def blend(self,base,paint):return base*(1-self.weight)+paint*self.weight
