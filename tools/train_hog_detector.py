"""Fit original linear HOG car windows on real annotated car crops/background regions."""
import collections
import json
import math
import random
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.hog_detector import descriptor,HOGCarDetector
from carvision.core import ROOT,read_image

torch.set_num_threads(2);cv2.setNumThreads(1);random.seed(3210);torch.manual_seed(3210)
root=ROOT/'data/public_prototype';rows=[json.loads(s) for s in (root/'labels.jsonl').read_text(encoding='utf-8').splitlines() if s.strip() and json.loads(s)['official_split']=='train']
random.shuffle(rows);train=rows[:2200];validation=rows[2200:2450]
windows=[(64,48),(64,64),(80,48)];output=ROOT/'models';output.mkdir(exist_ok=True);learned=[];patch_metrics=[];started=time.perf_counter()
def patches(records,window):
    hog=descriptor(window);xs=[];ys=[]
    for index,row in enumerate(records):
        image=np.asarray(read_image(root/row['image']));image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);height,width=image.shape[:2];x,y,x2,y2=[round(v) for v in row['box']]
        positive=image[y:y2,x:x2]
        if positive.size==0:continue
        positive=cv2.resize(positive,window)
        for patch in [positive,cv2.flip(positive,1)]:xs.append(hog.compute(patch).reshape(-1));ys.append(1.)
        regions=[]
        if y>=24:regions.append(image[:y,:])
        if height-y2>=24:regions.append(image[y2:,:])
        if x>=24:regions.append(image[:,:x])
        if width-x2>=24:regions.append(image[:,x2:])
        for negative in regions[:3]:
            if min(negative.shape[:2])<16:continue
            xs.append(hog.compute(cv2.resize(negative,window)).reshape(-1));ys.append(0.)
    return torch.tensor(np.stack(xs)),torch.tensor(ys).unsqueeze(1)
for window in windows:
    print('Extracting original HOG features',window,flush=True);x,y=patches(train,window);vx,vy=patches(validation,window)
    linear=torch.nn.Linear(x.shape[1],1);optimizer=torch.optim.LBFGS(linear.parameters(),lr=.8,max_iter=100,history_size=12,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();logits=linear(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)+.00005*linear.weight.square().sum();loss.backward();return loss
    optimizer.step(closure)
    with torch.inference_mode():
        logits=linear(vx);pred=logits>0;accuracy=float((pred==(vy>0)).float().mean());precision=float(((pred)&(vy>0)).sum()/pred.sum().clamp_min(1));recall=float(((pred)&(vy>0)).sum()/(vy>0).sum().clamp_min(1))
    coefficients=torch.cat([linear.weight.detach().reshape(-1),linear.bias.detach()])
    learned.append(dict(window=list(window),weights=coefficients));patch_metrics.append(dict(window=window,train_patches=len(y),validation_patches=len(vy),patch_accuracy=accuracy,patch_precision=precision,patch_recall=recall,scope='Cropped positive cars vs sampled background regions, NOT full-scene detector precision/recall. Border negatives may include unannotated objects.'))
    print(json.dumps(patch_metrics[-1]),flush=True)
checkpoint=dict(origin='team-trained-from-scratch',architecture='hog_linear_multiscale_v1',models=learned,raw_threshold=2.0,source='public Stanford Cars train split only; 2200 real annotated images, original human car boxes',pretrained=False)
torch.save(checkpoint,output/'detector.tmp');(output/'detector.tmp').replace(output/'detector.pt')
(output/'public_detector_training.json').write_text(json.dumps(dict(architecture=checkpoint['architecture'],pretrained=False,train_images=len(train),validation_images=len(validation),patch_metrics=patch_metrics,seconds=time.perf_counter()-started,limitations='Patch metrics are not full-scene metrics. HOG is a CPU baseline; evaluate small/occluded cars separately.'),indent=2),encoding='utf-8')
print('Original HOG detector fitted and saved.',flush=True)
