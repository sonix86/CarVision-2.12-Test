"""Train the team's brand decision on public data; no ready-made car-brand weights.

Default: random initialization. ImageNet use requires explicit --imagenet and user approval.
Official Stanford test images are never used for gradient updates or checkpoint selection.
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torchvision import models,transforms
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import Classifier,CompactResidualClassifier,ROOT,read_image

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=80);p.add_argument('--size',type=int,default=128);p.add_argument('--imagenet',action='store_true');p.add_argument('--resume');p.add_argument('--residual',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(3210);random.seed(3210);np.random.seed(3210)
    root=ROOT/'data/public_prototype';out=ROOT/('models/residual_candidate' if args.residual else 'models');out.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (root/'labels.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    development=[r for r in rows if r['official_split']=='train'];brands=sorted({r['brand'] for r in development});groups=collections.defaultdict(list)
    for r in development:groups[r['brand']].append(r)
    tr=[];va=[]
    for brand,records in sorted(groups.items()):
        records=sorted(records,key=lambda r:r['original_sha256']);random.Random(3210).shuffle(records);n=max(5,round(len(records)*.18));va.extend(records[:n]);tr.extend(records[n:])
    (out/'public_split.json').write_text(json.dumps(dict(train=[r['image'] for r in tr],validation=[r['image'] for r in va],official_test=[r['image'] for r in rows if r['official_split']=='test']),indent=2),encoding='utf-8')
    def cache(records):
        tensors=[];labels=[]
        for r in records:
            im=read_image(root/r['image']);x,y,x2,y2=r['box'];padx=(x2-x)*.04;pady=(y2-y)*.04
            crop=im.crop((max(0,x-padx),max(0,y-pady),min(im.width,x2+padx),min(im.height,y2+pady))).resize((args.size+16,args.size+16))
            tensors.append(torch.from_numpy(np.array(crop).copy()).permute(2,0,1));labels.append(brands.index(r['brand']))
        return torch.stack(tensors),torch.tensor(labels)
    print('Caching real annotated crops...',flush=True);train_x,train_y=cache(tr);val_x,val_y=cache(va)
    model=(CompactResidualClassifier if args.residual else Classifier)(len(brands));initialization='random-from-scratch'
    if args.imagenet:
        import os
        os.environ['TORCH_HOME']=str(ROOT/'cache')
        model.features=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).features;initialization='ImageNet general backbone; team-trained brand head'
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True);model.load_state_dict(ck['state']);initialization='continued own prototype training'
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002 if not args.imagenet else .0003,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    counts=torch.bincount(train_y,minlength=len(brands)).float();weights=counts.rsqrt();weights/=weights.mean();criterion=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    def batch_image(data,indices,augment=False):
        x=data[indices].float()/255
        if augment:
            top=random.randrange(17);left=random.randrange(17);x=x[:,:,top:top+args.size,left:left+args.size]
            if random.random()<.5:x=x.flip(3)
            x=(x*(.85+random.random()*.3)).clamp(0,1)
        else:x=x[:,:,8:8+args.size,8:8+args.size]
        return (x-mean)/std
    best=-1.;best_epoch=0;history=[];started=time.perf_counter();batch_size=32
    print(json.dumps(dict(train=len(tr),validation=len(va),brands=brands,initialization=initialization,input_size=args.size)),flush=True)
    for epoch in range(args.epochs):
        epoch_start=time.perf_counter();model.train();loss_sum=correct=0
        permutation=torch.randperm(len(train_y))
        for indices in permutation.split(batch_size):
            if len(indices)==1:continue
            x=batch_image(train_x,indices,True);target=train_y[indices];optimizer.zero_grad();b,_=model(x);loss=criterion(b,target);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(indices);correct+=int((b.argmax(1)==target).sum())
        scheduler.step();model.eval();pred=[];conf=[]
        with torch.inference_mode():
            for indices in torch.arange(len(val_y)).split(64):
                b,_=model(batch_image(val_x,indices));probs=b.softmax(1);pred.extend(probs.argmax(1).tolist());conf.extend(probs.max(1).values.tolist())
        accuracy=float((torch.tensor(pred)==val_y).float().mean())
        row=dict(stage='brand',epoch=epoch+1,loss=loss_sum/len(train_y),train_accuracy=correct/len(train_y),validation_accuracy=accuracy,seconds=round(time.perf_counter()-epoch_start,2));history.append(row);print(json.dumps(row),flush=True)
        if accuracy>best:
            best=accuracy;best_epoch=epoch+1
            architecture='compact_residual_avgmax_v1' if args.residual else 'mobilenet_v3_small_avgmax_dualhead_v1'
            torch.save(dict(state=model.state_dict(),brands=brands,architecture=architecture,origin='team-trained-from-scratch' if not args.imagenet else 'team-trained-brand-head-imagenet-backbone',initialization=initialization,image_size=args.size,colour_mode='body_histogram',colour_head_trained=False,source='public Stanford Cars; no own Tashkent data',validation_accuracy=accuracy),out/'classifier.tmp');(out/'classifier.tmp').replace(out/'classifier.pt')
        (out/'public_brand_training.json').write_text(json.dumps(dict(history=history,train_cars=len(tr),validation_cars=len(va),brands=brands,best_epoch=best_epoch,best_validation_accuracy=best,seconds=time.perf_counter()-started,initialization=initialization,colour_truth='Not available. Colour head NOT trained; runtime uses own non-learned method.',official_test_used_for_training=False),indent=2),encoding='utf-8')
    print('Real brand training complete. Best validation accuracy '+str(best),flush=True)

if __name__=='__main__':main()
