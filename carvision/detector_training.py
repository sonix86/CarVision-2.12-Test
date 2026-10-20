"""Train a two-class SSDLite detector on the team's bounding boxes, from random weights."""
import json
import random
import time
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms.functional import to_tensor,hflip
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import box_iou
from .core import read_image

class Scenes(Dataset):
    def __init__(self,records,augment=False):
        groups=defaultdict(list)
        for r in records:groups[str(r['path'])].append(r)
        self.scenes=list(groups.values());self.augment=augment
    def __len__(self):return len(self.scenes)
    def __getitem__(self,index):
        rows=self.scenes[index];im=read_image(rows[0]['path']);w,h=im.size
        boxes=torch.tensor([r['box'] for r in rows],dtype=torch.float32)
        im.thumbnail((1280,1280));boxes*=torch.tensor([im.width/w,im.height/h]*2)
        if self.augment and random.random()<.5:
            im=hflip(im);left=boxes[:,0].clone();boxes[:,0]=im.width-boxes[:,2];boxes[:,2]=im.width-left
        return to_tensor(im),dict(boxes=boxes,labels=torch.ones(len(boxes),dtype=torch.int64))

def collate(batch):return tuple(zip(*batch))

@torch.inference_mode()
def measure(model,loader,device):
    model.eval();tp=fp=fn=0
    for images,targets in loader:
        output=model([im.to(device) for im in images])
        for pred,target in zip(output,targets):
            boxes=pred['boxes'][pred['scores']>=.5].cpu();gt=target['boxes'];matched=set()
            for box in boxes:
                overlaps=box_iou(box.unsqueeze(0),gt)[0]
                candidates=[i for i in range(len(gt)) if i not in matched and overlaps[i]>=.5]
                if candidates:matched.add(max(candidates,key=lambda i:float(overlaps[i])));tp+=1
                else:fp+=1
            fn+=len(gt)-len(matched)
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return dict(precision=precision,recall=recall,f1=2*precision*recall/max(1e-9,precision+recall),tp=tp,fp=fp,fn=fn,scope='Single-view detector, score>=0.5 and IoU>=0.5 on held-out labelled scenes. Not brand accuracy, not tiled end-to-end performance.')

def train_detector(tr,va,out,epochs,seed,lr):
    torch.manual_seed(seed);random.seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=2).to(device)
    loader=DataLoader(Scenes(tr,True),batch_size=2,shuffle=True,num_workers=0,collate_fn=collate)
    validation=DataLoader(Scenes(va),batch_size=2,num_workers=0,collate_fn=collate)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    history=[];best=-1;started=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0.
        for images,targets in loader:
            if len(images)==1:
                for layer in model.modules():
                    if isinstance(layer,nn.BatchNorm2d):layer.eval()
            images=[im.to(device) for im in images];targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
            optimizer.zero_grad();losses=model(images,targets);loss=sum(losses.values())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();total+=float(loss.detach())
        metrics=measure(model,validation,device)
        row=dict(stage='detector',epoch=epoch+1,loss=total/max(1,len(loader)),**metrics);history.append(row);print(json.dumps(row),flush=True)
        if metrics['f1']>best:
            best=metrics['f1'];torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},origin='team-trained-from-scratch',architecture='ssdlite320_mobilenet_v3_large_two_classes',seed=seed,parameters=sum(p.numel() for p in model.parameters())),out/'detector.tmp');(out/'detector.tmp').replace(out/'detector.pt')
    (out/'detector_training.json').write_text(json.dumps(dict(history=history,epochs=epochs,seconds=time.perf_counter()-started,pretrained=False,hardware=device,parameters=sum(p.numel() for p in model.parameters())),indent=2),encoding='utf-8')
