"""Frozen image features shared by preparation and fitting; no labels fit here."""
import hashlib
import numpy as np,torch
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
from torchvision import models
from carvision.body_expert import body_tensor
from carvision.segmentation import isolate
TRAIN_VIEWS=['whole','phone','grey','badge_hidden','upper']
EVAL_VIEWS=['whole','phone','upper']

def key(row,kind):
    return hashlib.sha256(row['body_rgb'].numpy().tobytes()+row['body_mask'].numpy().tobytes()+kind.encode()).hexdigest()

def views(row,kind):
    im=Image.fromarray(row['body_rgb'].numpy());mask=row['body_mask'].numpy()
    if kind=='phone':
        im.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));im=ImageEnhance.Brightness(im).enhance(.45).filter(ImageFilter.GaussianBlur(.5))
    if kind=='grey':im=ImageOps.grayscale(im).convert('RGB')
    if kind=='occluded':
        im=im.copy();w,h=im.size;ImageDraw.Draw(im).rectangle((w*.3,h*.25,w*.55,h*.55),fill=(110,110,110))
    if kind=='badge_hidden':
        im=im.copy();w,h=im.size;ImageDraw.Draw(im).rectangle((w*.25,h*.45,w*.75,h*.8),fill=(110,110,110))
    if kind=='upper':
        bottom=max(1,round(im.height*.65));im=im.crop((0,0,im.width,bottom));mask=mask[:bottom]
    focused=isolate(im,mask)
    return torch.stack([body_tensor(v) for a in [im,focused] for v in [a,ImageOps.mirror(a)]])

def build(rows,checkpoint,source_sha,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    body=checkpoint['body_expert'];encoder=models.resnet18(weights=None);encoder.fc=torch.nn.Identity()
    encoder.load_state_dict({k:v for k,v in body['state'].items() if not k.startswith('fc.')});encoder.eval()
    old_head=torch.nn.Linear(512,len(checkpoint['brands']));old_head.load_state_dict({k[3:]:v for k,v in body['state'].items() if k.startswith('fc.')});old_head.eval()
    cache=torch.load(path,weights_only=True) if path.exists() else dict(source_sha=source_sha,items={})
    assert cache['source_sha']==source_sha
    jobs=[]
    for row in rows:
        for kind in (TRAIN_VIEWS if row['split']=='train' else EVAL_VIEWS):
            token=key(row,kind)
            if token not in cache['items']:jobs.append((token,row,kind))
    with torch.inference_mode():
        for first in range(0,len(jobs),8):
            batch=jobs[first:first+8];x=torch.cat([views(row,kind) for token,row,kind in batch]);z=encoder(x).reshape(-1,4,512)
            feature=z[:,:2].mean(1)*.75+z[:,2:].mean(1)*.25
            p=(old_head(z)/body['temperature']).softmax(2);p=p[:,:2].mean(1)*.75+p[:,2:].mean(1)*.25
            for j,(token,row,kind) in enumerate(batch):cache['items'][token]=dict(feature=torch.nn.functional.normalize(feature[j].clone(),dim=0),old=p[j].clone())
            if first%200==0:torch.save(cache,path);print('FEATURES',first,len(jobs),flush=True)
    torch.save(cache,path)
    return encoder,cache
