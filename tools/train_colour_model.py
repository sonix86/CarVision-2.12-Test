"""Learn original colour decisions on VCoR labels. Official test is evaluated once at the end."""
import collections,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS,read_image
from carvision.colour_model import PREPROCESS,colour_statistics,colour_head
from torchvision import models

torch.set_num_threads(4);torch.manual_seed(3210);random.seed(3210);np.random.seed(3210)
root=ROOT/'datasets/vcor/images';out=ROOT/'models';source='https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset'
aliases={c:c for c in COLOURS};aliases.update(gray='grey',silver='grey',beige='brown',gold='brown',tan='brown',orange='yellow')
rows=[];seen=set();perceptual=set();excluded=collections.Counter()
files=sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'))
for split,folder_names in [('train',{'train','training'}),('validation',{'val','valid','validation'}),('test',{'test','testing'})]:
    for path in files:
        parts=[p.lower() for p in path.relative_to(root).parts]
        if not any(p in folder_names for p in parts[:-1]):continue
        label=next((aliases[p] for p in parts[:-1] if p in aliases),None)
        if label is None:excluded['unsupported_colour']+=1;continue
        try:
            image=read_image(path);digest=hashlib.sha256(path.read_bytes()).hexdigest();a=np.asarray(image.resize((9,8)).convert('L'));dhash=np.packbits(a[:,1:]>a[:,:-1]).tobytes().hex()
        except Exception:excluded['unreadable']+=1;continue
        if digest in seen or dhash in perceptual:excluded['duplicate_or_identical_dhash']+=1;continue
        seen.add(digest);perceptual.add(dhash);rows.append(dict(image=path.relative_to(root).as_posix(),colour=label,split=split,sha256=digest,dhash=dhash))
print('Data',dict(collections.Counter(r['split'] for r in rows)),dict(excluded),flush=True)
if not all(any(r['split']==s for r in rows) for s in ['train','validation','test']):raise ValueError('Expected original VCoR train/val/test folders')
(ROOT/'data/public_prototype/colour_dataset_manifest.json').write_text(json.dumps(dict(source=source,rows=rows,excluded=dict(excluded),aliases=aliases,split='Original folders; exact SHA256 and identical 64-bit dHash duplicates removed across splits in train/validation/test order; vehicle identities unavailable'),indent=2))
cache=ROOT/'cache/vcor_features.pt';cache.parent.mkdir(exist_ok=True)
backbone=models.mobilenet_v3_small(weights=None).features.eval();backbone.load_state_dict(torch.load(out/'generic_brand_backbone.pt',weights_only=True)['state'])
fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
cached=torch.load(cache,weights_only=True) if cache.exists() else None
if cached and cached.get('fingerprint')==fingerprint:x=cached['x']
else:
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            images=[read_image(root/r['image']) for r in rows[start:start+32]];f=backbone(torch.stack([PREPROCESS(im) for im in images]));pooled=torch.cat([f.mean((2,3)),f.amax((2,3))],1)
            vectors.append(torch.cat([pooled,torch.stack([colour_statistics(im) for im in images])],1))
            if start%1024==0:print('Extracted',start,'/',len(rows),flush=True)
    x=torch.cat(vectors);torch.save(dict(x=x,fingerprint=fingerprint),cache)
y=torch.tensor([COLOURS.index(r['colour']) for r in rows]);indices={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
train=indices['train'];validation=indices['validation'];test=indices['test'];mean=x[train].mean(0);std=x[train].std(0).clamp_min(.05);x=(x-mean)/std
head=colour_head(x.shape[1]);optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100)
counts=torch.bincount(y[train],minlength=8).float();weights=counts.rsqrt();weights/=weights.mean();loss_fn=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
history=[];best=-1;best_epoch=0;started=time.perf_counter()
for epoch in range(100):
    head.train();loss_sum=0.;correct=0
    for batch in train[torch.randperm(len(train))].split(64):
        optimizer.zero_grad();p=head(x[batch]);loss=loss_fn(p,y[batch]);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(batch);correct+=int((p.argmax(1)==y[batch]).sum())
    scheduler.step();head.eval()
    with torch.inference_mode():pred=head(x[validation]).argmax(1);accuracy=float((pred==y[validation]).float().mean())
    row=dict(epoch=epoch+1,loss=loss_sum/len(train),train_accuracy=correct/len(train),validation_accuracy=accuracy);history.append(row);print(json.dumps(row),flush=True)
    if accuracy>best:
        best=accuracy;best_epoch=epoch+1
        ck=dict(origin='team-trained-colour-head-imagenet-backbone',architecture='general_mobilenet_avgmax_plus_spatial_colour_statistics_mlp_v1',classes=COLOURS,backbone=backbone.state_dict(),head=head.state_dict(),mean=mean,std=std,source=source,ready_made_colour_model_used=False,best_epoch=best_epoch)
        torch.save(ck,out/'colour_model.tmp');(out/'colour_model.tmp').replace(out/'colour_model.pt')
    if epoch+1-best_epoch>=15 and epoch>=29:break
ck=torch.load(out/'colour_model.pt',weights_only=True);head.load_state_dict(ck['head']);head.eval()
with torch.inference_mode():pred=head(x[test]).argmax(1)
matrix=torch.zeros((8,8),dtype=torch.int64)
for truth,prediction in zip(y[test],pred):matrix[int(truth),int(prediction)]+=1
metrics=dict(source=source,classes=COLOURS,train=len(train),validation=len(validation),test=len(test),counts={s:dict(collections.Counter(rows[i]['colour'] for i in index.tolist())) for s,index in indices.items()},best_epoch=best_epoch,best_validation_accuracy=best,test_accuracy=float((pred==y[test]).float().mean()),confusion_matrix=matrix.tolist(),history=history,seconds=time.perf_counter()-started,split='Original VCoR splits; exact/identical-dHash duplicates removed, no vehicle identity grouping available',ready_made_colour_model_used=False,method='Frozen general ImageNet MobileNetV3 features + original RGB/HSV statistics + independently trained MLP decision head',test_used_for_checkpoint_selection=False)
(out/'colour_training.json').write_text(json.dumps(metrics,indent=2));print('FINAL',json.dumps({k:v for k,v in metrics.items() if k not in ['history','counts']}),flush=True)
