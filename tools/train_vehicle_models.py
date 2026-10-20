"""Train a fresh model-name head on grouped regional labels, never on demo photos."""
import sys,json,copy,time,collections,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from PIL import ImageOps
from torchvision import transforms
from carvision.core import ROOT,Classifier,read_image,make_preprocess
from carvision.vehicle_model import pooled_features

torch.set_num_threads(3);torch.manual_seed(321014);started=time.perf_counter()
ck=torch.load(ROOT/'models/classifier.pt',weights_only=True);network=Classifier(len(ck['brands'])).eval();network.load_state_dict(ck['state'])
rows=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
def label(r):
    s=r['vehicle_category']
    if s=='VAZ-2101':return '2101'
    return re.sub(r' \(.*?\)','',s).removeprefix(r['brand']+' ').removesuffix(' sedan')
for r in rows:r['model']=label(r)
labels=sorted(set((r['brand'],r['model']) for r in rows));names=[v for b,v in labels];brands=[b for b,v in labels]
pre=make_preprocess(160,16)
augment=transforms.Compose([transforms.RandomResizedCrop(160,scale=(.65,1.),ratio=(.8,1.25)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.12,.12,.08),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
cache=ROOT/'cache/model_name_features.pt'
if cache.exists():data=torch.load(cache,weights_only=True)
else:
    data={}
    with torch.inference_mode():
        for split in ['train','validation','test']:
            xs=[];ys=[]
            for r in rows:
                if r['split']!=split:continue
                im=read_image(ROOT/'data/regional'/r['image']).crop(r['box'])
                views=[augment(im) for _ in range(8)] if split=='train' else [pre(im),pre(ImageOps.mirror(im))]
                xs.append(pooled_features(network,torch.stack(views)));ys.extend([labels.index((r['brand'],r['model']))]*len(views))
            data[split]=dict(x=torch.cat(xs),y=torch.tensor(ys))
            print('FEATURES',split,len(ys),flush=True)
    torch.save(data,cache)
mean=data['train']['x'].mean(0);std=data['train']['x'].std(0).clamp_min(.1)
for d in data.values():d['x']=(d['x']-mean)/std
head=nn.Linear(len(mean),len(labels));optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.1)
counts=torch.bincount(data['train']['y'],minlength=len(labels)).float();weights=counts.clamp_min(1).pow(-.5);weights/=weights.mean()
best=-1;history=[]
for epoch in range(180):
    head.train()
    for ids in torch.randperm(len(data['train']['y'])).split(128):
        optimizer.zero_grad();loss=nn.functional.cross_entropy(head(data['train']['x'][ids]),data['train']['y'][ids],weight=weights,label_smoothing=.1);loss.backward();optimizer.step()
    head.eval()
    with torch.no_grad():
        p=head(data['validation']['x']).reshape(-1,2,len(labels)).softmax(2).mean(1);truth=data['validation']['y'][::2];accuracy=float((p.argmax(1)==truth).float().mean());nll=float(nn.functional.nll_loss(p.log(),truth))
    score=accuracy-.01*nll
    if score>best:best=score;state=copy.deepcopy(head.state_dict());best_epoch=epoch+1
    history.append(dict(epoch=epoch+1,validation_accuracy=accuracy,nll=nll))
head.load_state_dict(state)
with torch.no_grad():v=head(data['validation']['x']);truth=data['validation']['y']
temperature=min([.75,1.,1.25,1.5,2.,3.],key=lambda t:float(nn.functional.cross_entropy(v/t,truth)))
with torch.no_grad():
    vp=(v/temperature).softmax(1).reshape(-1,2,len(labels)).mean(1);vy=truth[::2]
# Fix rejection using development only. Unknown models have no independent test here.
threshold=.65
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
    accept=vp.max(1).values>=t
    if int(accept.sum())>=15 and float((vp.argmax(1)[accept]==vy[accept]).float().mean())>=.9:threshold=t;break
with torch.no_grad():p=(head(data['test']['x'])/temperature).softmax(1).reshape(-1,2,len(labels)).mean(1);y=data['test']['y'][::2];accept=p.max(1).values>=threshold
info=dict(origin='project-trained-regional-model-head',architecture='frozen project brand backbone + independently trained 1152-dimensional linear model head',labels=names,brands=brands,counts={b+' '+m:dict(collections.Counter(r['split'] for r in rows if (r['brand'],r['model'])==(b,m))) for b,m in labels},best_epoch=best_epoch,temperature=temperature,threshold=threshold,validation_accuracy=float((vp.argmax(1)==vy).float().mean()),test_count=len(y),test_accuracy=float((p.argmax(1)==y).float().mean()),test_coverage=float(accept.float().mean()),test_accepted_accuracy=float((p.argmax(1)[accept]==y[accept]).float().mean()) if accept.any() else None,history=history,seconds=time.perf_counter()-started,limitations='Public category labels; sparse Damas/Nexia/Cobalt samples; no own photographs. Existing grouped train/validation/test preserved. Backbone previously trained on regional training only. No open-set model benchmark. Demo photos excluded.')
torch.save(dict(origin=info['origin'],labels=names,brands=brands,mean=mean,std=std,state=state,temperature=temperature,threshold=threshold),ROOT/'models/vehicle_model.pt')
(ROOT/'models/vehicle_model_training.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in info.items() if k!='history'},ensure_ascii=False),flush=True)
