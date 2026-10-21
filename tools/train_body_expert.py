"""One finite corrected 12-epoch candidate. Select only on grouped validation.

ImageNet ResNet18 is a generic feature initializer, never a car classifier.
The immutable previous classifier is retained unless validation guards pass.
"""
import sys,json,time,copy,random,hashlib,collections,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image,ImageOps,ImageFilter
from torchvision import models,transforms
from carvision.core import ROOT,Classifier,make_preprocess
from carvision.segmentation import isolate
from carvision.body_expert import body_tensor
from carvision.vehicle_model import VehicleModelClassifier
from carvision.visual_parts import VisualReferences

torch.set_num_threads(3);torch.manual_seed(321020);random.seed(321020)
parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--warmstart',default='models/body_expert_round1/expert.pt');args=parser.parse_args()
START=time.perf_counter();OUT=ROOT/'models/body_expert_candidate';OUT.mkdir(exist_ok=True)
data=torch.load(ROOT/'cache/body_expert_v20.pt',weights_only=True);pairs=data['pairs'];rows=data['rows'];classes=len(pairs)
data_sha=hashlib.sha256((ROOT/'cache/body_expert_v20.pt').read_bytes()).hexdigest()
source=ROOT/'models/vehicle_model.pt';source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
base_ck=torch.load(source,weights_only=True);base=VehicleModelClassifier(base_ck);base.references=VisualReferences(torch.load(ROOT/'models/visual_references.pt',weights_only=True))
splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
views=['whole','small64','upper55'];chevy=torch.tensor([b=='Chevrolet' for b,m in pairs])

def images(row,kind):
    rgb=Image.fromarray(row['body_rgb'].numpy());mask=row['body_mask'].numpy()
    # Baseline receives precisely its original cached shape/pixel evidence.
    old=Image.fromarray(row['rgb'].numpy());oldmask=row['mask'].numpy()
    if kind=='upper55':
        end=max(1,round(rgb.height*.55));rgb=rgb.crop((0,0,rgb.width,end));mask=mask[:end]
        old=old.crop((0,0,176,97)).resize((176,176),Image.Resampling.BICUBIC);oldmask=np.asarray(Image.fromarray(oldmask[:97]).resize((176,176),Image.Resampling.NEAREST))
    elif kind=='small64':
        rgb.thumbnail((64,64));mask=np.asarray(Image.fromarray(mask).resize(rgb.size,Image.Resampling.NEAREST))
        # Old square crop shrunk to the same number of physical body pixels.
        small=Image.fromarray(row['body_rgb'].numpy());small.thumbnail((64,64));old=small.resize((176,176),Image.Resampling.BICUBIC)
    return rgb,mask,old,oldmask

@torch.inference_mode()
def old_probability(row,kind):
    rgb,mask,old,oldmask=images(row,kind)
    if kind=='small64':oldmask=row['mask'].numpy()
    p=base.probabilities(old,None,None)*.75+base.probabilities(isolate(old,oldmask),None,None)*.25
    return p

def metric(p,y):
    out={};c=chevy[y]
    for name,idx in [('all',torch.ones(len(y),dtype=torch.bool)),('chevrolet',c),('other',~c)]:
        if not idx.any():continue
        good=(p.argmax(1)==y);top3=(p.topk(3).indices==y[:,None]).any(1)
        macro=[float(good[y==k].float().mean()) for k in y[idx].unique()]
        out[name]=dict(count=int(idx.sum()),top1=float(good[idx].float().mean()),top3=float(top3[idx].float().mean()),macro_top1=sum(macro)/len(macro))
    return out

cache=ROOT/'cache/body_expert_validation_v20.pt'
if cache.exists():
    cached=torch.load(cache,weights_only=True)
    if cached['source_sha']!=source_sha or cached.get('data_sha')!=data_sha or cached['images']!=[r['image'] for r in splits['validation']]:raise RuntimeError('Stale validation cache')
    validation=cached['validation']
else:
    validation={}
    for kind in views:
        tensors=[];old=[]
        for i,row in enumerate(splits['validation']):
            rgb,mask,_,_=images(row,kind);focused=isolate(rgb,mask)
            tensors.append(torch.stack([body_tensor(v) for im in [rgb,focused] for v in [im,ImageOps.mirror(im)]]))
            old.append(old_probability(row,kind))
        validation[kind]=dict(x=torch.cat(tensors),old=torch.stack(old))
        print('BASELINE READY',kind,flush=True)
    torch.save(dict(source_sha=source_sha,data_sha=data_sha,images=[r['image'] for r in splits['validation']],validation=validation),cache)
yval=torch.tensor([r['y'] for r in splits['validation']]);baseline={k:metric(v['old'],yval) for k,v in validation.items()}
(OUT/'baseline_validation.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
print('BASELINE',json.dumps(baseline),flush=True)
del base
net=models.resnet18(weights=None);net.fc=torch.nn.Linear(512,classes)
warm=torch.load(ROOT/args.warmstart,weights_only=True)
assert warm['origin']=='project-trained-body-expert-v1' and warm['pairs']==[list(p) for p in pairs]
net.load_state_dict(warm['state'])
for name,param in net.named_parameters():param.requires_grad_(name.startswith(('layer3.','layer4.','fc.')))
optimizer=torch.optim.AdamW([dict(params=net.layer3.parameters(),lr=.000035),dict(params=net.layer4.parameters(),lr=.00008),dict(params=net.fc.parameters(),lr=.0007)],weight_decay=.02)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
train=splits['train'];y=torch.tensor([r['y'] for r in train]);counts=torch.bincount(y,minlength=classes).float();sampling=counts[y].clamp_min(1).pow(-.5)*torch.where(chevy[y],1.7,1.)
jitter=transforms.ColorJitter(.25,.18,.08,.015)
best=-float('inf');best_epoch=0;history=[]
@torch.inference_mode()
def logits_validation():
    return {k:torch.cat([net(x) for x in v['x'].split(32)]).reshape(-1,4,classes) for k,v in validation.items()}

def select(logits):
    candidates=[]
    for temp in [.75,1.,1.5,2.]:
        expert={k:(z/temp).softmax(2)[:,:2].mean(1)*.75+(z/temp).softmax(2)[:,2:].mean(1)*.25 for k,z in logits.items()}
        for weight in [.25,.5,.75,1.]:
            metrics={k:metric(validation[k]['old']*(1-weight)+p*weight,yval) for k,p in expert.items()}
            guard=metrics['whole']['all']['top1']>=baseline['whole']['all']['top1']-.01 and metrics['whole']['all']['top3']>=baseline['whole']['all']['top3']-.01 and all(metrics[k]['all']['top3']>=baseline[k]['all']['top3']-.02 for k in views)
            gain=sum(metrics[k]['chevrolet']['macro_top1']-baseline[k]['chevrolet']['macro_top1'] for k in views)/len(views)
            score=sum(.6*metrics[k]['chevrolet']['macro_top1']+.4*metrics[k]['all']['top1'] for k in views)/len(views)
            candidates.append(dict(guard=guard and gain>=.02,score=score,chevrolet_macro_gain=gain,weight=weight,temperature=temp,validation=metrics))
    valid=[c for c in candidates if c['guard']]
    return max(valid,key=lambda c:c['score']) if valid else max(candidates,key=lambda c:c['score'])

for epoch in range(args.epochs):
    net.train()
    # Stable generic low-level filters and running statistics on this small set.
    for module in net.modules():
        if isinstance(module,torch.nn.BatchNorm2d):module.eval()
    ids=torch.multinomial(sampling,len(train)*2,replacement=True)
    for batch in ids.split(24):
        xs=[]
        for idx in batch:
            row=train[int(idx)];im=Image.fromarray(row['body_rgb'].numpy());mask=row['body_mask'].numpy()
            if random.random()<.6:im=isolate(im,mask)
            if random.random()<.22:im=im.crop((0,0,im.width,max(1,round(im.height*random.uniform(.5,.8)))))
            if random.random()<.5:im=ImageOps.mirror(im)
            im=jitter(im)
            if random.random()<.45:
                side=random.choice([40,56,72,96]);im.thumbnail((side,side))
            if random.random()<.15:im=ImageOps.grayscale(im).convert('RGB')
            if random.random()<.12:im=im.filter(ImageFilter.GaussianBlur(.6))
            xs.append(body_tensor(im))
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(net(torch.stack(xs)),y[batch],label_smoothing=.04);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
    scheduler.step();net.eval();selection=select(logits_validation());record=dict(epoch=epoch+1,**selection);history.append(record)
    if selection['guard'] and selection['score']>best:
        best=selection['score'];best_epoch=epoch+1
        ck=dict(origin='project-trained-body-expert-v1',pairs=[list(p) for p in pairs],state=copy.deepcopy(net.state_dict()),weight=selection['weight'],temperature=selection['temperature'],initialization='Generic ImageNet ResNet18; our model-name head trained on training split only',source_model_sha256=source_sha)
        torch.save(ck,OUT/'expert.pt')
    (OUT/'progress.json').write_text(json.dumps(dict(best_epoch=best_epoch,history=history)),encoding='utf-8');print(json.dumps(record),flush=True)
info=dict(epochs=args.epochs,warmstart=args.warmstart,warmstart_sha256=hashlib.sha256((ROOT/args.warmstart).read_bytes()).hexdigest(),first_round_interrupted_after=9,first_round_reason='Correct source/mask crop alignment; no first-round weights promoted and no test used',best_epoch=best_epoch,selected=best_epoch>0,baseline_validation=baseline,history=history,selection='Grouped validation only. Whole all-class top1/top3 no more than1pp below baseline; each view top3 no more than2pp below. Mean Chevrolet macro top1 must improve2pp. Maximize60% Chevrolet macro and40% all-class top1 across whole/small64/upper55. No test/city selection.',source_model_sha256=source_sha,seconds=time.perf_counter()-START,test_used_for_selection=False)
(OUT/'training.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print('COMPLETE',best_epoch,info['seconds'],flush=True)
