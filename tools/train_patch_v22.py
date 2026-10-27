"""One bounded expanded body expert; select on grouped validation, never user photos."""
import sys,json,hashlib,copy,random,time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image,ImageOps,ImageFilter,ImageEnhance,ImageDraw
from torchvision import models,transforms
from carvision.core import ROOT,make_preprocess
from carvision.body_expert import body_tensor
from carvision.vehicle_model import VehicleModelClassifier,pooled_features
from carvision.visual_parts import VisualReferences
from carvision.segmentation import isolate
from tools.cache_patch_baseline_v22 import key as baseline_key

torch.set_num_threads(3);torch.manual_seed(321022);random.seed(321022)
OUT=ROOT/'models/ev_patch_v22';start=time.perf_counter();epochs=10
data=torch.load(ROOT/'cache/ev_patch_v22.pt',weights_only=True);pairs=[tuple(p) for p in data['pairs']];rows=data['rows'];n=len(pairs)
baseck=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);refck=torch.load(ROOT/'models/visual_references.pt',weights_only=True);oldn=len(baseck['labels'])
baseline=VehicleModelClassifier(baseck);baseline.references=VisualReferences(refck)
source_sha=hashlib.sha256((ROOT/'models/vehicle_model.pt').read_bytes()).hexdigest()
baseline_path=ROOT/'cache/ev22_baseline_validation.pt'
baseline_cache=torch.load(baseline_path,weights_only=True) if baseline_path.exists() else dict(source_sha=source_sha,values={})
assert baseline_cache['source_sha']==source_sha
splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
if not any(r.get('source') for r in splits['train']):raise SystemExit('No newly collected data; do not pretend expansion')
pre=make_preprocess(160,16)
# Build new references only from newly admitted training photographs. Old bank intact.
features=[];targets=[];ratios=[]
with torch.inference_mode():
    for r in splits['train']:
        if r['y']<oldn:continue
        im=Image.fromarray(r['body_rgb'].numpy());focused=isolate(im,r['body_mask'].numpy())
        x=torch.stack([pre(v.resize((176,176),Image.Resampling.BICUBIC)) for a in [im,focused] for v in [a,ImageOps.mirror(a)]])
        f=baseline.network.neck(pooled_features(baseline.network,x));f=torch.nn.functional.normalize(f,dim=1)
        features.append(torch.nn.functional.normalize(f.mean(0),dim=0));targets.append(r['y']);ratios.append(im.width/im.height)
newref=dict(refck,labels=[m for b,m in pairs],brands=[b for b,m in pairs])
if features:
    newref.update(features=torch.cat([refck['features'],torch.stack(features)]),targets=torch.cat([refck['targets'],torch.tensor(targets)]),ratios=torch.cat([refck['ratios'],torch.tensor(ratios)]))
assert all((newref['targets']==j).any() for j in range(n))
expanded=dict(baseck,labels=[m for b,m in pairs],brands=[b for b,m in pairs],neural_labels=baseck['labels'],neural_brands=baseck['brands'])
expanded.pop('body_expert',None)
prior=VehicleModelClassifier(expanded);prior.references=VisualReferences(newref)

def view(r,kind):
    im=Image.fromarray(r['body_rgb'].numpy());mask=r['body_mask'].numpy()
    if kind=='phone':
        im.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST))
        im=ImageEnhance.Brightness(im).enhance(.48).filter(ImageFilter.GaussianBlur(.45))
    return im,mask

val=splits['validation'];yval=torch.tensor([r['y'] for r in val]);historic=torch.tensor(['source' not in r for r in val])
ev=torch.tensor([b in ['BYD','Geely','Zeekr','Hongqi','Weltmeister','XPeng'] or m in ['bZ4X','Ariya','Leaf'] for b,m in pairs])
def metrics(p):
    good=p.argmax(1)==yval;top3=(p.topk(3).indices==yval[:,None]).any(1);result={}
    for name,idx in [('all',torch.ones(len(val),dtype=torch.bool)),('historical',historic),('ev',ev[yval]),('new_photos',~historic)]:
        result[name]=dict(count=int(idx.sum()),top1=float(good[idx].float().mean()) if idx.any() else 0,top3=float(top3[idx].float().mean()) if idx.any() else 0,macro=float(torch.tensor([float(good[(yval==j)&idx].float().mean()) for j in yval[idx].unique()]).mean()) if idx.any() else 0)
    return result

validation={}
with torch.inference_mode():
    for kind in ['whole','phone']:
        old=[];pri=[];xs=[]
        for i,r in enumerate(val):
            im,mask=view(r,kind);focus=isolate(im,mask)
            cachekey=baseline_key(r,kind)
            if cachekey in baseline_cache['values']:a=baseline_cache['values'][cachekey]
            else:
                a=baseline.probabilities(im,None,None)*.75+baseline.probabilities(focus,None,None)*.25
                baseline_cache['values'][cachekey]=a
            old.append(torch.nn.functional.pad(a,(0,n-oldn)))
            pri.append(prior.probabilities(im,None,None)*.75+prior.probabilities(focus,None,None)*.25)
            xs.append(torch.stack([body_tensor(v) for a in [im,focus] for v in [a,ImageOps.mirror(a)]]))
            if i%50==0:print('VALIDATION',kind,i,len(val),flush=True)
        validation[kind]=dict(old=torch.stack(old),prior=torch.stack(pri),x=torch.cat(xs))
baseline_metrics={k:metrics(v['old']) for k,v in validation.items()}
torch.save(baseline_cache,baseline_path)
torch.save(dict(validation=validation,y=yval),OUT/'validation_inputs.pt')
print('BASELINE',json.dumps(baseline_metrics),flush=True)
net=models.resnet18(weights=None);net.fc=torch.nn.Linear(512,n)
state=copy.deepcopy(baseck['body_expert']['state']);newstate=net.state_dict()
for k,v in state.items():
    if not k.startswith('fc.'):newstate[k]=v
newstate['fc.weight'][:oldn]=state['fc.weight'];newstate['fc.bias'][:oldn]=state['fc.bias'];net.load_state_dict(newstate)
for name,p in net.named_parameters():p.requires_grad_(name.startswith(('layer4.','fc.')))
optimizer=torch.optim.AdamW([dict(params=net.layer4.parameters(),lr=.00004),dict(params=net.fc.parameters(),lr=.00065)],weight_decay=.02)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
train=splits['train'];ys=torch.tensor([r['y'] for r in train]);counts=torch.bincount(ys,minlength=n).float();sampling=counts[ys].clamp_min(1).pow(-.65)*torch.where(ev[ys],1.5,1.)
teacher=baseline.body_expert.network.eval()
for parameter in teacher.parameters():parameter.requires_grad_(False)
jitter=transforms.ColorJitter(.35,.25,.12,.025);history=[];best=-1.;bestepoch=0
def select(logits):
    options=[]
    for temp in [1.,1.5,2.]:
        for weight in [.5,.75,1.]:
            ps={}
            for k,z in logits.items():
                q=(z/temp).softmax(2);q=q[:,:2].mean(1)*.75+q[:,2:].mean(1)*.25
                ps[k]=validation[k]['prior']*(1-weight)+q*weight
            scores={k:metrics(p) for k,p in ps.items()}
            guard=(scores['whole']['historical']['top1']>=baseline_metrics['whole']['historical']['top1']-.01 and scores['whole']['historical']['top3']>=baseline_metrics['whole']['historical']['top3']-.02 and scores['phone']['historical']['top1']>=baseline_metrics['phone']['historical']['top1']-.02)
            gain=sum(scores[k]['ev']['macro']-baseline_metrics[k]['ev']['macro'] for k in scores)/2
            guard=guard and gain>=.03
            score=sum(.6*scores[k]['ev']['macro']+.4*scores[k]['all']['top1'] for k in scores)/2
            options.append(dict(guard=guard,score=score,ev_macro_gain=gain,temperature=temp,weight=weight,metrics=scores))
    allowed=[r for r in options if r['guard']]
    return max(allowed or options,key=lambda r:r['score'])

for epoch in range(epochs):
    net.train()
    for module in net.modules():
        if isinstance(module,torch.nn.BatchNorm2d):module.eval()
    ids=torch.multinomial(sampling,len(train)*2,replacement=True)
    for batch in ids.split(24):
        xs=[]
        for idx in batch:
            r=train[int(idx)];im=Image.fromarray(r['body_rgb'].numpy())
            if random.random()<.6:im=isolate(im,r['body_mask'].numpy())
            if random.random()<.5:im=ImageOps.mirror(im)
            im=jitter(im)
            if random.random()<.3:im=ImageEnhance.Brightness(im).enhance(random.uniform(.25,.7))
            if random.random()<.35:im.thumbnail((random.choice([64,96,128]),random.choice([64,96,128])))
            if random.random()<.15:im=ImageOps.grayscale(im).convert('RGB')
            if random.random()<.15:im=im.filter(ImageFilter.GaussianBlur(random.uniform(.3,.9)))
            if random.random()<.2:
                # Local occluder suppresses reliance on a visible badge/one headlight.
                w,h=im.size;x=random.randrange(max(1,w));y=random.randrange(max(1,h));im=im.copy();ImageDraw.Draw(im).rectangle((x,y,min(w,x+w//5),min(h,y+h//5)),fill=(110,110,110))
            xs.append(body_tensor(im))
        x=torch.stack(xs);optimizer.zero_grad();logits=net(x)
        loss=torch.nn.functional.cross_entropy(logits,ys[batch],label_smoothing=.06)
        old_items=ys[batch]<oldn
        if old_items.any():
            with torch.no_grad():soft=(teacher(x[old_items])/2).softmax(1)
            loss=loss+.5*4*torch.nn.functional.kl_div((logits[old_items,:oldn]/2).log_softmax(1),soft,reduction='batchmean')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
    scheduler.step();net.eval()
    with torch.inference_mode():zs={k:torch.cat([net(x) for x in v['x'].split(32)]).reshape(-1,4,n) for k,v in validation.items()}
    chosen=select(zs);record=dict(epoch=epoch+1,**chosen);history.append(record)
    if chosen['guard'] and chosen['score']>best:
        best=chosen['score'];bestepoch=epoch+1
        expert=dict(origin='project-trained-body-expert-v1',pairs=[list(p) for p in pairs],state=copy.deepcopy(net.state_dict()),weight=chosen['weight'],temperature=chosen['temperature'],make_threshold=.95,source_model_sha256=source_sha)
        torch.save(expert,OUT/'expert.pt');torch.save(zs,OUT/'selected_validation_logits.pt')
    (OUT/'progress.json').write_text(json.dumps(dict(best_epoch=bestepoch,history=history)),encoding='utf-8');print(json.dumps(record),flush=True)
torch.save(expanded,OUT/'base_expanded.pt');torch.save(newref,OUT/'references_expanded.pt')
report=dict(selected=bestepoch>0,best_epoch=bestepoch,epochs=epochs,source_model_sha256=source_sha,baseline_validation=baseline_metrics,history=history,test_used_for_selection=False,seconds=time.perf_counter()-start,selection='Grouped validation only; protect historical whole top1 within1pp, top3 within2pp, simulated phone top1 within2pp; require EV macro improvement3pp. Phone view is a synthetic robustness probe, not real unseen phone accuracy.')
(OUT/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('COMPLETE',bestepoch,flush=True)
