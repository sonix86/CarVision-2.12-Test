"""Fit a part-view reference classifier from our train images; select on grouped validation only."""
import sys,time,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image,ImageOps
from carvision.core import ROOT,Classifier,make_preprocess
from carvision.segmentation import isolate
start=time.perf_counter();torch.set_num_threads(3);torch.manual_seed(321017)
ck=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);net=Classifier(len(ck['labels'])).eval();net.load_state_dict(ck['state'])
rows=torch.load(ROOT/'cache/regional_instance_training.pt',weights_only=True)['rows'];pre=make_preprocess(160,16)
scopes={'whole':(0,0,1,1),'upper55':(0,0,1,.55),'upper35':(0,0,1,.35),'left65':(0,0,.65,1),'right65':(.35,0,1,1)}
def part(im,mask,kind):
 x,y,r,b=scopes[kind];box=[round(x*im.width),round(y*im.height),round(r*im.width),round(b*im.height)];return im.crop(box),mask[box[1]:box[3],box[0]:box[2]]
@torch.inference_mode()
def evidence(images):
 x=torch.stack([pre(v.resize((176,176),Image.Resampling.BICUBIC)) for im in images for v in [im,ImageOps.mirror(im)]])
 features=net.features(x);f=net.neck(torch.cat([features.mean((2,3)),features.amax((2,3))],1));logits=net.brand(f)/ck['temperature'];f=torch.nn.functional.normalize(f,dim=1).reshape(-1,2,256).mean(1);return torch.nn.functional.normalize(f,dim=1),logits.softmax(1).reshape(-1,2,len(ck['labels'])).mean(1)
cache=ROOT/'cache/visual_parts_v17.pt'
# Cache is tied to the exact source model file; never reuse stale descriptors.
import hashlib
model_sha=hashlib.sha256((ROOT/'models/vehicle_model.pt').read_bytes()).hexdigest()
if cache.exists():
 prepared=torch.load(cache,weights_only=True)
 if prepared['model_sha']!=model_sha:raise RuntimeError('Part cache model hash mismatch')
else:
 prepared={'model_sha':model_sha,'train':[],'validation':[],'test':[]}
 for split in ['train','validation','test']:
  jobs=[]
  for row in [r for r in rows if r['split']==split]:
   im=Image.fromarray(row['rgb'].numpy());mask=row['mask'].numpy()
   for kind in scopes if split=='train' else ['whole','upper55','upper35']:
    image,m=part(im,mask,kind)
    for masked in [False,True]:jobs.append((isolate(image,m) if masked else image,dict(y=row['y'],image=row['image'],kind=kind,masked=masked,ratio=image.width/image.height)))
  for first in range(0,len(jobs),32):
   batch=jobs[first:first+32];f,p=evidence([a for a,_ in batch])
   prepared[split].extend([dict(meta,feature=features,prob=prob) for (_,meta),features,prob in zip(batch,f,p)])
   if first%320==0:print('PART FEATURES',split,first,'/',len(jobs),flush=True)
  torch.save(prepared,cache)
source_rows={r['image']:r for r in [json.loads(line) for line in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]}
for split in ['train','validation','test']:
 for row in prepared[split]:
  box=source_rows[row['image']]['box'];x,y,r,b=scopes[row['kind']];row['ratio']=(box[2]-box[0])/(box[3]-box[1])*(r-x)/(b-y)
train=prepared['train'];bank=torch.stack([r['feature'] for r in train]);by=torch.tensor([r['y'] for r in train]);ratios=torch.tensor([r['ratio'] for r in train]);n=len(ck['labels'])
def dataset(split):
 out={}
 for kind in ['whole','upper55','upper35']:
  selected=[r for r in prepared[split] if r['kind']==kind];raw=selected[::2];masked=selected[1::2]
  out[kind]=dict(y=torch.tensor([r['y'] for r in raw]),base=torch.stack([a['prob']*.75+b['prob']*.25 for a,b in zip(raw,masked)]),features=[torch.stack([r['feature'] for r in group]) for group in [raw,masked]],ratios=torch.tensor([r['ratio'] for r in raw]))
 return out
val=dataset('validation');test=dataset('test')
def nearest(data,temp,penalty,k):
 probs=[]
 for features in data['features']:
  sim=features@bank.T-penalty*(data['ratios'][:,None].log()-ratios[None,:].log()).abs()
  scores=torch.stack([sim[:,by==i].topk(min(k,int((by==i).sum())),dim=1).values.mean(1) for i in range(n)],1)
  probs.append((scores/temp).softmax(1))
 return probs[0]*.75+probs[1]*.25
def metrics(p,y):return dict(top1=float((p.argmax(1)==y).float().mean()),top3=float((p.topk(3).indices==y[:,None]).any(1).float().mean()))
baseline={s:metrics(d['base'],d['y']) for s,d in val.items()};print('VALIDATION BASELINE',json.dumps(baseline),flush=True)
best=None;history=[]
for temp in [.03,.06,.1,.2]:
 for penalty in [0.,.05,.15]:
  for k in [1,3]:
   near={s:nearest(d,temp,penalty,k) for s,d in val.items()}
   for weight in [.25,.5,.75,1.]:
    combined={s:d['base']*(1-weight)+near[s]*weight for s,d in val.items()};m={s:metrics(combined[s],d['y']) for s,d in val.items()}
    clean_floor=m['whole']['top1']>=baseline['whole']['top1']-.02
    score=sum(v['top1']+.5*v['top3'] for v in m.values())/len(m)
    record=dict(temperature=temp,aspect_penalty=penalty,neighbours=k,weight=weight,validation=m,score=score,clean_floor=clean_floor);history.append(record)
    if clean_floor and (best is None or score>best['score']):best=record
print('BEST',json.dumps(best),flush=True)
validation_probs=torch.cat([d['base']*(1-best['weight'])+nearest(d,best['temperature'],best['aspect_penalty'],best['neighbours'])*best['weight'] for d in val.values()]);validation_y=torch.cat([d['y'] for d in val.values()])
def calibrate(p,t):
 p=p.clamp_min(1e-9).pow(1/t);return p/p.sum(1,keepdim=True)
probability_temperature=min([.5,.75,1.,1.5,2.,3.],key=lambda t:float(torch.nn.functional.nll_loss(calibrate(validation_probs,t).log(),validation_y)))
vp=calibrate(validation_probs,probability_temperature);confidence_threshold=.95
for threshold in [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]:
 accepted=vp.max(1).values>=threshold
 if int(accepted.sum())>=20 and float((vp.argmax(1)[accepted]==validation_y[accepted]).float().mean())>=.9:confidence_threshold=threshold;break
result={}
for split,data in [('validation',val),('test',test)]:
 result[split]={}
 for s,d in data.items():
  p=calibrate(d['base']*(1-best['weight'])+nearest(d,best['temperature'],best['aspect_penalty'],best['neighbours'])*best['weight'],probability_temperature);accept=p.max(1).values>=confidence_threshold;result[split][s]=dict(baseline=metrics(d['base'],d['y']),combined=metrics(p,d['y']),count=len(d['y']),accepted_count=int(accept.sum()),accepted_accuracy=float((p.argmax(1)[accept]==d['y'][accept]).float().mean()) if accept.any() else None)
# No model or fusion choice is made using test or the city development images.
info=dict(origin='project-trained-visible-part-references',selection='Grouped validation only; mean whole/upper55/upper35 top1 + half top3, whole top1 floor baseline minus2pts. Test and city images excluded from selection.',model_sha256=model_sha,train_photos=len([r for r in rows if r['split']=='train']),reference_views=len(train),scopes=list(scopes),selected=best,results=result,seconds=time.perf_counter()-start,candidates=history,test_used_for_selection=False,limitations='Closed set of 16 model names. Upper crops simulate missing body parts; they are not a real street occlusion test. Source categories and sparse Cobalt/Damas examples remain limitations.')
out=ROOT/'models/visual_parts_candidate';out.mkdir(exist_ok=True)
info.update(probability_temperature=probability_temperature,confidence_threshold=confidence_threshold,calibration='NLL temperature and >=90% accepted accuracy with >=20 examples, grouped validation whole+upper55+upper35 only; geometry-quality gate is separate.')
torch.save(dict(origin=info['origin'],labels=ck['labels'],brands=ck['brands'],features=bank,targets=by,ratios=ratios,**{key:best[key] for key in ['temperature','aspect_penalty','neighbours','weight']},source_model_sha256=model_sha),out/'visual_references.pt')
checkpoint=torch.load(out/'visual_references.pt',weights_only=True);checkpoint.update(probability_temperature=probability_temperature,confidence_threshold=confidence_threshold);torch.save(checkpoint,out/'visual_references.pt')
(out/'training.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print('FINAL',json.dumps({k:v for k,v in info.items() if k!='candidates'}),flush=True)
