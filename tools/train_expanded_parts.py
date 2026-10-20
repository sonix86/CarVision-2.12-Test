"""Bounded local training: 16 existing models plus eight exact Stanford annotation classes.
No download, no promotion, no use of city examples, colour pseudo-labels or test selection.
"""
import sys,json,time,hashlib,copy,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image,ImageOps
from scipy.io import loadmat
from torchvision import transforms
from torchvision.transforms import functional as TF
from carvision.core import ROOT,Classifier,make_preprocess,read_image
from carvision.segmentation import InstanceSegmenter,isolate,tight_box
started=time.perf_counter();torch.set_num_threads(3);torch.manual_seed(321018);random.seed(321018)
out=ROOT/'models/expanded_parts_candidate';out.mkdir(exist_ok=True)
old=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);old_pairs=list(zip(old['brands'],old['labels']))
new_names={'Toyota Camry Sedan 2012':('Toyota','Camry 2012'),'Toyota Corolla Sedan 2012':('Toyota','Corolla 2012'),'Hyundai Sonata Sedan 2012':('Hyundai','Sonata 2012'),'Hyundai Accent Sedan 2012':('Hyundai','Accent 2012'),'Hyundai Tucson SUV 2012':('Hyundai','Tucson 2012'),'Hyundai Santa Fe SUV 2012':('Hyundai','Santa Fe 2012'),'Honda Accord Sedan 2012':('Honda','Accord Sedan 2012'),'BMW 3 Series Sedan 2012':('BMW','3 Series Sedan 2012')}
classes=[str(v) for v in loadmat(ROOT/'datasets/stanford_cars/cars_annos.mat',squeeze_me=True,struct_as_record=False)['class_names']]
annotations={}
for split,file in [('train','cars_train_annos.mat'),('test','cars_test_annos_withlabels.mat')]:annotations[split]={str(r.fname):r for r in loadmat(ROOT/'datasets/stanford_cars'/file,squeeze_me=True,struct_as_record=False)['annotations']}
public_split=json.loads((ROOT/'models/public_split.json').read_text());train_set=set(public_split['train']);val_set=set(public_split['validation'])
public=[json.loads(l) for l in (ROOT/'data/public_prototype/labels.jsonl').read_text(encoding='utf-8').splitlines()]
added=[]
for row in public:
 split=row['official_split'];a=annotations[split][Path(row['original']).name];category=classes[int(getattr(a,'class'))-1]
 if category not in new_names:continue
 brand,name=new_names[category]
 if brand!=row['brand']:raise ValueError('Exact annotation brand mismatch')
 subset='test' if split=='test' else 'train' if row['image'] in train_set else 'validation' if row['image'] in val_set else None
 if subset is None:continue
 added.append(dict(image=row['image'],path=str(ROOT/'data/public_prototype'/row['image']),box=row['box'],brand=brand,model=name,split=subset,original=row['original'],source='Exact Stanford original MAT class joined through original Git blob mapping; not mirror fine_label',original_sha256=row['original_sha256']))
pairs=old_pairs+sorted(set((r['brand'],r['model']) for r in added))
assert len(pairs)==24
prepared=torch.load(ROOT/'cache/regional_instance_training.pt',weights_only=True)['rows']
expanded=[dict(r,y=pairs.index(old_pairs[r['y']]),legacy=True) for r in prepared]
cache=ROOT/'cache/expanded_parts_v18.pt';sha=hashlib.sha256((ROOT/'models/vehicle_model.pt').read_bytes()).hexdigest()
if cache.exists():
 cached=torch.load(cache,weights_only=True);assert cached['source_model_sha256']==sha and cached['pairs']==[list(p) for p in pairs];expanded=cached['rows']
else:
 segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True);fallback=0
 with torch.inference_mode():
  for i,r in enumerate(added):
   im=read_image(r['path']).crop(r['box']);im.thumbnail((384,384));d=segmenter.raw(im);ids=[j for j in range(len(d['scores'])) if int(d['labels'][j]) in (3,8)]
   mask=d['masks'][max(ids,key=lambda j:int((d['masks'][j,0]>.5).sum())),0].numpy()>.5 if ids else np.ones((im.height,im.width),bool)
   if not ids:fallback+=1
   if int(mask.sum())<20:mask=np.ones((im.height,im.width),bool);fallback+=1
   box=tight_box(mask);rgb=np.asarray(im.crop(box).resize((176,176),Image.Resampling.BICUBIC)).copy();m=np.asarray(Image.fromarray(mask[box[1]:box[3],box[0]:box[2]]).resize((176,176),Image.Resampling.NEAREST)).copy()
   expanded.append(dict(image='stanford/'+r['image'],split=r['split'],y=pairs.index((r['brand'],r['model'])),rgb=torch.from_numpy(rgb),mask=torch.from_numpy(m),legacy=False))
   if i%40==0:print('PREPARE NEW',i,'/',len(added),flush=True)
 del segmenter
 torch.save(dict(rows=expanded,pairs=[list(p) for p in pairs],source_model_sha256=sha,fallback_masks=fallback),cache)
sets={s:[r for r in expanded if r['split']==s] for s in ['train','validation','test']};pre=make_preprocess(160,16)
counts={b+' '+m:{s:sum(r['y']==j for r in sets[s]) for s in sets} for j,(b,m) in enumerate(pairs)}
(out/'dataset.json').write_text(json.dumps(dict(pairs=pairs,counts=counts,new_rows=added,source_model_sha256=sha,split='Existing regional groups and Stanford public_split preserved; official Stanford test remains test. Original MAT model labels, no colour pseudo-labels used. Stanford vehicle identities unavailable.',own_photos=0),ensure_ascii=False,indent=2),encoding='utf-8')
print('COUNTS',json.dumps(counts),flush=True)
normal=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);jitter=transforms.ColorJitter(.25,.2,.1,.02)
validation={}
for split in ['validation','test']:
 validation[split]={}
 for kind,end in [('whole',176),('upper55',97),('upper35',62)]:
  xs=[]
  for r in sets[split]:
   im=Image.fromarray(r['rgb'].numpy()).crop((0,0,176,end)).resize((176,176),Image.Resampling.BICUBIC);m=np.asarray(Image.fromarray(r['mask'].numpy()).crop((0,0,176,end)).resize((176,176),Image.Resampling.NEAREST));focused=isolate(im,m)
   xs.append(torch.stack([pre(im),pre(ImageOps.mirror(im)),pre(focused),pre(ImageOps.mirror(focused))]))
  validation[split][kind]=torch.cat(xs)
model=Classifier(len(pairs));state=model.state_dict()
for key,value in old['state'].items():
 if not key.startswith('brand.'):state[key]=value
for i,pair in enumerate(old_pairs):state['brand.weight'][pairs.index(pair)]=old['state']['brand.weight'][i];state['brand.bias'][pairs.index(pair)]=old['state']['brand.bias'][i]
model.load_state_dict(state);model.eval()
@torch.inference_mode()
def probs(x):
 logits=torch.cat([model(t)[0] for t in x.split(48)]).reshape(-1,4,len(pairs));p=(logits/old['temperature']).softmax(2);return p[:,:2].mean(1)*.75+p[:,2:].mean(1)*.25
ys={s:torch.tensor([r['y'] for r in sets[s]]) for s in sets};legacy={s:torch.tensor([r['legacy'] for r in sets[s]]) for s in sets}
def measure(split):
 result={}
 for kind,x in validation[split].items():
  p=probs(x);y=ys[split];result[kind]={}
  for scope,mask in [('all',torch.ones(len(y),dtype=torch.bool)),('legacy',legacy[split]),('new',~legacy[split])]:
   result[kind][scope]=dict(count=int(mask.sum()),top1=float((p.argmax(1)[mask]==y[mask]).float().mean()),top3=float((p.topk(3).indices[mask]==y[mask,None]).any(1).float().mean()))
 return result
baseline=measure('validation');best_state=copy.deepcopy(model.state_dict());best_epoch=0;best=-1.;history=[]
train_y=torch.tensor([r['y'] for r in sets['train']]);class_counts=torch.bincount(train_y,minlength=len(pairs)).float();sampling=class_counts[train_y].clamp_min(1).pow(-.5)
optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=.000045),dict(params=list(model.neck.parameters())+list(model.brand.parameters()),lr=.0003)],weight_decay=.015);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,24)
for epoch in range(24):
 model.train();ids=torch.multinomial(sampling,len(sets['train'])*2,replacement=True)
 for batch in ids.split(32):
  if len(batch)<2:continue
  xs=[]
  for idx in batch:
   r=sets['train'][int(idx)];im=Image.fromarray(r['rgb'].numpy());mask=r['mask'].numpy()
   if random.random()<.6:im=isolate(im,mask)
   choice=random.random()
   if choice<.18:im=im.crop((0,0,176,62))
   elif choice<.38:im=im.crop((0,0,176,97))
   elif choice<.50:im=im.crop((0,0,114,176))
   elif choice<.62:im=im.crop((62,0,176,176))
   elif choice<.72:im=im.crop((25,35,151,151))
   im=im.resize((160,160),Image.Resampling.BICUBIC)
   if random.random()<.5:im=ImageOps.mirror(im)
   im=jitter(im)
   if random.random()<.25:im=ImageOps.grayscale(im,num_output_channels=3)
   xs.append(normal(im))
  optimizer.zero_grad();logits=model(torch.stack(xs))[0];loss=torch.nn.functional.cross_entropy(logits,train_y[batch],label_smoothing=.05);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
 scheduler.step();model.eval();m=measure('validation');score=sum(v['all']['top1']+.5*v['all']['top3'] for v in m.values())/3
 guard=m['whole']['legacy']['top1']>=baseline['whole']['legacy']['top1']-.02 and m['whole']['legacy']['top3']>=baseline['whole']['legacy']['top3']-.02
 if guard and score>best:best=score;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());torch.save(dict(old,state=best_state,labels=[m for b,m in pairs],brands=[b for b,m in pairs]),out/'vehicle_model.pt')
 record=dict(epoch=epoch+1,validation=m,score=score,legacy_clean_guard=guard);history.append(record);print(json.dumps(record),flush=True)
 (out/'progress.json').write_text(json.dumps(dict(best_epoch=best_epoch,history=history)),encoding='utf-8')
model.load_state_dict(best_state);model.eval();metrics={s:measure(s) for s in ['validation','test']}
info=dict(origin='project-trained-regional-model-head',source_model_sha256=sha,labels=pairs,counts=counts,epochs=24,best_epoch=best_epoch,baseline_validation=baseline,results=metrics,history=history,selection='Grouped validation only; maximize mean whole/upper55/upper35 top1+half top3, legacy clean top1 and top3 floors minus2pp. No test/city selection. Candidate only: not automatically promoted. Compare final references against v1.7 before promotion.',augmentation='Whole/upper35/upper55/left65/right65/centre details;60% isolated, flip, mild colour jitter,25% grayscale. No class colour restrictions.',seconds=time.perf_counter()-started,test_used_for_selection=False)
(out/'training.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print('FINAL',json.dumps({k:v for k,v in info.items() if k!='history'}),flush=True)
