"""One bounded EV expansion, locally trained from our own v1.8 network."""
import sys,json,time,hashlib,copy,random,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image,ImageOps
from torchvision import transforms
from carvision.core import ROOT,Classifier,make_preprocess
from carvision.segmentation import isolate
started=time.perf_counter();torch.set_num_threads(3);torch.manual_seed(321019);random.seed(321019)
parser=argparse.ArgumentParser();parser.add_argument('--existing-only',action='store_true');parser.add_argument('--base',default='models/vehicle_model.pt');args=parser.parse_args()
out=ROOT/('models/occlusion_parts_candidate' if args.existing_only else 'models/ev_parts_candidate');out.mkdir(exist_ok=True)
old=torch.load(ROOT/args.base,weights_only=True);old_pairs=list(zip(old['brands'],old['labels']))
if args.existing_only:
 data=torch.load(ROOT/'cache/expanded_parts_v18.pt',weights_only=True);expanded=[dict(r,legacy=True) for r in data['rows']];pairs=old_pairs;added=[]
 (out/'dataset.json').write_text(json.dumps(dict(pairs=pairs,new_rows=[],purpose='Existing24 body-fragment/occluder augmentation round; no new model names or data claimed')),encoding='utf-8')
else:
 data=torch.load(ROOT/'cache/ev_parts_v19.pt',weights_only=True);expanded=data['rows'];pairs=[tuple(p) for p in data['pairs']]
 if len(pairs)<=len(old_pairs):raise SystemExit('Insufficient licensed grouped data for expansion; preserve baseline')
 added=json.loads((out/'dataset.json').read_text(encoding='utf-8'))['new_rows']
sha=hashlib.sha256((ROOT/args.base).read_bytes()).hexdigest()
sets={s:[r for r in expanded if r['split']==s] for s in ['train','validation','test']};pre=make_preprocess(160,16)
counts={b+' '+m:{s:sum(r['y']==j for r in sets[s]) for s in sets} for j,(b,m) in enumerate(pairs)}
(out/'training_dataset.json').write_text(json.dumps(dict(pairs=pairs,counts=counts,new_rows=added,source_model_sha256=sha,split='Existing regional groups and Stanford public_split preserved; official Stanford test remains test. Original MAT model labels, no colour pseudo-labels used. Stanford vehicle identities unavailable.',own_photos=0),ensure_ascii=False,indent=2),encoding='utf-8')
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
teacher=Classifier(len(old_pairs)).eval();teacher.load_state_dict(old['state'])
for parameter in teacher.parameters():parameter.requires_grad_(False)
@torch.inference_mode()
def probs(x):
 logits=torch.cat([model(t)[0] for t in x.split(48)]).reshape(-1,4,len(pairs));p=(logits/old['temperature']).softmax(2);return p[:,:2].mean(1)*.75+p[:,2:].mean(1)*.25
ys={s:torch.tensor([r['y'] for r in sets[s]]) for s in sets};legacy={s:torch.tensor([r['legacy'] for r in sets[s]]) for s in sets}
def measure(split):
 result={}
 for kind,x in validation[split].items():
  p=probs(x);y=ys[split];result[kind]={}
  for scope,mask in [('all',torch.ones(len(y),dtype=torch.bool)),('legacy',legacy[split]),('new',~legacy[split])]:
   if not mask.any():result[kind][scope]=dict(count=0,top1=None,top3=None);continue
   result[kind][scope]=dict(count=int(mask.sum()),top1=float((p.argmax(1)[mask]==y[mask]).float().mean()),top3=float((p.topk(3).indices[mask]==y[mask,None]).any(1).float().mean()))
 return result
baseline=measure('validation');best_state=copy.deepcopy(model.state_dict());best_epoch=0;best=-1.;history=[]
train_y=torch.tensor([r['y'] for r in sets['train']]);class_counts=torch.bincount(train_y,minlength=len(pairs)).float();sampling=class_counts[train_y].clamp_min(1).pow(-.75)
optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=.000025),dict(params=list(model.neck.parameters())+list(model.brand.parameters()),lr=.0002)],weight_decay=.015);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,16)
for epoch in range(16):
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
   if random.random()<.25:im=ImageOps.grayscale(im).convert('RGB')
   if random.random()<.35:
    # Missing body/detail evidence, without inventing a model label or image.
    from PIL import ImageDraw
    xx=random.randrange(0,120);yy=random.randrange(0,120);ww=random.randrange(18,55);hh=random.randrange(18,75)
    ImageDraw.Draw(im).rectangle((xx,yy,xx+ww,yy+hh),fill=tuple(random.randrange(50,200) for _ in range(3)))
   xs.append(normal(im))
  optimizer.zero_grad();images=torch.stack(xs);logits=model(images)[0];loss=torch.nn.functional.cross_entropy(logits,train_y[batch],label_smoothing=.05)
  keep=torch.tensor([sets['train'][int(i)]['legacy'] for i in batch])
  if keep.any():
   with torch.no_grad():target=(teacher(images[keep])[0]/2).softmax(1)
   loss+=.5*4*torch.nn.functional.kl_div((logits[keep,:len(old_pairs)]/2).log_softmax(1),target,reduction='batchmean')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
 scheduler.step();model.eval();m=measure('validation');score=sum(v['all']['top1']+.5*v['all']['top3'] for v in m.values())/3
 guard=m['whole']['legacy']['top1']>=baseline['whole']['legacy']['top1']-.02 and m['whole']['legacy']['top3']>=baseline['whole']['legacy']['top3']-.02
 if guard and score>best:best=score;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());torch.save(dict(old,state=best_state,labels=[m for b,m in pairs],brands=[b for b,m in pairs]),out/'vehicle_model.pt')
 record=dict(epoch=epoch+1,validation=m,score=score,legacy_clean_guard=guard);history.append(record);print(json.dumps(record),flush=True)
 (out/'progress.json').write_text(json.dumps(dict(best_epoch=best_epoch,history=history)),encoding='utf-8')
model.load_state_dict(best_state);model.eval();metrics={s:measure(s) for s in ['validation','test']}
info=dict(origin='project-trained-regional-model-head',source_model_sha256=sha,labels=pairs,counts=counts,epochs=16,best_epoch=best_epoch,baseline_validation=baseline,results=metrics,history=history,selection='Grouped validation only; maximize mean whole/upper55/upper35 top1+half top3, old24 clean top1 and top3 floors minus2pp. No test/city selection. Candidate only: not automatically promoted. Compare final references against v1.8 before promotion.',augmentation='Whole/upper35/upper55/left65/right65/centre details;60% isolated, flip, mild colour jitter,25% grayscale,35% random occluders. Old24 knowledge distillation from our v1.8 network. No class colour restrictions.',seconds=time.perf_counter()-started,test_used_for_selection=False)
(out/'training.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print('FINAL',json.dumps({k:v for k,v in info.items() if k!='history'}),flush=True)

