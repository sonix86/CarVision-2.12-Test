"""Improve our model-name network with visible masks and synthetic neighbour occlusions."""
import sys,json,random,copy,time,re,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image,ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF
from carvision.core import ROOT,Classifier,read_image,make_preprocess
from carvision.segmentation import InstanceSegmenter,tight_box,isolate
started=time.perf_counter();torch.set_num_threads(3);torch.manual_seed(321016);random.seed(321016)
old=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);labels=list(zip(old['brands'],old['labels']));rows=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
def name(r):return '2101' if r['vehicle_category']=='VAZ-2101' else re.sub(r' \(.*?\)','',r['vehicle_category']).removeprefix(r['brand']+' ').removesuffix(' sedan')
cache=ROOT/'cache/regional_instance_training.pt';prepared=[]
if cache.exists():prepared=torch.load(cache,weights_only=True)['rows'];assert all(a['image']==b['image'] for a,b in zip(prepared,rows))
segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True)
for r in rows[len(prepared):]:
 im=read_image(ROOT/'data/regional'/r['image']).crop(r['box']);im.thumbnail((384,384));d=segmenter.raw(im);ids=[i for i in range(len(d['scores'])) if int(d['labels'][i]) in (3,8)];m=d['masks'][max(ids,key=lambda i:int((d['masks'][i,0]>.5).sum())),0].numpy()>.5 if ids else np.ones((im.height,im.width),dtype=bool)
 if int(m.sum())<20:m=np.ones((im.height,im.width),dtype=bool)
 b=tight_box(m);im=im.crop(b).resize((176,176),Image.Resampling.BICUBIC);m=Image.fromarray(m[b[1]:b[3],b[0]:b[2]]).resize((176,176),Image.Resampling.NEAREST)
 prepared.append(dict(image=r['image'],split=r['split'],y=labels.index((r['brand'],name(r))),rgb=torch.from_numpy(np.asarray(im).copy()),mask=torch.from_numpy(np.asarray(m).copy())))
 if len(prepared)%25==0 or len(prepared)==len(rows):torch.save(dict(rows=prepared),cache);print('REGIONAL MASKS',len(prepared),'/',len(rows),flush=True)
del segmenter
sets={s:[r for r in prepared if r['split']==s] for s in ['train','validation','test']};pre=make_preprocess(160,16);normal=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);jitter=transforms.ColorJitter(.15,.15,.08)
def photo(r):return Image.fromarray(r['rgb'].numpy()),r['mask'].numpy()
def contaminate(im,mask,donor,side):
 block=(0,70,79,176) if side else (97,70,176,176);im=im.copy();im.paste(donor.crop(block),block);mask=mask.copy();mask[block[1]:block[3],block[0]:block[2]]=False;return im,mask
validation={}
for split in ['validation','test']:
 clean=[];partial=[]
 for i,r in enumerate(sets[split]):
  im,m=photo(r);donor,_=photo(sets['train'][int(hashlib.sha256(r['image'].encode()).hexdigest()[:8],16)%len(sets['train'])]);oc,om=contaminate(im,m,donor,i%2)
  for target,image,mask in [(clean,im,m),(partial,oc,om)]:
   target.append(torch.stack([pre(image),pre(ImageOps.mirror(image)),pre(isolate(image,mask)),pre(ImageOps.mirror(isolate(image,mask)))]))
 validation[split]=dict(clean=torch.cat(clean),partial=torch.cat(partial),y=torch.tensor([r['y'] for r in sets[split]]))
model=Classifier(len(labels));model.load_state_dict(old['state']);model.eval();mix=.25
@torch.inference_mode()
def probabilities(x):
 logits=torch.cat([model(v)[0] for v in x.split(48)]).reshape(-1,4,len(labels));p=(logits/old.get('temperature',1.)).softmax(2);return p[:,:2].mean(1)*(1-mix)+p[:,2:].mean(1)*mix
baseline={}
for split in ['validation','test']:
 y=validation[split]['y'];baseline[split]={c:float((probabilities(validation[split][c]).argmax(1)==y).float().mean()) for c in ['clean','partial']}
counts=torch.bincount(torch.tensor([r['y'] for r in sets['train']]),minlength=len(labels)).float();sampling=counts[torch.tensor([r['y'] for r in sets['train']])].pow(-.5)
optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=.00006),dict(params=list(model.neck.parameters())+list(model.brand.parameters()),lr=.0003)],weight_decay=.015);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,24);best=(baseline['validation']['clean']+baseline['validation']['partial'])/2;state=copy.deepcopy(old['state']);best_epoch=0;history=[]
for epoch in range(24):
 model.train();ids=torch.multinomial(sampling,len(sets['train'])*2,replacement=True)
 for batch in ids.split(32):
  if len(batch)<2:continue
  xs=[];ys=[]
  for i in batch:
   r=sets['train'][int(i)];im,m=photo(r)
   if random.random()<.5:
    donor,_=photo(random.choice(sets['train']));im,m=contaminate(im,m,donor,random.randrange(2))
   if random.random()<.6:im=isolate(im,m)
   im=TF.resized_crop(im,*transforms.RandomResizedCrop.get_params(im,(.65,1.),(.85,1.2)),(160,160),interpolation=transforms.InterpolationMode.BICUBIC)
   if random.random()<.5:im=ImageOps.mirror(im)
   xs.append(normal(jitter(im)));ys.append(r['y'])
  optimizer.zero_grad();logits=model(torch.stack(xs))[0];loss=torch.nn.functional.cross_entropy(logits,torch.tensor(ys),label_smoothing=.05);loss.backward();optimizer.step()
 scheduler.step();model.eval();y=validation['validation']['y'];clean=float((probabilities(validation['validation']['clean']).argmax(1)==y).float().mean());partial=float((probabilities(validation['validation']['partial']).argmax(1)==y).float().mean());score=(clean+partial)/2
 if clean>=baseline['validation']['clean']-.02 and score>best:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
 history.append(dict(epoch=epoch+1,validation_clean=clean,validation_synthetic_occlusion=partial));print(json.dumps(history[-1]),flush=True)
model.load_state_dict(state);model.eval();result={}
for split in ['validation','test']:
 y=validation[split]['y'];result[split]={}
 for c in ['clean','partial']:
  prob=probabilities(validation[split][c]);result[split][c]=dict(accuracy=float((prob.argmax(1)==y).float().mean()),top3_recall=float((prob.topk(3).indices==y[:,None]).any(1).float().mean()))
info=dict(origin=old['origin'],architecture='Own MobileNet model network, fine-tuned on source-labelled visible-instance/whole-crop views and synthetic neighbour occlusion',labels=old['labels'],brands=old['brands'],train=len(sets['train']),validation=len(sets['validation']),test=len(sets['test']),epochs=24,best_epoch=best_epoch,baseline=baseline,results=result,history=history,seconds=time.perf_counter()-started,optimizer='AdamW feature .00006 / head .0003, wd .015, cosine24, smoothing.05, batch32',augmentation='60% foreground-only views,50% lower-corner synthetic neighbour occlusion, crop65-100%, flip, mild colour jitter',selection='Existing grouped validation only; maximize mean clean/occluded, clean floor baseline minus2pts. Test results computed after checkpoint selection. Six city examples and own photos excluded.',test_used_for_selection=False,model_mask_weight=mix)
out=ROOT/'models/instance_vehicle_candidate';out.mkdir(exist_ok=True);torch.save(dict(old,state=state),out/'vehicle_model.pt');(out/'training.json').write_text(json.dumps(info,indent=2));print('FINAL',json.dumps({k:v for k,v in info.items() if k!='history'}),flush=True)
if best_epoch>0:
 backup=ROOT/'models/backups/v14';backup.mkdir(parents=True,exist_ok=True);torch.save(old,backup/'vehicle_model.pt');torch.save(dict(old,state=state),ROOT/'models/vehicle_model.pt');(ROOT/'models/instance_vehicle_training.json').write_text(json.dumps(info,indent=2));print('PROMOTED',flush=True)
