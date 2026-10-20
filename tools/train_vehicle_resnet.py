"""Frozen general ImageNet ResNet features + our model head, validation-selected fusion."""
import sys,json,copy,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torchvision import models,transforms
from PIL import ImageOps
from carvision.core import ROOT,read_image,make_preprocess,Classifier
from carvision.vehicle_model import VehicleModelClassifier
started=time.perf_counter();torch.set_num_threads(3);torch.manual_seed(321015);random.seed(321015);torch.hub.set_dir(str(ROOT/'cache/hub'))
old=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);labels=list(zip(old['brands'],old['labels']))
rows=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
import re
def name(r):return '2101' if r['vehicle_category']=='VAZ-2101' else re.sub(r' \(.*?\)','',r['vehicle_category']).removeprefix(r['brand']+' ').removesuffix(' sedan')
net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval();net.fc=nn.Identity();pre=make_preprocess(192,16)
aug=transforms.Compose([transforms.RandomResizedCrop(192,scale=(.7,1.),ratio=(.85,1.2)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.1,.1,.08),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
data={};oldnet=Classifier(len(labels)).eval();oldnet.load_state_dict(old['state']);oldpre=make_preprocess(160,16)
with torch.inference_mode():
 for split in ['train','validation','test']:
  xs=[];ys=[];oldpred=[]
  for r in rows:
   if r['split']!=split:continue
   im=read_image(ROOT/'data/regional'/r['image']).crop(r['box']);views=[aug(im) for _ in range(4)] if split=='train' else [pre(im),pre(ImageOps.mirror(im))]
   xs.append(net(torch.stack(views)));ys.extend([labels.index((r['brand'],name(r)))]*len(views))
   if split!='train':oldpred.append((oldnet(torch.stack([oldpre(im),oldpre(ImageOps.mirror(im))]))[0]/old['temperature']).softmax(1).mean(0))
  data[split]=dict(x=torch.cat(xs),y=torch.tensor(ys),old=torch.stack(oldpred) if oldpred else None);print('FEATURES',split,len(ys),flush=True)
mean=data['train']['x'].mean(0);std=data['train']['x'].std(0).clamp_min(.05)
for d in data.values():d['x']=(d['x']-mean)/std
head=nn.Linear(512,len(labels));opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.2);counts=torch.bincount(data['train']['y'],minlength=len(labels)).float();cw=counts.pow(-.5);cw/=cw.mean();best=-1
for epoch in range(200):
 for ids in torch.randperm(len(data['train']['y'])).split(128):
  opt.zero_grad();loss=nn.functional.cross_entropy(head(data['train']['x'][ids]),data['train']['y'][ids],weight=cw,label_smoothing=.1);loss.backward();opt.step()
 with torch.no_grad():p=head(data['validation']['x']).softmax(1).reshape(-1,2,len(labels)).mean(1);y=data['validation']['y'][::2];a=float((p.argmax(1)==y).float().mean());score=a-.01*float(nn.functional.nll_loss(p.log(),y))
 if score>best:best=score;state=copy.deepcopy(head.state_dict());best_epoch=epoch+1
head.load_state_dict(state);head.eval()
with torch.no_grad():v=head(data['validation']['x']);truth=data['validation']['y']
temperature=min([.5,.75,1.,1.5,2.],key=lambda t:float(nn.functional.cross_entropy(v/t,truth)));p=(v/temperature).softmax(1).reshape(-1,2,len(labels)).mean(1);y=truth[::2];best_score=-1
for w in [0.,.25,.5,.75,1.]:
 q=p*w+data['validation']['old']*(1-w);accuracy=float((q.argmax(1)==y).float().mean());score=accuracy-.01*float(nn.functional.nll_loss(q.log(),y))
 if score>best_score:best_score=score;weight=w;selected=q
threshold=.65
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
 accept=selected.max(1).values>=t
 if int(accept.sum())>=15 and float((selected.argmax(1)[accept]==y[accept]).float().mean())>=.9:threshold=t;break
candidate=dict(origin=old['origin'],architecture='generic-resnet18-own-model-head',labels=old['labels'],brands=old['brands'],backbone=net.state_dict(),mean=mean,std=std,state=state,temperature=temperature,threshold=threshold)
with torch.no_grad():t=(head(data['test']['x'])/temperature).softmax(1).reshape(-1,2,len(labels)).mean(1);q=t*weight+data['test']['old']*(1-weight);y=data['test']['y'][::2];accept=q.max(1).values>=threshold
info=dict(best_epoch=best_epoch,validation_accuracy=float((selected.argmax(1)==truth[::2]).float().mean()),resnet_weight=weight,temperature=temperature,threshold=threshold,test_accuracy=float((q.argmax(1)==y).float().mean()),test_count=len(y),test_coverage=float(accept.float().mean()),test_accepted_accuracy=float((q.argmax(1)[accept]==y[accept]).float().mean()),seconds=time.perf_counter()-started,initialization='Official torchvision ResNet18 IMAGENET1K_V1 only; original ImageNet head removed, our model head trained from scratch on regional training groups',head_optimizer='AdamW 0.003 wd .2; 200 epochs; weighted CE smoothing .1; features frozen',selection='Grouped validation only for head epoch, temperature, mixture and rejection. Six city images excluded.',test_used_for_selection=False)
out=ROOT/'models/vehicle_resnet_candidate';out.mkdir(exist_ok=True);torch.save(candidate,out/'vehicle_model.pt');(out/'training.json').write_text(json.dumps(info,indent=2));print('FINAL',json.dumps(info),flush=True)
if weight>0 and info['validation_accuracy']>=json.loads((ROOT/'models/vehicle_model_training.json').read_text(encoding='utf-8'))['validation_accuracy']:
 backup=ROOT/'models/vehicle_mobilenet_candidate';backup.mkdir(exist_ok=True);torch.save(old,backup/'vehicle_model.pt');original=json.loads((ROOT/'models/vehicle_model_training.json').read_text(encoding='utf-8'));(backup/'training.json').write_text(json.dumps(original,ensure_ascii=False,indent=2),encoding='utf-8')
 final=candidate if weight==1 else dict(origin=old['origin'],architecture='own-model-evidence-ensemble',labels=old['labels'],brands=old['brands'],components=[old,candidate],weights=[1-weight,weight],temperature=1.,threshold=threshold)
 torch.save(final,ROOT/'models/vehicle_model.pt');original.update(architecture='Validation-selected fusion of our fine-tuned MobileNet model network and independently trained model head on frozen general ImageNet ResNet18',resnet_experiment=info,**{k:info[k] for k in ['validation_accuracy','test_accuracy','test_count','test_coverage','test_accepted_accuracy','threshold']});(ROOT/'models/vehicle_model_training.json').write_text(json.dumps(original,ensure_ascii=False,indent=2),encoding='utf-8');print('PROMOTED',weight,flush=True)
