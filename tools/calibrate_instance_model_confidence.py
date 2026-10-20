"""Calibrate the newly fine-tuned model head on grouped development only."""
import sys,json,hashlib,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageOps
from carvision.core import ROOT,Classifier,make_preprocess
from carvision.segmentation import isolate
torch.set_num_threads(3);ck=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);network=Classifier(len(ck['labels'])).eval();network.load_state_dict(ck['state']);rows=torch.load(ROOT/'cache/regional_instance_training.pt',weights_only=True)['rows'];train=[r for r in rows if r['split']=='train'];pre=make_preprocess(160,16);sets={}
@torch.inference_mode()
def logits(images):return torch.cat([network(v)[0] for v in torch.stack(images).split(48)]).reshape(-1,4,len(ck['labels']))
for split in ['validation','test']:
 clean=[];partial=[];selected=[r for r in rows if r['split']==split]
 for i,r in enumerate(selected):
  image=Image.fromarray(r['rgb'].numpy());mask=r['mask'].numpy();donor=Image.fromarray(train[int(hashlib.sha256(r['image'].encode()).hexdigest()[:8],16)%len(train)]['rgb'].numpy());oc=image.copy();block=(0,70,79,176) if i%2 else (97,70,176,176);oc.paste(donor.crop(block),block);om=mask.copy();om[block[1]:block[3],block[0]:block[2]]=False
  for target,im,m in [(clean,image,mask),(partial,oc,om)]:
   focus=isolate(im,m);target.extend([pre(im),pre(ImageOps.mirror(im)),pre(focus),pre(ImageOps.mirror(focus))])
 sets[split]=dict(clean=logits(clean),partial=logits(partial),y=torch.tensor([r['y'] for r in selected]))
def probability(logits,temp):
 p=(logits/temp).softmax(2);return p[:,:2].mean(1)*.75+p[:,2:].mean(1)*.25
truth=sets['validation']['y'].repeat(2)
def dev(temp):return torch.cat([probability(sets['validation'][c],temp) for c in ['clean','partial']])
temperature=min([.5,.75,1.,1.25,1.5,2.,3.],key=lambda t:float(torch.nn.functional.nll_loss(dev(t).clamp_min(1e-8).log(),truth)));vp=dev(temperature);threshold=.9
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
 accept=vp.max(1).values>=t
 if int(accept.sum())>=20 and float((vp.argmax(1)[accept]==truth[accept]).float().mean())>=.9:threshold=t;break
info=dict(temperature=temperature,threshold=threshold,selection='Combined clean/synthetic-occluded grouped validation logits only, minimum NLL temperature;90% accepted development accuracy target. No city or test calibration.',test_used_for_selection=False,metrics={})
for split in ['validation','test']:
 info['metrics'][split]={};y=sets[split]['y']
 for condition in ['clean','partial']:
  p=probability(sets[split][condition],temperature);accept=p.max(1).values>=threshold;info['metrics'][split][condition]=dict(count=len(y),top1=float((p.argmax(1)==y).float().mean()),top3=float((p.topk(3).indices==y[:,None]).any(1).float().mean()),coverage=float(accept.float().mean()),accuracy_when_accepted=float((p.argmax(1)[accept]==y[accept]).float().mean()) if accept.any() else None)
ck['temperature']=temperature;ck['threshold']=threshold;tmp=ROOT/'models/vehicle_model.calibrated.tmp';torch.save(ck,tmp);tmp.replace(ROOT/'models/vehicle_model.pt');(ROOT/'models/instance_vehicle_calibration.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
