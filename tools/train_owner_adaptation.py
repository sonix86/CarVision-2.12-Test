"""Finite owner-data adaptation: frozen own body features,12FC epochs; val only.

The 10 phone photos are one TRAIN group. Existing validation/test stay unchanged.
This is a new user-authorized experiment after the completed body experiment.
"""
import sys,json,copy,random,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image,ImageOps
from torchvision import transforms
from carvision.body_expert import BodyExpert,body_tensor
from carvision.badge_evidence import descriptor,BadgeEvidence
from carvision.segmentation import isolate
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'models/owner_candidate';out.mkdir(exist_ok=True)
torch.set_num_threads(3);torch.manual_seed(321021);random.seed(321021);started=time.time()
src=ROOT/'models/body_expert_candidate';ck=torch.load(src/'vehicle_model.pt',weights_only=True);expert=copy.deepcopy(ck['body_expert']);pairs=expert['pairs'];body=BodyExpert(expert,pairs);net=body.network;head=copy.deepcopy(net.fc);net.fc=torch.nn.Identity();net.eval()
data=torch.load(ROOT/'cache/body_expert_v20.pt',weights_only=True);train=[r for r in data['rows'] if r['split']=='train'];val=[r for r in data['rows'] if r['split']=='validation'];cache=torch.load(ROOT/'cache/body_expert_validation_v20.pt',weights_only=True)
@torch.inference_mode()
def features(xs):return torch.cat([net(torch.stack(xs[i:i+32])) for i in range(0,len(xs),32)])
print('Extract training features',flush=True)
x=features([body_tensor(Image.fromarray(r['body_rgb'].numpy())) for r in train]);y=torch.tensor([r['y'] for r in train])
vf={k:torch.cat([net(batch).detach() for batch in v['x'].split(32)]) for k,v in cache['validation'].items()};vy=torch.tensor([r['y'] for r in val]);chevy=torch.tensor([b=='Chevrolet' for b,m in pairs])
owner=ROOT/'data/own_oct21';labels=[json.loads(s) for s in (owner/'labels.jsonl').read_text(encoding='utf-8').splitlines()];ownx=[];owny=[];jitter=transforms.ColorJitter(.15,.1,.05,.01)
templates=[]
for r in labels:
 im=Image.open(owner/r['image']).convert('RGB')
 if r.get('badge_box'):
  templates.append(dict(brand=r['brand'],descriptor=torch.from_numpy(descriptor(im.crop(r['badge_box']))),source_image_sha256=hashlib.sha256((owner/r['image']).read_bytes()).hexdigest(),group=r['group']))
 if r['model'] is None:continue
 target=pairs.index([r['brand'],r['model']]);crop=im.crop(r['box'])
 for n in range(8):
  view=crop.copy() if n==0 else jitter(crop)
  if n%2:view=ImageOps.mirror(view)
  if n>=4:view.thumbnail((random.choice([64,96,128]),)*2)
  ownx.append(body_tensor(view));owny.append(target)
ox=features(ownx);oy=torch.tensor(owny);del net
@torch.no_grad()
def evaluate():
 result={};ps={}
 for kind,feat in vf.items():
  z=head(feat).reshape(-1,4,len(pairs));q=(z/expert['temperature']).softmax(2);q=q[:,:2].mean(1)*.75+q[:,2:].mean(1)*.25
  p=cache['validation'][kind]['old']*(1-expert['weight'])+q*expert['weight'];ps[kind]=p
  good=(p.argmax(1)==vy);top=(p.topk(3).indices==vy[:,None]).any(1);idx=chevy[vy]
  result[kind]=dict(top1=float(good.float().mean()),top3=float(top.float().mean()),chevrolet=float(good[idx].float().mean()))
 return result,ps
baseline,_=evaluate();bestscore=sum(r['top1']+r['chevrolet'] for r in baseline.values());best=None;history=[];optimizer=torch.optim.AdamW(head.parameters(),lr=.00015,weight_decay=.02)
for epoch in range(12):
 for ids in torch.randperm(len(x)).split(48):
  oi=torch.randint(len(ox),(16,));bx=torch.cat([x[ids],ox[oi]]);by=torch.cat([y[ids],oy[oi]])
  optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(bx),by,label_smoothing=.04);loss.backward();optimizer.step()
 metrics,_=evaluate();score=sum(r['top1']+r['chevrolet'] for r in metrics.values());guard=all(metrics[k]['top1']>=baseline[k]['top1']-.01 and metrics[k]['top3']>=baseline[k]['top3']-.01 and metrics[k]['chevrolet']>=baseline[k]['chevrolet']-.01 for k in baseline)
 if guard and score>bestscore+.003:bestscore=score;best=copy.deepcopy(head.state_dict());bestepoch=epoch+1
 history.append(dict(epoch=epoch+1,guard=guard,validation=metrics));print('OWNER EPOCH',epoch+1,'accepted improvement',best is not None,flush=True)
if best:
 expert['state']['fc.weight']=best['weight'];expert['state']['fc.bias']=best['bias'];torch.save(expert,out/'expert.pt')
# Badge acceptance is fitted ONLY to public validation make labels, no test.
badge=BadgeEvidence(dict(templates=templates,threshold=1.,enabled=False));scores=[]
for r in val:
 im=Image.fromarray(r['body_rgb'].numpy());s=badge.predict(im,r['body_mask'].numpy())
 if s:scores.append(dict(**s,truth=pairs[r['y']][0],image=r['image']))
gate=None
for threshold in [.65,.7,.75,.8,.85,.9,.95]:
 accepted=[r for r in scores if r['similarity']>=threshold and r['margin']>=.08]
 # No false badge decisions on validation; at least3positive independent
 # source photos. Counts are explicitly small and are not universal precision.
 if len(accepted)>=3 and all(r['brand']==r['truth'] for r in accepted):gate=dict(threshold=threshold,count=len(accepted));break
badgeck=dict(templates=templates,enabled=gate is not None,threshold=gate['threshold'] if gate else 1.,origin='Own manually annotated phone badge crops; local normalized contrast template comparison',validation_gate=gate)
torch.save(badgeck,out/'badge.pt')
report=dict(selected=best is not None,best_epoch=bestepoch if best else 0,epochs=12,baseline_validation=baseline,history=history,owner_photos=10,owner_labelled_cars=9,owner_model_crops=8,owner_groups=1,owner_split='train only',architecture='Frozen project-trained ResNet18 features; retrained32-way FC;75%public rehearsal and25%owner crops per batch',badge=dict(enabled=gate is not None,gate=gate,validation_scores=scores),test_used_for_selection=False,seconds=time.time()-started,limitations='Sameouting owner cars cannot support independent own-data accuracy. Existing public validation does not include Cobalt. All prior user-city diagnostics are development images, never independent test.')
(out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('COMPLETE',report['selected'],'badge',report['badge']['enabled'],flush=True)
