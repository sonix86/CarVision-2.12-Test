"""Freeze validation selection, calibrate there, then evaluate heldout originals once."""
import sys,json,hashlib,time,collections
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image,ImageEnhance,ImageFilter
import numpy as np
from carvision.core import ROOT
from carvision.vehicle_model import VehicleModelClassifier
from carvision.visual_parts import VisualReferences
from carvision.segmentation import isolate

torch.set_num_threads(3);out=ROOT/'models/ev_patch_v22';started=time.perf_counter()
info=json.loads((out/'training.json').read_text(encoding='utf-8'))
audit=json.loads((out/'data_audit.json').read_text(encoding='utf-8'))
assert audit['passed'] and hashlib.sha256((ROOT/'cache/ev_patch_v22.pt').read_bytes()).hexdigest()==audit['dataset_sha256']
if not info['selected']:raise SystemExit('Validation rejected candidate: no patch promotion')
assert hashlib.sha256((ROOT/'models/vehicle_model.pt').read_bytes()).hexdigest()==info['source_model_sha256']
expert=torch.load(out/'expert.pt',weights_only=True);ck=torch.load(out/'base_expanded.pt',weights_only=True);ref=torch.load(out/'references_expanded.pt',weights_only=True)
v=torch.load(out/'validation_inputs.pt',weights_only=True);zs=torch.load(out/'selected_validation_logits.pt',weights_only=True)
ps=[]
for k,z in zs.items():
    q=(z/expert['temperature']).softmax(2);q=q[:,:2].mean(1)*.75+q[:,2:].mean(1)*.25
    ps.append(v['validation'][k]['prior']*(1-expert['weight'])+q*expert['weight'])
p=torch.cat(ps);y=v['y'].repeat(len(ps))
temperatures=[1.,1.5,2.,3.]
def calibrate(p,t):
    p=p.clamp_min(1e-9).pow(1/t);return p/p.sum(1,keepdim=True)
losses=[float(-calibrate(p,t)[torch.arange(len(y)),y].log().mean()) for t in temperatures];temperature=temperatures[int(np.argmin(losses))];p=calibrate(p,temperature)
conf,guess=p.max(1);gates=[]
for threshold in [.7,.75,.8,.85,.9,.95,.98]:
    idx=conf>=threshold
    if int(idx.sum())>=10 and float((guess[idx]==y[idx]).float().mean())>=.9:gates.append(dict(threshold=threshold,count=int(idx.sum()),precision=float((guess[idx]==y[idx]).float().mean())))
gate=max(gates,key=lambda r:r['count']) if gates else dict(threshold=1.,count=0,precision=None)
brands=sorted(set(ck['brands']));assignment=torch.tensor([brands.index(b) for b in ck['brands']]);mp=torch.zeros(len(p),len(brands));mp.scatter_add_(1,assignment[None].expand(len(p),-1),p)
mc,mg=mp.max(1);makegates=[]
for threshold in [.9,.925,.95,.975,.99]:
    idx=mc>=threshold
    if int(idx.sum())>=10 and float((mg[idx]==assignment[y[idx]]).float().mean())>=.95:makegates.append(dict(threshold=threshold,count=int(idx.sum()),precision=float((mg[idx]==assignment[y[idx]]).float().mean())))
makegate=max(makegates,key=lambda r:r['count']) if makegates else dict(threshold=1.,count=0,precision=None)
expert['make_threshold']=makegate['threshold'];ck['body_expert']=expert
torch.save(ck,out/'vehicle_model.pt');sha=hashlib.sha256((out/'vehicle_model.pt').read_bytes()).hexdigest()
ref.update(source_model_sha256=sha,probability_temperature=temperature,confidence_threshold=gate['threshold']);torch.save(ref,out/'visual_references.pt')
base=VehicleModelClassifier(torch.load(ROOT/'models/vehicle_model.pt',weights_only=True));base.references=VisualReferences(torch.load(ROOT/'models/visual_references.pt',weights_only=True))
candidate=VehicleModelClassifier(ck);candidate.references=VisualReferences(ref)
data=torch.load(ROOT/'cache/ev_patch_v22.pt',weights_only=True);pairs=data['pairs'];test=[r for r in data['rows'] if r['split']=='test'];records=[]
with torch.inference_mode():
    for i,r in enumerate(test):
        im=Image.fromarray(r['body_rgb'].numpy());focus=isolate(im,r['body_mask'].numpy())
        a=base.probabilities(im,None,None)*.75+base.probabilities(focus,None,None)*.25
        b=candidate.probabilities(im,None,None)*.75+candidate.probabilities(focus,None,None)*.25;b=candidate.references.calibrated(b)
        records.append(dict(image=r['image'],truth=r['y'],historical='source' not in r,baseline=int(a.argmax()),candidate=int(b.argmax()),baseline_top3=a.topk(3).indices.tolist(),candidate_top3=b.topk(3).indices.tolist(),confidence=float(b.max())))
        if i%50==0:print('TEST',i,len(test),flush=True)
ev=lambda j:pairs[j][0] in ['BYD','Geely','Zeekr','Weltmeister','Hongqi','XPeng'] or pairs[j][1] in ['Leaf','Ariya','bZ4X']
def measure(rs):
    result=dict(count=len(rs))
    for name in ['baseline','candidate']:
        result[name]=dict(top1=sum(r[name]==r['truth'] for r in rs)/len(rs) if rs else None,top3=sum(r['truth'] in r[name+'_top3'] for r in rs)/len(rs) if rs else None,false_captiva=sum(r[name]==2 and r['truth']!=2 for r in rs),false_byd=sum(pairs[r[name]][0]=='BYD' and pairs[r['truth']][0]!='BYD' for r in rs))
    return result
results={name:measure(rs) for name,rs in [('all',records),('historical',[r for r in records if r['historical']]),('new_photos',[r for r in records if not r['historical']]),('ev',[r for r in records if ev(r['truth'])]),('existing_ev',[r for r in records if ev(r['truth']) and r['truth']<len(base.labels)]),('added_families',[r for r in records if r['truth']>=len(base.labels)])]}
per_class={' '.join(pair):measure([r for r in records if r['truth']==j]) for j,pair in enumerate(pairs)}
report=dict(baseline_sha256=info['source_model_sha256'],candidate_sha256=sha,best_epoch=info['best_epoch'],results=results,per_class=per_class,calibration=dict(temperature=temperature,gate=gate,make_gate=makegate,split='grouped validation; whole and simulated phone views correlated'),records=records,test_used_for_selection=False,seconds=time.perf_counter()-started,limitations='New source-session heldouts plus historical heldouts; not an independent real phone field test. Public vehicle identity unknown. User night image excluded from training and selection. Model family names do not establish propulsion variant.')
(out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('COMPLETE',json.dumps(results),flush=True)
