"""Locked post-selection test comparison; never tunes on heldout images."""
import sys,json,hashlib,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image,ImageOps
from carvision.core import ROOT
from carvision.vehicle_model import VehicleModelClassifier
from carvision.visual_parts import VisualReferences
from carvision.segmentation import isolate
from carvision.body_expert import BodyExpert,body_tensor

torch.set_num_threads(3);start=time.perf_counter();out=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'models/body_expert_candidate')
info=json.loads((out/'training.json').read_text(encoding='utf-8'))
if not info['selected']:raise SystemExit('Validation rejected this candidate; preserve previous release')
source=ROOT/'models/vehicle_model.pt';sha=hashlib.sha256(source.read_bytes()).hexdigest()
if sha!=info['source_model_sha256']:raise RuntimeError('Baseline changed after training')
ck=torch.load(source,weights_only=True);ref=torch.load(ROOT/'models/visual_references.pt',weights_only=True)
base=VehicleModelClassifier(ck);base.references=VisualReferences(ref)
expert_ck=torch.load(out/'expert.pt',weights_only=True);pairs=list(zip(ck['brands'],ck['labels']));expert=BodyExpert(expert_ck,pairs)
data=torch.load(ROOT/'cache/body_expert_v20.pt',weights_only=True)
data_sha=hashlib.sha256((ROOT/'cache/body_expert_v20.pt').read_bytes()).hexdigest()
test=[r for r in data['rows'] if r['split']=='test'];y=torch.tensor([r['y'] for r in test]);chevy=torch.tensor([b=='Chevrolet' for b,m in pairs])

def images(row,kind):
    rgb=Image.fromarray(row['body_rgb'].numpy());mask=row['body_mask'].numpy();old=Image.fromarray(row['rgb'].numpy());oldmask=row['mask'].numpy()
    if kind=='upper55':
        end=max(1,round(rgb.height*.55));rgb=rgb.crop((0,0,rgb.width,end));mask=mask[:end]
        old=old.crop((0,0,176,97)).resize((176,176),Image.Resampling.BICUBIC);oldmask=np.asarray(Image.fromarray(oldmask[:97]).resize((176,176),Image.Resampling.NEAREST))
    elif kind=='small64':
        rgb.thumbnail((64,64));mask=np.asarray(Image.fromarray(mask).resize(rgb.size,Image.Resampling.NEAREST));old=rgb.resize((176,176),Image.Resampling.BICUBIC)
    return rgb,mask,old,oldmask

def metrics(p):
    result={}
    for name,idx in [('all',torch.ones(len(y),dtype=torch.bool)),('chevrolet',chevy[y]),('other',~chevy[y])]:
        good=p.argmax(1)==y;top3=(p.topk(3).indices==y[:,None]).any(1)
        result[name]=dict(count=int(idx.sum()),top1=float(good[idx].float().mean()),top3=float(top3[idx].float().mean()))
    result['per_class']={b+' '+m:dict(count=int((y==i).sum()),correct=int(((y==i)&(p.argmax(1)==y)).sum()),top3_correct=int(((y==i)&(p.topk(3).indices==y[:,None]).any(1)).sum())) for i,(b,m) in enumerate(pairs)}
    predicted=p.argmax(1)
    result['chevrolet_as_bmw']=sum(1 for truth,guess in zip(y.tolist(),predicted.tolist()) if pairs[truth][0]=='Chevrolet' and pairs[guess][0]=='BMW')
    result['chevrolet_wrong_candidates']={b+' '+m:sum(1 for truth,guess in zip(y.tolist(),predicted.tolist()) if pairs[truth][0]=='Chevrolet' and truth!=guess and guess==i) for i,(b,m) in enumerate(pairs) if any(pairs[t][0]=='Chevrolet' and t!=g and g==i for t,g in zip(y.tolist(),predicted.tolist()))}
    return result

results={};probabilities={}
with torch.inference_mode():
    for kind in ['whole','small64','upper55']:
        cached=ROOT/f'cache/{out.name}_test_{kind}_v20.pt'
        if cached.exists():
            stored=torch.load(cached,weights_only=True)
            if stored['source_sha']!=sha or stored.get('data_sha')!=data_sha or stored['expert_sha']!=hashlib.sha256((out/'expert.pt').read_bytes()).hexdigest():raise RuntimeError('Stale test cache')
            old=stored['old'];new=stored['new']
        else:
            old=[];new=[]
            for n,row in enumerate(test):
                rgb,mask,im,oldmask=images(row,kind)
                p=base.probabilities(im,None,None)*.75+base.probabilities(isolate(im,oldmask),None,None)*.25
                q=expert.probabilities(rgb)*.75+expert.probabilities(isolate(rgb,mask))*.25
                old.append(p);new.append(p*(1-expert.weight)+q*expert.weight)
                if n%100==0:print('TEST',kind,n,'/',len(test),flush=True)
            old=torch.stack(old);new=torch.stack(new)
            torch.save(dict(source_sha=sha,data_sha=data_sha,expert_sha=hashlib.sha256((out/'expert.pt').read_bytes()).hexdigest(),old=old,new=new),cached)
        results[kind]=dict(baseline=metrics(old),candidate=metrics(new));probabilities[kind]=dict(old=old,new=new)
        print(kind,json.dumps({k:{s:v for s,v in m.items() if s!='per_class'} for k,m in results[kind].items()}),flush=True)

# Confidence calibration uses only the original validation predictions.
v=torch.load(ROOT/'cache/body_expert_validation_v20.pt',weights_only=True);valrows=[r for r in data['rows'] if r['split']=='validation'];vy=torch.tensor([r['y'] for r in valrows]);vp=[]
if v.get('data_sha') is None:
    # The first corrected launcher may predate the cache-metadata addition.
    # Bind it only after reproducing EVERY input tensor from the current source
    # geometry, not by trusting a timestamp or silently accepting an old cache.
    assert v['images']==[r['image'] for r in valrows]
    for kind,values in v['validation'].items():
        for i,row in enumerate(valrows):
            rgb,mask,_,_=images(row,kind);focused=isolate(rgb,mask)
            expected=torch.stack([body_tensor(view) for im in [rgb,focused] for view in [im,ImageOps.mirror(im)]])
            assert torch.equal(expected,values['x'][4*i:4*i+4]),'Cached geometry differs: '+row['image']
    v['data_sha']=data_sha;torch.save(v,ROOT/'cache/body_expert_validation_v20.pt')
assert v['data_sha']==data_sha,'Calibration geometry differs from training geometry'
with torch.inference_mode():
    for kind,values in v['validation'].items():
        z=torch.cat([expert.network(x) for x in values['x'].split(32)]).reshape(-1,4,len(pairs));q=(z/expert.temperature).softmax(2);q=q[:,:2].mean(1)*.75+q[:,2:].mean(1)*.25
        vp.append(values['old']*(1-expert.weight)+q*expert.weight)
vp=torch.cat(vp);vy=vy.repeat(len(v['validation']))
temps=[.5,.75,1.,1.5,2.,3.];nll=[]
for temp in temps:
    p=vp.clamp_min(1e-9).pow(1/temp);p=p/p.sum(1,keepdim=True);nll.append(float(-p[torch.arange(len(vy)),vy].clamp_min(1e-9).log().mean()))
temperature=temps[int(np.argmin(nll))];p=vp.clamp_min(1e-9).pow(1/temperature);p=p/p.sum(1,keepdim=True);conf=p.max(1).values;correct=p.argmax(1)==vy
gates=[]
for threshold in [.7,.75,.8,.85,.9,.95,.98]:
    accepted=conf>=threshold;n=int(accepted.sum());precision=float(correct[accepted].float().mean()) if n else None
    if n>=10 and precision>=.9:gates.append(dict(threshold=threshold,count=n,precision=precision))
gate=max(gates,key=lambda r:r['count']) if gates else dict(threshold=1.,count=0,precision=None)
# Calibrate the make decision independently: several uncertain Chevrolet
# models can establish Chevrolet without confirming any particular model.
make_names=sorted({b for b,m in pairs});assignment=torch.tensor([make_names.index(b) for b,m in pairs])
make_probability=torch.zeros(len(p),len(make_names));make_probability.scatter_add_(1,assignment[None].expand(len(p),-1),p)
make_confidence,make_prediction=make_probability.max(1);make_correct=make_prediction==assignment[vy];make_gates=[]
for threshold in [.9,.925,.95,.975,.99,.995]:
    accepted=make_confidence>=threshold;n=int(accepted.sum());precision=float(make_correct[accepted].float().mean()) if n else None
    if n>=10 and precision>=.95:make_gates.append(dict(threshold=threshold,count=n,precision=precision))
make_gate=max(make_gates,key=lambda r:r['count']) if make_gates else dict(threshold=1.,count=0,precision=None)
expert_ck['make_threshold']=make_gate['threshold']
candidate=dict(ck,body_expert=expert_ck)
torch.save(candidate,out/'vehicle_model.pt');candidate_sha=hashlib.sha256((out/'vehicle_model.pt').read_bytes()).hexdigest()
newref=dict(ref,source_model_sha256=candidate_sha,probability_temperature=temperature,confidence_threshold=gate['threshold'])
torch.save(newref,out/'visual_references.pt')
report=dict(baseline_sha256=sha,candidate_sha256=candidate_sha,expert_sha256=hashlib.sha256((out/'expert.pt').read_bytes()).hexdigest(),best_epoch=info['best_epoch'],test_used_for_selection=False,results=results,confidence_calibration=dict(split='validation only; three correlated views per photo, not three independent samples',temperature=temperature,nll=dict(zip(map(str,temps),nll)),gate=gate,make_gate=make_gate,make_gate_scope='Overall represented32-family validation; not a precision guarantee on conflicting heads or unrepresented models'),seconds=time.perf_counter()-start,limitations='Chevrolet Cobalt has2oldtest photographs, Nexia1, Damas0. Most Chevrolet test images are other families. These counts cannot substantiate broad Cobalt/Nexia/Damas accuracy. City and new Flickr photos were not used for selection.')
(out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('COMPLETE',report['seconds'],flush=True)
