"""One finite abstention-only retrieval fallback, validation selected, no fit.

Preserve every accepted v25 result. Only fill abstentions with a separately
calibrated reference mixture. No user diagnostic or test decisions select it.
"""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead,embedding_features
from carvision.visual_retrieval import retrieval_probabilities
from tools.cache_semantic_v25 import token
torch.set_num_threads(3)
out=ROOT/'models/fallback_v253';out.mkdir(exist_ok=True)
policy=dict(k=8,alpha=.5,beta=50.,selection='Fixed best prior validation-scoring reference mixture; only fills v25 abstentions',minimum_new_groups=3,minimum_new_precision=.95,minimum_new_accepted=20,minimum_overall_precision=.95,gradient_fits=0,test_used_for_selection=False,user_diagnostics_used_for_selection=False,no_more_parameter_searches=True)
with (out/'policy.json').open('x',encoding='utf-8') as f:json.dump(policy,f,indent=2)
path=ROOT/'models/semantic_v25/semantic_make.pt';ck=torch.load(path,weights_only=True)
data=torch.load(ROOT/'cache/multimodal_v24_data.pt',weights_only=True);brands=data['brands'];train=[r for r in data['rows'] if r['split']=='train'];rows=[r for r in data['rows'] if r['split']=='validation']
cache=torch.load(ROOT/'cache/semantic_v25/features.pt',weights_only=True)['items'];keys=torch.stack([torch.stack([embedding_features(cache[token(r,k)],False) for r in train]) for k in ('whole','phone')]);labels=torch.tensor([r['y'] for r in train]);names=sorted({r['group'] for r in train});groups=torch.tensor([names.index(r['group']) for r in train]);truth=torch.tensor([r['y'] for r in rows]*2);groupnames=sorted({r['group'] for r in rows});valgroups=torch.tensor([groupnames.index(r['group']) for r in rows]*2)
q=torch.cat([torch.stack([embedding_features(cache[token(r,k)],False) for r in rows]) for k in ('whole','phone')]);geometry=[]
for k in ('whole','phone'):
    for r in rows:
        h,w=r['body_rgb'].shape[:2];s=min(1,96/max(h,w)) if k=='phone' else 1;geometry.append(min(round(w*s),round(h*s))>=40)
geometry=torch.tensor(geometry);head=CosineMakeHead(torch.zeros((len(brands),512))).eval();head.load_state_dict(ck['head'])
with torch.inference_mode():
    p=(head(q)/ck['temperature']).softmax(1);v,ids=p.topk(2,dim=1);gates=torch.tensor([ck['thresholds'][brands[int(i)]] for i in ids[:,0]]);old=(gates<1)&(v[:,0]>=gates)&((v[:,0]-v[:,1])>=ck['margin'])&geometry
    mixed=p*(1-policy['alpha'])+retrieval_probabilities(q,keys,labels,groups,len(brands),policy['k'],policy['beta'])*policy['alpha'];mv,mi=mixed.topk(2,dim=1);thresholds={};added=torch.zeros(len(q),dtype=torch.bool)
    for j,b in enumerate(brands):
        options=[]
        for t in (.5,.6,.7,.8,.9,.95,.98):
            accept=(~old)&(mi[:,0]==j)&(mv[:,0]>=t)&((mv[:,0]-mv[:,1])>=ck['margin'])&geometry
            if len(valgroups[accept].unique())>=policy['minimum_new_groups'] and float((truth[accept]==j).float().mean())>=policy['minimum_new_precision']:options.append((int(accept.sum()),t))
        _,t=max(options,default=(0,1.));thresholds[b]=t
        if t<1:added|=(~old)&(mi[:,0]==j)&(mv[:,0]>=t)&((mv[:,0]-mv[:,1])>=ck['margin'])&geometry
    chosen=torch.where(old[:,None],p,mixed);pred=chosen.argmax(1);accepted=old|added
    metrics={}
    for n,slice_ in [('whole',slice(0,len(rows))),('phone',slice(len(rows),None))]:
        good=(pred==truth)[slice_];a=accepted[slice_];base=old[slice_];hist=torch.tensor([r['historical'] for r in rows]);metrics[n]=dict(top1=float(good.float().mean()),historical=float(good[hist].float().mean()),confirmed=int(a.sum()),correct_confirmed=int((good&a).sum()),confirmed_precision=float(good[a].float().mean()),baseline_confirmed=int(base.sum()))
    precision=float((pred[accepted]==truth[accepted]).float().mean());new_precision=float((pred[added]==truth[added]).float().mean()) if added.any() else 0
    checks=dict(new_coverage=int(added.sum())>=policy['minimum_new_accepted'],new_precision=new_precision>=policy['minimum_new_precision'],total_precision=precision>=policy['minimum_overall_precision'],accepted_preserved=bool(torch.equal(pred[old],ids[old,0])))
    report=dict(passed=all(checks.values()),checks=checks,metrics=metrics,additional_confirmed=int(added.sum()),additional_precision=new_precision,total_precision=precision,thresholds=thresholds,source_model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),policy=policy)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    if report['passed']:
        result=dict(ck,retrieval=dict(keys=keys,labels=labels,groups=groups,k=policy['k'],alpha=policy['alpha'],beta=policy['beta'],fallback_thresholds=thresholds,mode='abstention-only'),calibration_source_model_sha256=report['source_model_sha256'])
        torch.save(result,out/'semantic_make.pt')
    else:(out/'NO_CHANGE.txt').write_text('Finite fallback rejected; no further parameter search.',encoding='utf-8')
