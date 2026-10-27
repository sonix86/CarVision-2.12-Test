"""One finite reference-bank calibration; zero additional gradient training.

Only frozen TRAIN references and VALIDATION decisions are used. Four mixtures,
then stop. User phone-error images do not select the mixture or thresholds.
"""
import hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead,embedding_features
from carvision.visual_retrieval import retrieval_probabilities
from tools.cache_semantic_v25 import token
torch.set_num_threads(3);out=ROOT/'models/retrieval_v252';out.mkdir(exist_ok=True)
with (out/'once.json').open('x') as f:json.dump(dict(settings=[dict(k=k,alpha=a) for k in (3,8) for a in (.5,.75)],beta=50,gradient_training_rounds=0,selection='validation only',no_more_searches=True),f)
path=ROOT/'models/semantic_v25/semantic_make.pt';source_sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,weights_only=True);assert not ck['two_views'];data=torch.load(ROOT/'cache/multimodal_v24_data.pt',weights_only=True);brands=data['brands'];train=[r for r in data['rows'] if r['split']=='train'];val=[r for r in data['rows'] if r['split']=='validation'];cache=torch.load(ROOT/'cache/semantic_v25/features.pt',weights_only=True)['items']
keys=torch.stack([torch.stack([embedding_features(cache[token(r,kind)],False) for r in train]) for kind in ('whole','phone')]);bank_labels=torch.tensor([r['y'] for r in train]);names=sorted({r['group'] for r in train});bank_groups=torch.tensor([names.index(r['group']) for r in train]);y=torch.tensor([r['y'] for r in val]);historical=torch.tensor([r['historical'] for r in val]);queries={k:torch.stack([embedding_features(cache[token(r,k)],False) for r in val]) for k in ('whole','phone')}
head=CosineMakeHead(torch.zeros((len(brands),512))).eval();head.load_state_dict(ck['head']);baseline=json.loads((ROOT/'models/semantic_v25/training.json').read_text(encoding='utf-8'));started=time.time();candidates=[];best=None
group_names=sorted({r['group'] for r in val});groups=torch.tensor([group_names.index(r['group']) for r in val]*2);geometry=[]
for kind in ('whole','phone'):
    for r in val:
        h,w=r['body_rgb'].shape[:2];s=min(1,96/max(w,h)) if kind=='phone' else 1;geometry.append(min(round(w*s),round(h*s))>=40)
geometry=torch.tensor(geometry);labels=y.repeat(2)
with torch.inference_mode():
    neural={k:(head(q)/ck['temperature']).softmax(1) for k,q in queries.items()}
    for k in (3,8):
        nearest={kind:retrieval_probabilities(q,keys,bank_labels,bank_groups,len(brands),k) for kind,q in queries.items()}
        for alpha in (.5,.75):
            probs={kind:neural[kind]*(1-alpha)+nearest[kind]*alpha for kind in neural};metrics={}
            for kind,p in probs.items():
                good=p.argmax(1)==y;metrics[kind]=dict(top1=float(good.float().mean()),historical=float(good[historical].float().mean()),macro=float(torch.tensor([float(good[y==j].float().mean()) for j in y.unique()]).mean()))
            good=probs['whole'].argmax(1)==y;priority_ids=[brands.index(b) for b in baseline['policy']['priority_makes']];pm=float(torch.tensor([float(good[y==j].float().mean()) for j in priority_ids]).mean());score=.5*metrics['whole']['macro']+.2*metrics['phone']['macro']+.3*pm
            guard=all(metrics[kind]['historical']>=baseline['selected_metrics'][kind]['historical']-.005 for kind in metrics) and metrics['whole']['top1']>=baseline['selected_metrics']['whole']['top1']
            p=torch.cat([probs[kind] for kind in ('whole','phone')]);values,ids=p.topk(2,dim=1);thresholds={};accepted=torch.zeros(len(p),dtype=torch.bool)
            for j,b in enumerate(brands):
                options=[]
                for gate in (.5,.6,.7,.8,.9,.95,.98):
                    eligible=(ids[:,0]==j)&(values[:,0]>=gate)&((values[:,0]-values[:,1])>=.06)&geometry
                    if len(groups[eligible].unique())>=3 and float((labels[eligible]==j).float().mean())>=.9:options.append((int(eligible.sum()),gate))
                _,gate=max(options,default=(0,1.));thresholds[b]=gate
                if gate<1:accepted|=(ids[:,0]==j)&(values[:,0]>=gate)&((values[:,0]-values[:,1])>=.06)&geometry
            priority_enabled={b:thresholds[b]<1 for b in baseline['policy']['priority_makes']};precision=float((ids[accepted,0]==labels[accepted]).float().mean()) if accepted.any() else 0.
            eligible=guard and all(priority_enabled.values()) and int(accepted.sum())>=40 and precision>=.9
            record=dict(k=k,alpha=alpha,beta=50,metrics=metrics,priority_macro=pm,score=score,guard=guard,priority_enabled=priority_enabled,confirmed_views=int(accepted.sum()),confirmed_precision=precision,eligible=eligible);candidates.append(record)
            if eligible and (best is None or score>best['score']):best=dict(record,thresholds=thresholds)
passed=best is not None;report=dict(passed=passed,selected=best,candidates=candidates,source_model_sha256=source_sha,gradient_training_rounds=0,train_reference_images=len(train),train_reference_groups=len(names),validation_images=len(val),user_errors_used=False,test_used_for_selection=False,seconds=time.time()-started)
(out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('REFERENCE CALIBRATION COMPLETE',json.dumps(report),flush=True)
if passed:
    result=dict(ck,thresholds=best['thresholds'],retrieval=dict(keys=keys,labels=bank_labels,groups=bank_groups,k=best['k'],alpha=best['alpha'],beta=50.),calibration_source_model_sha256=source_sha)
    torch.save(result,out/'semantic_make.pt')
else:(out/'NO_CHANGE.txt').write_text('Four predeclared reference mixtures rejected. No more fitting or parameter searches in this phase. Do not replace the working model.',encoding='utf-8')
