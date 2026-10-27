"""One heldout evaluation after the bounded DINO validation decision is frozen."""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cache_semantic_v25 import token
from tools.ev26_helpers import geometry,references,route,metrics
from carvision.semantic_make import embedding_features,load_make_head,CosineMakeHead

torch.set_num_threads(3);out=ROOT/'models/dino_v28';policy=json.loads((out/'policy.json').read_text());assert json.loads((out/'validation.json').read_text())['passed']
with (out/'evaluation_once.json').open('x') as f:json.dump(dict(selection_complete=True),f)
d=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True);rows=[r for r in d['rows'] if r['split']=='test'];brands=d['brands'];valid=geometry(rows)
df=torch.load(ROOT/'cache/dino_v28/features.pt',weights_only=True)['items'];cf=torch.load(ROOT/'cache/ev_views_v26_features.pt',weights_only=True)['items']
dx=torch.cat([torch.stack([df[token(r,k)] for r in rows]) for k in ('whole','phone')]);cx=torch.cat([torch.stack([embedding_features(cf[token(r,k)],False) for r in rows]) for k in ('whole','phone')]);candidate=torch.load(out/'dino_make.pt',weights_only=True);old=torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True);predictions={};results={}
with torch.inference_mode():
    for name,ck in [('baseline',old),('candidate',candidate)]:
        if name=='baseline':x=cx;head=load_make_head(ck)
        else:
            x=dx if ck['kind']=='dino' else torch.nn.functional.normalize(torch.cat([dx,cx],1),dim=1)
            head=CosineMakeHead(torch.zeros(len(brands),x.shape[1]));head.load_state_dict(ck['head']);head.eval()
        p=(head(x)/ck['temperature']).softmax(1);ref=references(x,ck['retrieval'],len(brands));p,a=route(p,ref,ck['thresholds'],ck['retrieval']['fallback_thresholds'],brands,valid);predictions[name]=(p,a);results[name]=metrics(p,a,rows,brands,policy['priority_makes'])
a=results['candidate'];b=results['baseline'];checks=dict(whole=a['whole']['top1']>=b['whole']['top1'],phone=a['phone']['top1']>=b['phone']['top1'],historical=all(a[k]['historical']>=b[k]['historical']-.01 for k in a),confirmed_precision_both=all(a[k]['precision']>=max(.95,b[k]['precision']-.01) for k in a),correct_confirmed_both=all(a[k]['correct_confirmed']>=b[k]['correct_confirmed'] for k in a),priority=a['whole']['priority_macro']>=b['whole']['priority_macro'])
records=[]
for i,r in enumerate(rows):
    rec=dict(image=r['image'],group=r['group'],truth=r['brand'],historical=r['historical'])
    for j,k in enumerate(('whole','phone')):
        rec[k]={}
        for name,(p,accepted) in predictions.items():
            index=j*len(rows)+i;rec[k][name]=dict(candidate=brands[int(p[index].argmax())],accepted=bool(accepted[index]),confidence=float(p[index].max()))
    records.append(rec)
report=dict(passed=all(checks.values()),checks=checks,results=results,records=records,candidate_sha256=hashlib.sha256((out/'dino_make.pt').read_bytes()).hexdigest(),limitations='Same reused public test, not an independent Uzbekistan phone exam. Seven known owner errors never used for fitting or validation selection.')
(out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],checks=checks,results=results)),flush=True)
