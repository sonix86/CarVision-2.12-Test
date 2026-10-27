"""Exactly one frozen-candidate evaluation, including newly collected source groups."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import embedding_features,load_make_head
from tools.cache_semantic_v25 import token
from tools.ev26_helpers import geometry,references,route
torch.set_num_threads(3);out=ROOT/'models/ev_views_v26';assert json.loads((out/'validation_gate.json').read_text())['passed']
with (out/'evaluation_once.json').open('x') as f:json.dump(dict(selection_complete=True),f)
data=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True);rows=[r for r in data['rows'] if r['split']=='test'];brands=data['brands'];features=torch.load(ROOT/'cache/ev_views_v26_features.pt',weights_only=True)['items'];q=torch.cat([torch.stack([embedding_features(features[token(r,k)],False) for r in rows]) for k in ('whole','phone')]);valid=geometry(rows);candidate=torch.load(out/'semantic_make.pt',weights_only=True);baseline=torch.load(ROOT/'models/fallback_v253/semantic_make.pt',weights_only=True);training=json.loads((out/'training.json').read_text());policy=training['policy'];predictions={}
with torch.inference_mode():
    for name,ck in [('baseline',baseline),('candidate',candidate)]:
        head=load_make_head(ck);p=(head(q)/ck['temperature']).softmax(1);ref=references(q,ck['retrieval'],len(brands));p,a=route(p,ref,ck['thresholds'],ck['retrieval']['fallback_thresholds'],brands,valid);predictions[name]=(p,a)
records=[]
for i,r in enumerate(rows):
    rec=dict(image=r['image'],truth=r['brand'],historical=r['historical'],new_source=r['image'].startswith('ev_views_v26/'),group=r['group'])
    for j,k in enumerate(('whole','phone')):
        rec[k]={}
        for name,(p,a) in predictions.items():
            index=j*len(rows)+i;rec[k][name]=dict(candidate=brands[int(p[index].argmax())],accepted=bool(a[index]),confidence=float(p[index].max()))
    records.append(rec)
def measure(rs,k,name):
    accepted=[r for r in rs if r[k][name]['accepted']];return dict(count=len(rs),top1=sum(r[k][name]['candidate']==r['truth'] for r in rs)/len(rs) if rs else None,confirmed=len(accepted),correct_confirmed=sum(r[k][name]['candidate']==r['truth'] for r in accepted),confirmed_precision=sum(r[k][name]['candidate']==r['truth'] for r in accepted)/len(accepted) if accepted else None)
cohorts=dict(all=records,historical=[r for r in records if r['historical']],new_source=[r for r in records if r['new_source']],priority=[r for r in records if r['truth'] in policy['priority_makes']]);results={c:{k:{n:measure(rs,k,n) for n in predictions} for k in ('whole','phone')} for c,rs in cohorts.items()};a=results['all']['whole'];h=results['historical']['whole'];fresh=results['new_source']['whole'];priority=results['priority']['whole'];g=policy['test_release']
checks=dict(all_accuracy=a['candidate']['top1']>=max(g['all_top1_min'],a['baseline']['top1']),historical=h['candidate']['top1']>=g['historical_top1_min'],confirmed_precision=(a['candidate']['confirmed_precision'] or 0)>=g['confirmed_precision_min'],correct_confirmed=a['candidate']['correct_confirmed']>=a['baseline']['correct_confirmed'],new_source_count=fresh['candidate']['count']>=g['new_source_min_count'],new_source_accuracy=bool(fresh['candidate']['count']) and fresh['candidate']['top1']>=fresh['baseline']['top1'],priority_gain=priority['candidate']['top1']>=priority['baseline']['top1']+g['priority_top1_gain_min'])
report=dict(passed=all(checks.values()),checks=checks,results=results,records=records,per_make={b:{k:measure([r for r in records if r['truth']==b],k,'candidate') for k in ('whole','phone')} for b in brands},candidate_sha256=hashlib.sha256((out/'semantic_make.pt').read_bytes()).hexdigest(),test_used_for_selection=False,limitations='Historical public test reused. New_source is a newly collected group-separated public-image check, not a blind Uzbekistan phone exam. Private labels are training-only. Exact model variants are not certified by category tags.')
(out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(out/'release_gate.json').write_text(json.dumps(dict(passed=report['passed'],checks=checks),indent=2),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],checks=checks,results=results)),flush=True)
