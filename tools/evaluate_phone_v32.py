"""Frozen correction on reused public test: regression evidence, never blind accuracy."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from tools.ev26_helpers import references,route,metrics,acceptance
from tools.train_truth_v30 import valid
def main():
    torch.set_num_threads(3);out=ROOT/'models/phone_v32';assert json.loads((out/'scoped_validation.json').read_text())['passed']
    with (out/'regression_once.json').open('x') as f:json.dump(dict(no_retuning_after=True),f)
    ck=torch.load(out/'scoped_candidate.pt',weights_only=True);d=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True);rows=json.loads((ROOT/'cache/truth_v31/rows.json').read_text(encoding='utf-8'));n=len(rows);ids=torch.tensor([i for i,r in enumerate(rows) if r['split']=='test']);rs=[dict(rows[int(i)],y=ck['brands'].index(rows[int(i)]['brand']),historical=rows[int(i)].get('historical',False)) for i in ids];q=d['x'][torch.cat([ids,ids+n])];brands=ck['brands'];geom=torch.cat([valid(rs,k) for k in ('whole','phone')]);cg=torch.cat([valid(rs,k,True) for k in ('whole','phone')]);make=CosineMakeHead(torch.zeros(len(brands),1280)).eval();make.load_state_dict(ck['head']);f=ck['model_head'];pairs=list(map(tuple,f['pairs']));keys=[b+'|'+m for b,m in pairs];family=CosineMakeHead(torch.zeros(len(pairs),1280)).eval();family.load_state_dict(f['head']);mapping=torch.tensor([brands.index(b) for b,m in pairs]);c=ck['make_correction_head'];cor=CosineMakeHead(torch.zeros(len(brands),1280)).eval();cor.load_state_dict(c['head'])
    with torch.no_grad():
        p,a=route((make(q)/ck['temperature']).softmax(1),references(q,ck['retrieval'],len(brands)),ck['thresholds'],ck['retrieval']['fallback_thresholds'],brands,geom);fp=(family(q)/f['temperature']).softmax(1);v,ii=fp.topk(2,dim=1);ix=ii[:,0];g=torch.tensor([f['thresholds'][keys[int(i)]] for i in ix]);sg=torch.tensor([f['similarity_thresholds'][keys[int(i)]] for i in ix]);sim=q@torch.nn.functional.normalize(family.weight,dim=1).T;fa=(~a)&cg&(g<1)&(v[:,0]>=g)&((v[:,0]-v[:,1])>=f['margin'])&(sim.gather(1,ix[:,None])[:,0]>=sg)&torch.isin(mapping[ix],torch.tensor([brands.index(b) for b in f['manufacturer_fallback_brands']]));mp=torch.zeros_like(p);mp.scatter_add_(1,mapping.expand(len(q),-1),fp);bp=torch.where(fa[:,None],mp,p);ba=a|fa;cp=(cor(q)/c['temperature']).softmax(1);ca=acceptance(cp,c['thresholds'],brands,cg,c['margin']);np=torch.where(ca[:,None],cp,bp);na=ba|ca
    base=metrics(bp,ba,rs,brands,c['scope']);new=metrics(np,na,rs,brands,c['scope']);checks=dict(correct_confirmations=all(new[k]['correct_confirmed']>=base[k]['correct_confirmed'] for k in base),precision=all(new[k]['precision']>=base[k]['precision']-.002 for k in base),top1=all(new[k]['top1']>=base[k]['top1'] for k in base));report=dict(passed=all(checks.values()),checks=checks,baseline=base,candidate=new,make_images=len(rs),independent_test=False,scope='Frozen post-selection evaluation on reused development benchmark. New six phone corrections are TRAIN and absent here. No threshold changes or retries after this result.')
    (out/'regression.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
