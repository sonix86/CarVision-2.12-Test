"""Validation-only fusion of the two existing candidates for specified makes."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from tools.ev26_helpers import references,route,acceptance,calibrate,metrics
from tools.train_truth_v30 import valid
def main():
    torch.set_num_threads(3);out=ROOT/'models/phone_v32';orig=torch.load(ROOT/'models/truth_v31/scoped_candidate.pt',weights_only=True);rows=json.loads((ROOT/'cache/truth_v31/rows.json').read_text(encoding='utf-8'));d=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True);own=torch.load(ROOT/'cache/phone_v32/features.pt',weights_only=True);brands=orig['brands'];n=len(rows);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation']);vr=[dict(rows[int(i)],y=brands.index(rows[int(i)]['brand']),historical=rows[int(i)].get('historical',False)) for i in vi];q=d['x'][torch.cat([vi,vi+n])];y=torch.tensor([r['y'] for r in vr]*2);groups=torch.tensor([sorted({r['group'] for r in vr}).index(r['group']) for r in vr]*2);geom=torch.cat([valid(vr,k) for k in ('whole','phone')]);cg=torch.cat([valid(vr,k,True) for k in ('whole','phone')]);scope=['Zeekr','Leapmotor','BYD'];priority=scope+['Voyah','Chevrolet','Dongfeng']
    def head(state):
        h=CosineMakeHead(torch.zeros(len(brands),1280)).eval();h.load_state_dict(state);return h
    mh=head(orig['head']);fh=orig['model_head'];pairs=list(map(tuple,fh['pairs']));fkeys=[b+'|'+m for b,m in pairs];family=CosineMakeHead(torch.zeros(len(pairs),1280)).eval();family.load_state_dict(fh['head']);mapping=torch.tensor([brands.index(b) for b,m in pairs])
    def base(z,good,fgood):
        p,a=route((mh(z)/orig['temperature']).softmax(1),references(z,orig['retrieval'],len(brands)),orig['thresholds'],orig['retrieval']['fallback_thresholds'],brands,good);fp=(family(z)/fh['temperature']).softmax(1);v,ids=fp.topk(2,dim=1);idx=ids[:,0];g=torch.tensor([fh['thresholds'][fkeys[int(i)]] for i in idx]);sg=torch.tensor([fh['similarity_thresholds'][fkeys[int(i)]] for i in idx]);sim=z@torch.nn.functional.normalize(family.weight,dim=1).T;ok=fgood&(g<1)&(v[:,0]>=g)&((v[:,0]-v[:,1])>=fh['margin'])&(sim.gather(1,idx[:,None])[:,0]>=sg)&torch.isin(mapping[idx],torch.tensor([brands.index(b) for b in fh['manufacturer_fallback_brands']]));fallback=(~a)&ok;mp=torch.zeros_like(p);mp.scatter_add_(1,mapping.expand(len(z),-1),fp);return torch.where(fallback[:,None],mp,p),a|fallback
    with torch.no_grad():bp,ba=base(q,geom,cg);bm=metrics(bp,ba,vr,brands,priority)
    plan=dict(scope=scope,reason='Full replacement rejected. Apply a second own head only when its global top1 is a calibrated scoped make; preserve all other primary/family decisions. Same two fits, no additional training. Validation selects; no test consulted.',gates='At least3 source groups, precision=100% on validation; minimum3 groups, global probability and margin plus geometry. No manual threshold reduction.');(out/'scoped_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');report=dict(baseline=bm,rounds=[],test_used=False);best=None
    for rid in (1,2):
        path=out/f'round{rid}.pt'
        if not path.exists():continue
        ck=torch.load(path,weights_only=True);h=head(ck['head'])
        with torch.no_grad():
            p=(h(q)/ck['temperature']).softmax(1);gates=calibrate(p,y,groups,brands,cg,1.0);gates={b:(gates[b] if b in scope else 1.) for b in brands};a=acceptance(p,gates,brands,cg);cp=torch.where(a[:,None],p,bp);ca=a|ba;m=metrics(cp,ca,vr,brands,priority);op=(h(own['x'])/ck['temperature']).softmax(1);og=torch.tensor([min(v['size'])>=64 for v in own['meta']]);oa=acceptance(op,gates,brands,og);ids=torch.tensor([i for i,v in enumerate(own['meta']) if v['kind']=='whole' and v['brand'] in scope]);fit=bool((oa[ids]&(op[ids].argmax(1)==own['y'][ids])).all());checks=dict(known_scope_train=fit,precision=all(m[k]['precision']>=bm[k]['precision'] for k in m),confirmed=all(m[k]['correct_confirmed']>=bm[k]['correct_confirmed'] for k in m),top1=all(m[k]['top1']>=bm[k]['top1'] for k in m));entry=dict(round=rid,passed=all(checks.values()),checks=checks,metrics=m,gates={b:gates[b] for b in scope},own=[dict(brand=own['meta'][int(i)]['brand'],candidate=brands[int(op[i].argmax())],accepted=bool(oa[i])) for i in ids]);report['rounds'].append(entry)
        if entry['passed']:
            score=sum(v['correct_confirmed'] for v in m.values())
            if best is None or score>best['score']:
                best=dict(round=rid,score=score);result=dict(orig);result['make_correction_head']=dict(origin='project-trained-phone-make-correction-v32',brands=brands,head=ck['head'],temperature=ck['temperature'],thresholds=gates,margin=.06,minimum_size=64,scope=scope,known_failures_are_train=True);torch.save(result,out/'scoped_candidate.pt')
        print('SCOPED32',json.dumps(entry),flush=True)
    report.update(passed=best is not None,selected=best);(out/'scoped_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('SCOPE COMPLETE',best,flush=True)
if __name__=='__main__':main()
