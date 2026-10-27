"""Validation-only routing ablation of the same two fits, no new training/test use."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from tools.ev26_helpers import route,references,metrics
from tools.train_truth_v30 import valid
def main():
    torch.set_num_threads(3);out=ROOT/'models/phone_v32';original=torch.load(ROOT/'models/truth_v31/scoped_candidate.pt',weights_only=True);base=json.loads((out/'validation.json').read_text(encoding='utf-8'))['baseline'];rows=json.loads((ROOT/'cache/truth_v31/rows.json').read_text(encoding='utf-8'));d=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True);own=torch.load(ROOT/'cache/phone_v32/features.pt',weights_only=True);brands=original['brands'];n=len(rows);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation']);vr=[dict(rows[int(i)],y=brands.index(rows[int(i)]['brand']),historical=rows[int(i)].get('historical',False)) for i in vi];q=d['x'][torch.cat([vi,vi+n])];geom=torch.cat([valid(vr,k) for k in ('whole','phone')]);report=dict(reason='New calibration disabled some existing common classes because current validation support is limited. Evaluate exact original production gates and temperature, never lower gates; same two weight candidates, no extra training. Validation ablation declared after first validation failures, before any test. Not a predeclared success or independent result.',test_used=False,rounds=[]);(out/'retained_gates_plan.json').write_text(json.dumps(report,indent=2),encoding='utf-8');best=None
    for rid in (1,2):
        p=out/f'round{rid}.pt'
        if not p.exists():continue
        ck=torch.load(p,weights_only=True);h=CosineMakeHead(torch.zeros(len(brands),1280)).eval();h.load_state_dict(ck['head']);bank=ck['retrieval'];bank['fallback_thresholds']=original['retrieval']['fallback_thresholds'];ck.update(temperature=original['temperature'],thresholds=original['thresholds'],retrieval=bank)
        with torch.no_grad():
            rp,ra=route((h(q)/ck['temperature']).softmax(1),references(q,bank,len(brands)),ck['thresholds'],bank['fallback_thresholds'],brands,geom);m=metrics(rp,ra,vr,brands,['Zeekr','Leapmotor','Voyah','BYD','Chevrolet','Dongfeng']);oq,oa=route((h(own['x'])/ck['temperature']).softmax(1),references(own['x'],bank,len(brands)),ck['thresholds'],bank['fallback_thresholds'],brands,torch.tensor([min(v['size'])>=40 for v in own['meta']]));ids=torch.tensor([i for i,v in enumerate(own['meta']) if v['kind']=='whole']);known=bool((oa[ids]&(oq[ids].argmax(1)==own['y'][ids])).all())
        checks=dict(known_train_originals_correct=known,precision=all(m[k]['precision']>=base[k]['precision']-.002 for k in m),confirmed=all(m[k]['correct_confirmed']>=base[k]['correct_confirmed'] for k in m),top1=all(m[k]['top1']>=base[k]['top1']-.005 for k in m));r=dict(round=rid,checks=checks,passed=all(checks.values()),metrics=m,train=[dict(brand=own['meta'][int(i)]['brand'],accepted=bool(oa[i]),candidate=brands[int(oq[i].argmax())]) for i in ids]);report['rounds'].append(r)
        if r['passed']:
            score=sum(v['correct_confirmed'] for v in m.values())
            if best is None or score>best['score']:best=dict(round=rid,score=score);ck['phone_v32']['calibration']='Original production thresholds, temperature, fallback gates retained; validation-only ablation';torch.save(ck,out/'candidate.pt')
        print('RETAINED GATES',json.dumps(r),flush=True)
    report.update(passed=best is not None,selected=best);(out/'retained_gates_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('RETAINED COMPLETE',best,flush=True)
if __name__=='__main__':main()
