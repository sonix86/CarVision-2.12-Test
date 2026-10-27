"""One fixed-model heldout evaluation. No fitting, epoch or threshold reselection.

The cached 2.2 body head is the like-for-like candidate baseline. The auxiliary
pipeline decisions use current compatibility source plus 2.2 weights, not the
old frozen EXE; they are explicitly identified in the report.
"""
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cache_semantic_v25 import token
from tools.make_scope_features_v23 import key
from carvision.core import Pipeline
from carvision.scene import classify
from carvision.semantic_make import CosineMakeHead,embedding_features
torch.set_num_threads(3);out=ROOT/'models/semantic_v25';started=time.time()
assert not (out/'evaluation.json').exists(),'Do not rerun heldout evaluation'
with (out/'evaluation_once.json').open('x') as f:json.dump(dict(started=started),f)
training=json.loads((out/'training.json').read_text(encoding='utf-8'));policy=training['policy']['test_release']
assert json.loads((out/'validation_gate.json').read_text(encoding='utf-8'))['passed']
data_path=ROOT/'cache/multimodal_v24_data.pt';assert hashlib.sha256(data_path.read_bytes()).hexdigest()==training['dataset_sha256']
data=torch.load(data_path,weights_only=True);rows=[r for r in data['rows'] if r['split']=='test'];brands=data['brands']
ck=torch.load(out/'semantic_make.pt',weights_only=True);candidate_sha=hashlib.sha256((out/'semantic_make.pt').read_bytes()).hexdigest()
features=torch.load(ROOT/'cache/semantic_v25/features.pt',weights_only=True)['items']
assert ck['encoder_source_sha256']==training['encoder_source_sha256']
old_features=torch.load(ROOT/'cache/make_scope_v23_features.pt',weights_only=True)['items']
baseline_dir=ROOT/'outputs/Transfer extracted София/CarVision/models'
assert hashlib.sha256((baseline_dir/'vehicle_model.pt').read_bytes()).hexdigest()=='9c5ba7e381a4e9d8bfc0f704d6857c18a027e4a9882c01ceb83217c52ae443d5'
old_brands=torch.load(baseline_dir/'vehicle_model.pt',weights_only=True)['brands']
# The identical old pipeline has already been measured once on every row.
# Reuse that frozen baseline protocol; do not run 910 unchanged inferences again.
baseline_report_path=ROOT/'models/multimodal_v24/evaluation.json'
baseline_report=json.loads(baseline_report_path.read_text(encoding='utf-8'))
assert baseline_report['dataset_sha256']==training['dataset_sha256']
baseline_records={r['image']:r for r in baseline_report['records']}
assert set(baseline_records)=={r['image'] for r in rows}
network=CosineMakeHead(torch.zeros((len(brands),1024 if ck['two_views'] else 512))).eval();network.load_state_dict(ck['head']);records=[]
with torch.inference_mode():
    for i,r in enumerate(rows):
        record=dict(image=r['image'],truth=r['brand'],group=r['group'],historical=r['historical'])
        for kind in ('whole','phone'):
            x=embedding_features(features[token(r,kind)],ck['two_views'])
            p=(network(x)/ck['temperature']).softmax(0);values,ids=p.topk(3);b=brands[int(ids[0])]
            w,h=r['body_rgb'].shape[1],r['body_rgb'].shape[0]
            if kind=='phone':scale=min(1,96/max(w,h));w,h=round(w*scale),round(h*scale)
            accepted=ck['thresholds'][b]<1 and float(values[0])>=ck['thresholds'][b] and float(values[0]-values[1])>=ck['margin'] and min(w,h)>=40
            record[kind]=dict(candidate=b,confidence=float(values[0]),accepted=accepted,top3=[brands[int(j)] for j in ids])
        old_p=old_features[key(r,'whole')]['old'];scores=collections.defaultdict(float)
        for j,b in enumerate(old_brands):scores[b]+=float(old_p[j])
        record['baseline_body_candidate']=max(scores,key=scores.get)
        old=baseline_records[r['image']];assert old['truth']==r['brand'] and old['historical']==r['historical']
        assert old['baseline_body_candidate']==record['baseline_body_candidate']
        record['baseline_app_brand']=old['baseline_app_brand']
        records.append(record)
        if i%50==0:
            (out/'evaluation_progress.json').write_text(json.dumps(dict(completed=i+1,total=len(rows),seconds=time.time()-started)),encoding='utf-8');print('TEST',i+1,len(rows),flush=True)
def measure(rs,kind):
    n=len(rs);confirmed=[r for r in rs if r[kind]['accepted']]
    return dict(count=n,top1=sum(r[kind]['candidate']==r['truth'] for r in rs)/n if n else None,top3=sum(r['truth'] in r[kind]['top3'] for r in rs)/n if n else None,confirmed=len(confirmed),confirmed_precision=sum(r[kind]['candidate']==r['truth'] for r in confirmed)/len(confirmed) if confirmed else None,correct_confirmed=sum(r[kind]['candidate']==r['truth'] for r in confirmed),baseline_body_top1=sum(r['baseline_body_candidate']==r['truth'] for r in rs)/n if n and kind=='whole' else None,baseline_app_correct=sum(r['baseline_app_brand']==r['truth'] for r in rs) if kind=='whole' else None,baseline_app_confirmed=sum(r['baseline_app_brand']!='unknown' for r in rs) if kind=='whole' else None)
cohorts={'all':records,'historical':[r for r in records if r['historical']],'priority_byd_li_zeekr':[r for r in records if r['truth'] in ('BYD','Li Auto','Zeekr')],'new_data':[r for r in records if not r['historical']]}
results={name:{kind:measure(rs,kind) for kind in ('whole','phone')} for name,rs in cohorts.items()};all_result=results['all']['whole'];historical=results['historical']['whole']
checks=dict(top1_gain=all_result['top1']>=all_result['baseline_body_top1']+policy['whole_top1_gain_min'],historical_guard=historical['top1']>=historical['baseline_body_top1']+policy['historical_min_delta'],confirmed_precision=(all_result['confirmed_precision'] or 0)>=policy['confirmed_precision_min'],confirmed_count=all_result['confirmed']>=policy['confirmed_count_min'])
checks['correct_confirmed_guard']=all_result['correct_confirmed']>=all_result['baseline_app_correct']
baseline_reuse=dict(path=str(baseline_report_path),sha256=hashlib.sha256(baseline_report_path.read_bytes()).hexdigest(),unchanged_vehicle_model_sha256=hashlib.sha256((baseline_dir/'vehicle_model.pt').read_bytes()).hexdigest(),rows=len(baseline_records))
report=dict(results=results,per_make={b:measure([r for r in records if r['truth']==b],'whole') for b in brands},records=records,candidate_sha256=candidate_sha,dataset_sha256=training['dataset_sha256'],seconds=time.time()-started,test_used_for_selection=False,baseline_app_scope='Current compatibility source with 2.2 weights and no new make head; not the previous frozen EXE. Primary candidate comparison uses the unchanged cached 2.2 body head.',limitations='Existing grouped public-source heldout; some historical data was evaluated by earlier releases. Synthetic phone views are correlated. This is not a new Uzbekistan phone field trial or proof of all 325 exact models.')
report['frozen_baseline_reuse']=baseline_reuse
(out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');(out/'release_gate.json').write_text(json.dumps(dict(passed=all(checks.values()),checks=checks),indent=2),encoding='utf-8');print('EVALUATION COMPLETE',json.dumps(dict(results=results,checks=checks)),flush=True)
