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
from tools.cache_multimodal_v24 import token
from tools.make_scope_features_v23 import key
from carvision.core import Pipeline
from carvision.scene import classify
torch.set_num_threads(3);out=ROOT/'models/multimodal_v24';started=time.time()
assert not (out/'evaluation.json').exists(),'Do not rerun heldout evaluation'
with (out/'evaluation_once.json').open('x') as f:json.dump(dict(started=started),f)
training=json.loads((out/'training.json').read_text(encoding='utf-8'));policy=training['policy']['release_requirements']
assert training['accepted_validation_views']>=policy['validation_confirmation_views_min'] and training['accepted_validation_precision']>=policy['validation_confirmation_precision_min']
data_path=ROOT/'cache/multimodal_v24_data.pt';assert hashlib.sha256(data_path.read_bytes()).hexdigest()==training['dataset_sha256']
data=torch.load(data_path,weights_only=True);rows=[r for r in data['rows'] if r['split']=='test'];brands=data['brands']
ck=torch.load(out/'multimodal_make.pt',weights_only=True);candidate_sha=hashlib.sha256((out/'multimodal_make.pt').read_bytes()).hexdigest()
features=torch.load(ROOT/'cache/multimodal_v24_features.pt',weights_only=True)['items']
texts=torch.load(ROOT/'cache/vehicle_text_v24.pt',weights_only=True)['items'] if ck['with_text'] else {}
old_features=torch.load(ROOT/'cache/make_scope_v23_features.pt',weights_only=True)['items']
baseline_dir=ROOT/'outputs/Transfer extracted София/CarVision/models'
assert hashlib.sha256((baseline_dir/'vehicle_model.pt').read_bytes()).hexdigest()=='9c5ba7e381a4e9d8bfc0f704d6857c18a027e4a9882c01ceb83217c52ae443d5'
baseline=Pipeline(baseline_dir,mode='fast');baseline.make_expert=None;baseline.colour_classifier=None
network=torch.nn.Linear(len(ck['mean']),len(brands)).eval();network.load_state_dict(ck['head']);records=[]
with torch.inference_mode():
    for i,r in enumerate(rows):
        record=dict(image=r['image'],truth=r['brand'],group=r['group'],historical=r['historical'])
        for kind in ('whole','phone'):
            x=features[token(r,kind)]
            if ck['with_text']:
                text_key=hashlib.sha256(r['body_rgb'].numpy().tobytes()+r['body_mask'].numpy().tobytes()).hexdigest()
                x=torch.cat([x,texts[text_key]['features'] if kind=='whole' else torch.zeros(512)])
            p=(network((x-ck['mean'])/ck['std'])/ck['temperature']).softmax(0);values,ids=p.topk(3);b=brands[int(ids[0])]
            w,h=r['body_rgb'].shape[1],r['body_rgb'].shape[0]
            if kind=='phone':scale=min(1,96/max(w,h));w,h=round(w*scale),round(h*scale)
            accepted=ck['thresholds'][b]<1 and float(values[0])>=ck['thresholds'][b] and float(values[0]-values[1])>=ck['margin'] and min(w,h)>=40
            record[kind]=dict(candidate=b,confidence=float(values[0]),accepted=accepted,top3=[brands[int(j)] for j in ids])
        old_p=old_features[key(r,'whole')]['old'];scores=collections.defaultdict(float)
        for j,b in enumerate(baseline.vehicle_classifier.brands):scores[b]+=float(old_p[j])
        record['baseline_body_candidate']=max(scores,key=scores.get)
        old=classify(baseline,Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy());record['baseline_app_brand']=old['brand']
        records.append(record)
        if i%50==0:
            (out/'evaluation_progress.json').write_text(json.dumps(dict(completed=i+1,total=len(rows),seconds=time.time()-started)),encoding='utf-8');print('TEST',i+1,len(rows),flush=True)
def measure(rs,kind):
    n=len(rs);confirmed=[r for r in rs if r[kind]['accepted']]
    return dict(count=n,top1=sum(r[kind]['candidate']==r['truth'] for r in rs)/n if n else None,top3=sum(r['truth'] in r[kind]['top3'] for r in rs)/n if n else None,confirmed=len(confirmed),confirmed_precision=sum(r[kind]['candidate']==r['truth'] for r in confirmed)/len(confirmed) if confirmed else None,correct_confirmed=sum(r[kind]['candidate']==r['truth'] for r in confirmed),baseline_body_top1=sum(r['baseline_body_candidate']==r['truth'] for r in rs)/n if n and kind=='whole' else None,baseline_app_correct=sum(r['baseline_app_brand']==r['truth'] for r in rs) if kind=='whole' else None,baseline_app_confirmed=sum(r['baseline_app_brand']!='unknown' for r in rs) if kind=='whole' else None)
cohorts={'all':records,'historical':[r for r in records if r['historical']],'priority_byd_li_zeekr':[r for r in records if r['truth'] in ('BYD','Li Auto','Zeekr')],'new_data':[r for r in records if not r['historical']]}
results={name:{kind:measure(rs,kind) for kind in ('whole','phone')} for name,rs in cohorts.items()};all_result=results['all']['whole'];historical=results['historical']['whole']
checks=dict(top1_gain=all_result['top1']>=all_result['baseline_body_top1']+policy['heldout_make_top1_gain_min'],historical_guard=historical['top1']>=historical['baseline_body_top1']+policy['heldout_historical_top1_min_delta'],confirmed_precision=(all_result['confirmed_precision'] or 0)>=policy['heldout_confirmed_precision_min'],confirmed_count=all_result['confirmed']>=policy['heldout_confirmed_count_min'])
report=dict(results=results,per_make={b:measure([r for r in records if r['truth']==b],'whole') for b in brands},records=records,candidate_sha256=candidate_sha,dataset_sha256=training['dataset_sha256'],seconds=time.time()-started,test_used_for_selection=False,baseline_app_scope='Current compatibility source with 2.2 weights and no new make head; not the previous frozen EXE. Primary candidate comparison uses the unchanged cached 2.2 body head.',limitations='Existing grouped public-source heldout; some historical data was evaluated by earlier releases. Synthetic phone views are correlated. This is not a new Uzbekistan phone field trial or proof of all 325 exact models.')
(out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');(out/'release_gate.json').write_text(json.dumps(dict(passed=all(checks.values()),checks=checks),indent=2),encoding='utf-8');print('EVALUATION COMPLETE',json.dumps(dict(results=results,checks=checks)),flush=True)
