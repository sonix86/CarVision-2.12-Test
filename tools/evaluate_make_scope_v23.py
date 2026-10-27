"""One post-selection make evaluation, with abstention and actual prior app decisions."""
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import Pipeline
from carvision.scene import classify
from carvision.make_expert import MakeExpert
from tools.make_scope_features_v23 import EVAL_VIEWS,views
torch.set_num_threads(3);OUT=ROOT/'models/make_scope_v23';started=time.perf_counter()
report=json.loads((OUT/'training.json').read_text(encoding='utf-8'))
dataset=ROOT/'cache/make_scope_v23.pt'
if not dataset.exists():dataset=ROOT/'Development/TrainingInputs/make_scope_v23.pt'
assert report['dataset_sha256']==hashlib.sha256(dataset.read_bytes()).hexdigest()
assert not (OUT/'evaluation.json').exists(),'Do not rerun or overwrite heldout evaluation'
data=torch.load(dataset,weights_only=True);brands=data['brands'];test=[r for r in data['rows'] if r['split'] in ['test','family_test']]
cache=torch.load(ROOT/'cache/make_scope_v23_features.pt',weights_only=True)
expert=MakeExpert(torch.load(OUT/'make_expert.pt',weights_only=True));assert expert.brands==brands
baseline_dir=ROOT/'outputs/Transfer extracted София/CarVision/models'
if not baseline_dir.exists():baseline_dir=ROOT/'models'
assert hashlib.sha256((baseline_dir/'vehicle_model.pt').read_bytes()).hexdigest()==report['source_body_sha256']
baseline=Pipeline(baseline_dir,mode='fast');baseline.make_expert=None
old_brands=baseline.vehicle_classifier.brands
# Colour is independent of the make decision; do not benchmark colour here.
baseline.colour_classifier=None
def key(r,kind):return hashlib.sha256(r['body_rgb'].numpy().tobytes()+r['body_mask'].numpy().tobytes()+kind.encode()).hexdigest()
records=[]
with torch.inference_mode():
    for i,r in enumerate(test):
        row=dict(image=r['image'],truth=r['brand'],historical=r['historical'],group=r['group'],family=r.get('family'),split=r['split'])
        old_prob=cache['items'][key(r,'whole')]['old'];old_make=torch.zeros(len(brands))
        for j,b in enumerate(old_brands):old_make[brands.index(b)]+=old_prob[j]
        row['baseline_body_candidate']=brands[int(old_make.argmax())]
        for kind in EVAL_VIEWS:
            # Evaluate the selected encoder itself, including any final fine-tune.
            z=expert.encoder(views(r,kind));feature=z[:2].mean(0)*.75+z[2:].mean(0)*.25
            p=(expert.head(torch.nn.functional.normalize(feature,dim=0))/expert.temperature).softmax(0)
            values,indices=p.topk(2);brand=brands[int(indices[0])];confidence=float(values[0]);margin=float(values[0]-values[1]);threshold=float(expert.thresholds[brand])
            width=r['body_rgb'].shape[1];height=r['body_rgb'].shape[0]
            if kind=='phone':width,height=round(width*min(1,96/max(width,height))),round(height*min(1,96/max(width,height)))
            if kind=='upper':height=max(1,round(height*.65))
            accepted=threshold<1. and confidence>=threshold and margin>=expert.margin and min(width,height)>=40
            row[kind]=dict(candidate=brand,confidence=confidence,accepted=accepted,brand=brand if accepted else 'unknown',top3=[brands[int(j)] for j in p.topk(3).indices])
        old=classify(baseline,Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy())
        row['baseline_brand']=old['brand'];row['baseline_confidence']=old['brand_confidence'];records.append(row)
        if i%50==0:print('TEST',i,len(test),flush=True)

def measure(rs,kind='whole'):
    accepted=[r for r in rs if r[kind]['accepted']]
    return dict(count=len(rs),top1=sum(r[kind]['candidate']==r['truth'] for r in rs)/len(rs) if rs else None,
                top3=sum(r['truth'] in r[kind]['top3'] for r in rs)/len(rs) if rs else None,
                confirmed=len(accepted),coverage=len(accepted)/len(rs) if rs else None,
                confirmed_precision=sum(r[kind]['brand']==r['truth'] for r in accepted)/len(accepted) if accepted else None,
                correct_confirmed=sum(r[kind]['brand']==r['truth'] for r in accepted),
                baseline_app_correct=sum(r['baseline_brand']==r['truth'] for r in rs),
                baseline_body_top1=sum(r['baseline_body_candidate']==r['truth'] for r in rs)/len(rs) if rs else None,
                baseline_app_abstained=sum(r['baseline_brand']=='unknown' for r in rs))
results={name:{kind:measure(rs,kind) for kind in EVAL_VIEWS} for name,rs in [('all',records),('historical',[r for r in records if r['historical']]),('new_data',[r for r in records if not r['historical']]),('target_scope_new',[r for r in records if r['image'].startswith(('make_scope_v23/','make_flickr_v23/'))]),('supplementary_makes',[r for r in records if r['image'].startswith('public_prototype/')]),('priority_byd_li_zeekr',[r for r in records if r['truth'] in ['BYD','Li Auto','Zeekr']]),('unseen_family',[r for r in records if r['split']=='family_test'])]}
per_make={b:{kind:measure([r for r in records if r['truth']==b],kind) for kind in EVAL_VIEWS} for b in brands}
evaluation=dict(results=results,per_make=per_make,records=records,source_model_sha256=report['source_body_sha256'],
                confusion_counts={kind:dict(collections.Counter(r['truth']+' -> '+r[kind]['candidate'] for r in records if r[kind]['candidate']!=r['truth'])) for kind in EVAL_VIEWS},
                candidate_sha256=hashlib.sha256((OUT/'make_expert.pt').read_bytes()).hexdigest(),
                dataset_sha256=report['dataset_sha256'],best_epoch=report['best_epoch'],test_used_for_selection=False,
                seconds=time.perf_counter()-started,
                limitations='Grouped source/public photo heldouts; historical cohort already used by previous releases. Synthetic phone views correlated with originals. Not an independent Uzbekistan phone field trial. Tiny per-make sample counts cannot establish reliability.')
(OUT/'evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',json.dumps(results),flush=True)
