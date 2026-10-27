"""Paired known-error review after freezing selection; never feeds training."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from carvision.core import Pipeline
from carvision.semantic_make import SemanticMakeExpert

OUT=ROOT/'models/encoder_adaptation_v27'
assert (OUT/'evaluation.json').exists(), 'Complete the fixed heldout evaluation first'
with (OUT/'phone_diagnostics_once.json').open('x') as f:json.dump(dict(post_selection_only=True),f)
torch.set_num_threads(3)
pipeline=Pipeline(ROOT/'outputs/CarVision-2.6-Full/CarVision/models',mode='fast')
experts={'baseline':pipeline.make_expert,'candidate':SemanticMakeExpert(torch.load(OUT/'semantic_make.pt',weights_only=True))}
registry=json.loads((OUT/'user_error_registry.json').read_text())
mapping=json.loads((OUT/'screenshot_original_mapping.json').read_text())
cases=[]
for label,m in zip(registry,mapping):
    score,path,_=m['matches'][0]
    assert score>.9 and score-m['matches'][1][0]>.2
    cases.append(dict(number=label['number'],image=path,user_brand=label['brand'],user_model=label['model'],truth_status='user-corrected' if label['number']<=5 else 'tentative' if label['number']==6 else 'unlabelled',same_vehicle_group='GAC-Aion-first-two' if label['number']<=2 else str(label['number'])))
cases.append(dict(number='previous-Toyota',image='D:/машинки/photo_16_2026-10-25_20-44-40.jpg',user_brand='Toyota',user_model='unknown',truth_status='visible-emblem prior development review'))
cases.append(dict(number='previous-night',image='C:/Temp/codex-clipboard-779e498d-79c4-4db9-9fd4-61cd3491b753.png',user_brand='Weltmeister',user_model='unknown',truth_status='assistant visual reference comparison; not independent certified truth'))
records=[]
for case in cases:
    rec=dict(case)
    for name,expert in experts.items():
        pipeline.make_expert=expert
        result=pipeline.predict(case['image'])
        row=max(result['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]),default={})
        evidence=row.get('make_evidence') or {}
        rec[name]=dict(brand=row.get('brand'),candidate=evidence.get('candidate'),confidence=evidence.get('confidence'),accepted=evidence.get('accepted'),model=row.get('model'),model_alternatives=row.get('model_alternatives'),colour=row.get('colour'),alternatives=evidence.get('alternatives'),car_count=len(result['cars']))
    records.append(rec)
    (OUT/'phone_diagnostics.json').write_text(json.dumps(dict(records=records,limitations='Known development errors only; repeated same vehicle in first two. Cases6/7 lack certified truth. Not an independent accuracy estimate. Timings omitted because segmentation cache makes paired timings unequal.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(number=case['number'],truth=case['user_brand'],baseline=rec['baseline'],candidate=rec['candidate'])),flush=True)
