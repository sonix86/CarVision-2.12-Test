"""One post-calibration reused heldout evaluation, plus unscored owner diagnostics."""
import json,sys,hashlib
from pathlib import Path
import torch
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cache_semantic_v25 import token
from carvision.semantic_make import CosineMakeHead,embedding_features,SemanticMakeExpert
from carvision.visual_retrieval import routed_probabilities
from carvision.segmentation import decode_mask
torch.set_num_threads(3);out=ROOT/'models/fallback_v253';assert json.loads((out/'validation.json').read_text())['passed']
with (out/'evaluation_once.json').open('x') as f:json.dump({'selection_complete':True,'test_reused_not_fresh_field_trial':True},f)
ck=torch.load(out/'semantic_make.pt',weights_only=True);data=torch.load(ROOT/'cache/multimodal_v24_data.pt',weights_only=True);rows=[r for r in data['rows'] if r['split']=='test'];features=torch.load(ROOT/'cache/semantic_v25/features.pt',weights_only=True)['items'];brands=ck['brands'];head=CosineMakeHead(torch.zeros((len(brands),512))).eval();head.load_state_dict(ck['head']);prior=json.loads((ROOT/'models/semantic_v25/evaluation.json').read_text(encoding='utf-8'));old={r['image']:r for r in prior['records']};records=[]
with torch.inference_mode():
    for r in rows:
        rec=dict(image=r['image'],truth=r['brand'],historical=r['historical'])
        for kind in ('whole','phone'):
            x=embedding_features(features[token(r,kind)],False);neural=(head(x)/ck['temperature']).softmax(0);h,w=r['body_rgb'].shape[:2];s=min(1,96/max(h,w)) if kind=='phone' else 1;geometry=min(round(h*s),round(w*s))>=40
            p,t,route=routed_probabilities(x,neural,ck['retrieval'],brands,ck['thresholds'],ck['margin'],geometry);v,ids=p.topk(2);b=brands[int(ids[0])];accepted=t<1 and float(v[0])>=t and float(v[0]-v[1])>=ck['margin'] and geometry
            rec[kind]=dict(candidate=b,accepted=accepted,confidence=float(v[0]),route=route)
            if old[r['image']][kind]['accepted']:assert b==old[r['image']][kind]['candidate'] and accepted
        records.append(rec)
def measure(rs,kind):
    accepted=[r for r in rs if r[kind]['accepted']];return dict(count=len(rs),top1=sum(r[kind]['candidate']==r['truth'] for r in rs)/len(rs),confirmed=len(accepted),correct_confirmed=sum(r[kind]['candidate']==r['truth'] for r in accepted),confirmed_precision=sum(r[kind]['candidate']==r['truth'] for r in accepted)/len(accepted))
results={c:{k:measure(rs,k) for k in ('whole','phone')} for c,rs in {'all':records,'historical':[r for r in records if r['historical']],'priority':[r for r in records if r['truth'] in ('BYD','Li Auto','Zeekr')]}.items()};a=results['all']['whole'];base=prior['results']['all']['whole'];checks=dict(top1_not_worse_than_v25=a['top1']>=base['top1'],historical_not_worse=results['historical']['whole']['top1']>=prior['results']['historical']['whole']['top1'],confirmed_precision=a['confirmed_precision']>=.95,confirmed_correct_gain=a['correct_confirmed']>base['correct_confirmed'],primary_accepted_preserved=True)
report=dict(results=results,checks=checks,passed=all(checks.values()),records=records,candidate_sha256=hashlib.sha256((out/'semantic_make.pt').read_bytes()).hexdigest(),limitations='Reuses existing grouped public heldout. No fresh Uzbekistan field-accuracy claim. Diagnostics below do not select parameters.')
(out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(results=results,checks=checks)),flush=True)
expert=SemanticMakeExpert(ck);diagnostics=[]
for number in (2,5,16):
    d=json.loads((ROOT/f'outputs/semantic25-recipient-verification/owner-diagnostic-{number}.stdout.txt').read_text(encoding='utf-8-sig'));r=max(d['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]));im=ImageOps.exif_transpose(Image.open(d['image'])).convert('RGB').crop(r['box']);e=expert.predict(im,decode_mask(r['segmentation']));diagnostics.append(dict(photo=number,brand=e['brand'],candidate=e['candidate'],confidence=e['confidence'],alternatives=e['alternatives']))
(out/'owner_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8');print('OWNER DIAGNOSTICS',json.dumps(diagnostics),flush=True)
