"""Bounded preprocessing-only comparison: no further gradient training.

Original encoder/head weights remain immutable. Four fixed view mixtures are
compared on validation; reported user errors never enter this selection.
"""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import GeneralVisionEncoder,CosineMakeHead,image_tensor,visual_inputs,embedding_features
from tools.cache_semantic_v25 import token
torch.set_num_threads(3);out=ROOT/'models/view_calibration_v251';out.mkdir(exist_ok=True)
with (out/'once.json').open('x') as f:json.dump(dict(started=time.time(),gradient_training_rounds=0,schemes=['mask','mask_raw','mask_raw_gray','raw_gray'],selection='validation only',user_errors_used=False),f)
ck_path=ROOT/'models/semantic_v25/semantic_make.pt';source_sha=hashlib.sha256(ck_path.read_bytes()).hexdigest();ck=torch.load(ck_path,weights_only=True);assert not ck['two_views']
data=torch.load(ROOT/'cache/multimodal_v24_data.pt',weights_only=True);rows=[r for r in data['rows'] if r['split']=='validation'];brands=data['brands'];y=torch.tensor([r['y'] for r in rows]);historical=torch.tensor([r['historical'] for r in rows]);old_cache=torch.load(ROOT/'cache/semantic_v25/features.pt',weights_only=True)['items']
encoder=GeneralVisionEncoder().eval();encoder.load_state_dict(ck['encoder_state']);head=CosineMakeHead(torch.zeros((len(brands),512))).eval();head.load_state_dict(ck['head'])
features={};started=time.time()
with torch.inference_mode():
    for kind in ('whole','phone'):
        logits={name:[] for name in ('mask','raw','gray')}
        for first in range(0,len(rows),8):
            batch=rows[first:first+8];inputs=[]
            for r in batch:
                im=Image.fromarray(r['body_rgb'].numpy())
                if kind=='phone':im.thumbnail((96,96));im=ImageEnhance.Brightness(im).enhance(.55).filter(ImageFilter.GaussianBlur(.35))
                inputs.extend([image_tensor(im),image_tensor(ImageOps.grayscale(im).convert('RGB'))])
            z=torch.nn.functional.normalize(encoder(torch.stack(inputs)),dim=1).reshape(-1,2,512)
            for j,r in enumerate(batch):
                logits['mask'].append(head(embedding_features(old_cache[token(r,kind)],False)))
                logits['raw'].append(head(z[j,0]));logits['gray'].append(head(z[j,1]))
            if first%200==0:print('VALIDATION VIEWS',kind,first,len(rows),flush=True)
        features[kind]={k:torch.stack(v) for k,v in logits.items()}
schemes={'mask':['mask'],'mask_raw':['mask','raw'],'mask_raw_gray':['mask','raw','gray'],'raw_gray':['raw','gray']};records=[];best=None
baseline=json.loads((ROOT/'models/semantic_v25/training.json').read_text(encoding='utf-8'))
with torch.inference_mode():
    for name,views in schemes.items():
        def probs(kind,t):return torch.stack([(features[kind][v]/t).softmax(1) for v in views]).mean(0)
        temperature=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(-torch.cat([probs(k,t) for k in features])[torch.arange(2*len(y)),y.repeat(2)].clamp_min(1e-12).log().mean()))
        p={k:probs(k,temperature) for k in features};metrics={}
        for k,q in p.items():
            good=q.argmax(1)==y;metrics[k]=dict(top1=float(good.float().mean()),historical=float(good[historical].float().mean()),macro=float(torch.tensor([float(good[y==j].float().mean()) for j in y.unique()]).mean()))
        priority=[brands.index(b) for b in baseline['policy']['priority_makes']];good=p['whole'].argmax(1)==y;pm=float(torch.tensor([float(good[y==j].float().mean()) for j in priority]).mean());score=.5*metrics['whole']['macro']+.2*metrics['phone']['macro']+.3*pm
        guard=all(metrics[k]['historical']>=baseline['selected_metrics'][k]['historical']-.005 for k in metrics) and metrics['whole']['top1']>=baseline['selected_metrics']['whole']['top1']
        record=dict(scheme=name,temperature=temperature,metrics=metrics,priority_macro=pm,score=score,guard=guard);records.append(record)
        if guard and (best is None or score>best['score']):best=dict(record,p=p)
    assert best is not None
    probs_all=torch.cat([best['p'][k] for k in ('whole','phone')]);labels=y.repeat(2);values,ids=probs_all.topk(2,dim=1);group_names=sorted({r['group'] for r in rows});groups=torch.tensor([group_names.index(r['group']) for r in rows]*2);geometry=[]
    for kind in ('whole','phone'):
        for r in rows:
            h,w=r['body_rgb'].shape[:2];scale=min(1,96/max(w,h)) if kind=='phone' else 1;geometry.append(min(round(w*scale),round(h*scale))>=40)
    geometry=torch.tensor(geometry);thresholds={};accepted=torch.zeros(len(labels),dtype=torch.bool)
    for j,b in enumerate(brands):
        options=[]
        for gate in (.5,.6,.7,.8,.9,.95,.98):
            eligible=(ids[:,0]==j)&(values[:,0]>=gate)&((values[:,0]-values[:,1])>=.06)&geometry
            if len(groups[eligible].unique())>=3 and float((labels[eligible]==j).float().mean())>=.9:options.append((int(eligible.sum()),gate))
        _,threshold=max(options,default=(0,1.));thresholds[b]=threshold
        if threshold<1:accepted|=(ids[:,0]==j)&(values[:,0]>=threshold)&((values[:,0]-values[:,1])>=.06)&geometry
    precision=float((ids[accepted,0]==labels[accepted]).float().mean()) if accepted.any() else 0.;priority_enabled={b:thresholds[b]<1 for b in baseline['policy']['priority_makes']}
    passed=best['scheme']!='mask' and all(priority_enabled.values()) and int(accepted.sum())>=40 and precision>=.9
    report=dict(passed=passed,selected={k:v for k,v in best.items() if k!='p'},candidates=records,priority_enabled=priority_enabled,validation_confirmed_views=int(accepted.sum()),validation_confirmed_precision=precision,source_model_sha256=source_sha,gradient_training_rounds=0,test_used_for_selection=False,user_controls_used=False,seconds=time.time()-started)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('VIEW CALIBRATION COMPLETE',json.dumps(report),flush=True)
    if passed:
        result=dict(ck,view_policy=best['scheme'],temperature=best['temperature'],thresholds=thresholds,calibration_source_model_sha256=source_sha)
        torch.save(result,out/'semantic_make.pt')
    else:(out/'NO_CHANGE.txt').write_text('No validation-approved preprocessing improvement. Keep the previous fixed model; do not add more searches in this calibration phase.',encoding='utf-8')
