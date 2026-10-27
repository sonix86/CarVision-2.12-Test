"""Freeze author/date locks and review decisions before any new fit."""
import collections,hashlib,json,re,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'models/truth_v30';cache=ROOT/'cache/truth_v30';assert not (out/'training_once.json').exists()
    old=json.loads((cache/'rows.json').read_text(encoding='utf-8'));locks=collections.defaultdict(set)
    for r in old:locks[r['group']].add(r['split'])
    data=torch.load(cache/'e70_crops.pt',weights_only=True);rows=data['rows'];groups={}
    for r in rows:
        source=r['source'];g=hashlib.sha256(str((source['author'],re.sub('[^0-9]','',source['date'])[:8])).encode()).hexdigest()[:20];assert len(locks[g])<=1;r['group']=g;groups[g]=next(iter(locks[g])) if locks[g] else None
    unlocked=sorted([g for g,v in groups.items() if v is None],key=lambda g:hashlib.sha256(('v30:'+g).encode()).hexdigest());needed=max(0,3-sum(s=='validation' for s in groups.values()))
    for i,g in enumerate(unlocked):groups[g]='validation' if i<needed else 'train' if i==needed else 'test'
    for r in rows:r['split']=groups[r['group']]
    assert sum(v=='validation' for v in groups.values())>=3
    torch.save(data,cache/'e70_crops.pt');(out/'e70_dataset.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('body_rgb','body_mask')} for r in rows],ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'e70_crop_approval.json').write_text(json.dumps(dict(reviewed_indices=list(range(len(rows))),geometry_review='All13 isolated instance crops viewed. E70 dominant vehicle retained. No neighbour selected.',group_rule='Same author/date hash as historical corpus; obey existing split locks. New groups deterministic, minimum3 validation groups. Fix replaces provisional local-prefixed grouping BEFORE any fitting.',groups=groups,split_counts=dict(collections.Counter(r['split'] for r in rows)),features_sha256=hashlib.sha256((cache/'e70_features.pt').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    song=json.loads((ROOT/'outputs/user-error-audit-v30/song_plus_sources.json').read_text(encoding='utf-8'))
    (out/'song_plus_source_review.json').write_text(json.dumps(dict(reviewed_source_images=[r['image'] for r in song],scope='All30 source photos viewed; broad Song Plus family includes petrol, EV and DM-i styling/generations. This is not identical-trim or fuel-type classification. Labels come from original photo authors, not predictions. Own rear Song Plus lettering directly readable; own photo5 excluded from family training.'),indent=2),encoding='utf-8')
    policy=dict(make_fits_max=2,model_fits_max=2,epochs_each=100,make_initial='v29 trained joint DINO+CLIP; encoders frozen',model='New own43-family cosine heads on shared general features; no automotive pretrained classifier',family_truth='Only original42-family corpus with exact source/own labels, excluding make-only provisional make_scope rows; plus reviewed E70. Never use predicted labels.',owner_errors='First2 E70 not fitted; photo5 not fitted for family. Existing make-head owner TRAIN remainsTRAIN. Known diagnostics not blind.',validation='Historical groups immutable; same source author/date locks for new E70. Calibrate make>=95% and family>=95% precision, >=3sourcegroups. Make top1/historical/confirmed guards nonregression vsv29. Model selection validation only; whole/phone top1 and coverage independently reported. Known user diagnostics never used to select epochs.',test='One fixed candidate evaluation if validation passes. Reused public test plus new E70 source groups; not real blind phone exam.',changes='No colour fits in this iteration. No forced top-label/filename/emblem string rules. No changing sealed2.9bundle.',final_quality_goal='No100% guarantee. Exact model and make separately graded; known first-photo failures explicitly included in diagnostic report.')
    with (out/'policy.json').open('x',encoding='utf-8') as f:json.dump(policy,f,indent=2)
    print(json.dumps(dict(groups=groups,counts=dict(collections.Counter(r['split'] for r in rows)))),flush=True)
if __name__=='__main__':main()
