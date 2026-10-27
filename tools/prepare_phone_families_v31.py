"""Verified phone corrections become TRAIN; never count them as blind accuracy."""
import copy,json,hashlib,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cache_owner_domain_v29 import view
from tools.cache_semantic_v25 import token
from carvision.semantic_make import GeneralVisionEncoder,visual_inputs,embedding_features
from carvision.dino_features import DinoEncoder,dino_input,descriptor

def main():
    torch.set_num_threads(3);cache=ROOT/'cache/truth_v31';cache.mkdir(exist_ok=True);out=ROOT/'models/truth_v31';out.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'cache/truth_v30/all_rows.json').read_text(encoding='utf-8'));raw=torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True)['rows'];proposals=torch.load(ROOT/'models/owner_domain_v29/proposals.pt',weights_only=True)['rows'];first=copy.deepcopy(proposals[15]);assert Path(first['source']['source']).name=='11223344.jpg';first.update(brand='Dongfeng',family='Aeolus E70',owner=True,domain_v31=True);first['source'].update(brand='Dongfeng',model='Aeolus E70',previous_assumed_label='GAC Aion, disputed then corrected by independent E70 exterior-source comparison');owner=[]
    for number,family in [(4,'Dolphin'),(5,'Song Plus')]:
        r=next(r for r in raw if r.get('domain_v29') and Path(r.get('source',{}).get('source','')).name==f'photo_{number}_2026-10-25_20-44-40.jpg');r=copy.deepcopy(r);r['family']=family;owner.append(r)
        m=next(m for m in rows if m['image']==r['image']);assert m['split']=='train';m['family']=family;m['phone_family_truth']='User correction + full-original visual/source review; known error becomesTRAIN, not an independent control.'
    owner.append(first);r=first;rows.append({k:v for k,v in r.items() if k not in ('body_rgb','body_mask')});rows[-1].update(shape=list(r['body_rgb'].shape),mask_fraction=float(r['body_mask'].float().mean()),mask_pixels=int(r['body_mask'].sum()),split='train',historical=False)
    base=torch.load(ROOT/'cache/truth_v30/joint.pt',weights_only=True);ex=torch.load(ROOT/'cache/truth_v30/e70_features.pt',weights_only=True)['x'];ne=len(ex)//2;oldn=(len(rows)-1)-ne;oldx=torch.cat([base['x'][:oldn],ex[:ne],base['x'][oldn:],ex[ne:]]);pairs=[tuple(p) for p in base['pairs']]+[('Dongfeng','Aeolus E70')];owncache=torch.load(ROOT/'cache/owner_domain_v29/features.pt',weights_only=True);clip=GeneralVisionEncoder().eval();clip.load_state_dict(torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True)['encoder_state']);dino=DinoEncoder().eval();dino.load_state_dict(torch.load(ROOT/'cache/dino_v28/dinov2_vits14_pretrain.pth',weights_only=True));fx=[];fy=[];firstviews={}
    with torch.inference_mode():
        for r in owner:
            for k in ('whole','phone','flip','cast'):
                if r is first:
                    im,m,kind=view(r,k);c=torch.nn.functional.normalize(clip(visual_inputs(im,m,kind)[:1])[0],dim=0);d=descriptor(dino(dino_input(im,m,kind).unsqueeze(0)))[0];z=torch.nn.functional.normalize(torch.cat([d,c]),dim=0);firstviews[k]=z
                else:z=torch.nn.functional.normalize(torch.cat([owncache['dino'][token(r,k)],embedding_features(owncache['clip'][token(r,k)],False)]),dim=0)
                fx.append(z);fy.append(pairs.index((r['brand'],r['family'])))
    no=len(rows)-1;x=torch.cat([oldx[:no],firstviews['whole'][None],oldx[no:],firstviews['phone'][None]]);torch.save(dict(x=x,owner_x=torch.stack(fx),owner_y=torch.tensor(fy),pairs=pairs,brands=base['brands']),cache/'features.pt');(cache/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    policy=dict(rounds_max=2,epochs_each=100,rounds=[dict(lr=.0001,phone_weight=16),dict(lr=.0002,phone_weight=24)],initial='Fixed own family v30round1epoch49; generic encoders remain frozen. Only Song Plus/Dolphin/E70 classifier vectors adapt; other vectors and logit scale frozen.',private_data='Only reviewed own1E70,4Dolphin,5SongPlus. All sameTRAIN outing. Own2E70 notfitted: known same-vehicle angle diagnostic, not blind independent test.',truth='E70 independent author-labelled/source-visual comparison; SongPlus rear text; Dolphin user correction and full-original visual review. No classifier labels.',selection='Epoch selection uses unchanged public validation rawwhole/phone top1 >=v30parent; private TRAIN mustfit allviews andconfidence>=99%. These constraints areTRAINfit, notaccuracy. No own control used asvalidation. Maximize validation mean rawtop1; no extra rounds.',calibration='Same v30joint routing-aware class probability+absolute similarity calibration on full makevalidation, exactfamilies+other-make negatives; no threshold lowering by hand.',test='One final fixed-candidate reused public test +new E70 split, includingedgeprobe. No retuning after test.',public_splits_unchanged=True,own_cases_now_train=True)
    # A verified make correction must also update its cached numeric class.
    for row in rows:row['y']=base['brands'].index(row['brand'])
    (cache/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'policy.json').open('x',encoding='utf-8') as f:json.dump(policy,f,indent=2)
    (out/'truth_corrections.json').write_text(json.dumps([dict(image=r['image'],brand=r['brand'],family=r['family'],source=r['source'],role='TRAIN correction, not independent accuracy') for r in owner],ensure_ascii=False,indent=2),encoding='utf-8');print('PHONE FAMILY TRAIN CORRECTIONS PREPARED',len(owner),flush=True)
if __name__=='__main__':main()
