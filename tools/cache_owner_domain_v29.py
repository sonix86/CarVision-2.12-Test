"""Freeze reviewed TRAIN corrections and cache aligned generic image features."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageOps,ImageEnhance
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import GeneralVisionEncoder,visual_inputs
from carvision.dino_features import DinoEncoder,dino_input,descriptor
from tools.cache_semantic_v25 import token

OUT=ROOT/'models/owner_domain_v29';CACHE=ROOT/'cache/owner_domain_v29';CACHE.mkdir(exist_ok=True)

def view(row,kind):
    image=Image.fromarray(row['body_rgb'].numpy());mask=row['body_mask'].numpy()
    if kind=='flip':image=ImageOps.mirror(image);mask=mask[:,::-1].copy()
    elif kind=='cast':image=Image.fromarray(np.clip(np.asarray(image,dtype=np.float32)*np.array([.92,1.08,.94])*.7,0,255).astype('uint8'))
    return image,mask,'phone' if kind=='phone' else 'whole'

def main():
    torch.set_num_threads(3);approval=json.loads((OUT/'review_approval.json').read_text());assert hashlib.sha256((OUT/'proposals.pt').read_bytes()).hexdigest()==approval['proposals_sha256']
    frozen=CACHE/'data.pt'
    if not frozen.exists():
        base=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True);p=torch.load(OUT/'proposals.pt',weights_only=True);assert len(approval['reviewed_indices'])==len(p['rows'])
        refreshed=[dict(r,domain_v29=True) for i,r in enumerate(p['rows']) if str(i) not in approval['exclusions']]
        old=[r for r in base['rows'] if r['image'].startswith('owner_oct25/')];rows=[r for r in base['rows'] if not r['image'].startswith('owner_oct25/')]+refreshed
        assert len({r['image'] for r in refreshed})==len(refreshed) and all(r['split']=='train' for r in refreshed)
        groups={s:{r['group'] for r in rows if r['split']==s} for s in ('train','validation','test')};assert groups['train'].isdisjoint(groups['validation']|groups['test']) and groups['validation'].isdisjoint(groups['test'])
        torch.save(dict(rows=rows,brands=base['brands']),frozen)
        (OUT/'dataset.json').write_text(json.dumps(dict(rows=len(rows),replaced_private=len(old),refreshed_private=len(refreshed),new_private=len(refreshed)-len(old),public_records_unchanged=True,all_private_augmentation_train_only=True,known_errors_now_train=True,disputed_GAC_labels_excluded=True,dataset_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    data=torch.load(frozen,weights_only=True);own=[r for r in data['rows'] if r.get('domain_v29')];assert all(r['split']=='train' for r in own)
    path=CACHE/'features.pt';features=torch.load(path,weights_only=True) if path.exists() else dict(clip={},dino={})
    clip=GeneralVisionEncoder().eval();clip.load_state_dict(torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True)['encoder_state']);dino=DinoEncoder().eval();dino.load_state_dict(torch.load(ROOT/'cache/dino_v28/dinov2_vits14_pretrain.pth',weights_only=True))
    jobs=[(r,k) for r in own for k in ('whole','phone','flip','cast') if token(r,k) not in features['clip']];start=time.time()
    with torch.inference_mode():
        for first in range(0,len(jobs),8):
            batch=jobs[first:first+8];views=[view(r,k) for r,k in batch];cx=torch.stack([visual_inputs(im,mask,k)[0] for im,mask,k in views]);dx=torch.stack([dino_input(im,mask,k) for im,mask,k in views]);cz=clip(cx);dz=descriptor(dino(dx))
            for i,(r,k) in enumerate(batch):features['clip'][token(r,k)]=torch.stack([cz[i],cz[i]]).clone();features['dino'][token(r,k)]=dz[i].clone()
            torch.save(features,path);print('OWN FEATURES',first+len(batch),len(jobs),round(time.time()-start,1),flush=True)
    (OUT/'features.json').write_text(json.dumps(dict(complete=True,private_rows=len(own),items=len(features['clip']),augmentation_kinds=['whole','phone','flip','cast'],all_train=True),indent=2),encoding='utf-8')

if __name__=='__main__':main()
