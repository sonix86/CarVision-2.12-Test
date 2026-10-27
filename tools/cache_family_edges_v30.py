"""Fixed validation edge probe before fitting; test extraction is gated."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import GeneralVisionEncoder,visual_inputs
from carvision.dino_features import DinoEncoder,dino_input,descriptor
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');a=ap.parse_args();torch.set_num_threads(3);out=ROOT/'models/truth_v30';cache=ROOT/'cache/truth_v30'
    if a.test:assert json.loads((out/'joint_validation.json').read_text())['passed']
    target=cache/('edge_test.pt' if a.test else 'edge_validation.pt');assert not target.exists()
    base=torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True)['rows'];extra=torch.load(cache/'e70_crops.pt',weights_only=True)['rows'];rows=base+extra;pairs=json.loads((ROOT/'outputs/user-error-audit-v30/family_truth_audit.json').read_text())['pairs']+[['Dongfeng','Aeolus E70']];split='test' if a.test else 'validation';ids=[i for i,r in enumerate(rows) if r['split']==split and [r['brand'],r.get('family')] in pairs and not r['image'].startswith('make_scope_v23/')]
    clip=GeneralVisionEncoder().eval();clip.load_state_dict(torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True)['encoder_state']);dino=DinoEncoder().eval();dino.load_state_dict(torch.load(ROOT/'cache/dino_v28/dinov2_vits14_pretrain.pth',weights_only=True));views=[];geometry=[]
    for j,i in enumerate(ids):
        r=rows[i];im=Image.fromarray(r['body_rgb'].numpy());mask=r['body_mask'].numpy();trim=max(1,round(im.width*.12));box=(trim,0,im.width,im.height) if j%2 else (0,0,im.width-trim,im.height);im=im.crop(box);mask=mask[box[1]:box[3],box[0]:box[2]].copy();views.append((im,mask));geometry.append(min(im.size)>=64 and mask.sum()>=1200 and mask.mean()>=.35 and im.width/im.height<=3.2)
    features=[]
    with torch.inference_mode():
        for first in range(0,len(views),8):
            batch=views[first:first+8];c=torch.nn.functional.normalize(clip(torch.stack([visual_inputs(im,m)[:1][0] for im,m in batch])),dim=1);d=descriptor(dino(torch.stack([dino_input(im,m) for im,m in batch])));features.extend(torch.nn.functional.normalize(torch.cat([d,c],dim=1),dim=1));print('EDGE FEATURES',split,first+len(batch),len(views),flush=True)
    torch.save(dict(indices=ids,x=torch.stack(features),geometry=torch.tensor(geometry),probe='Remove12%of one side, alternating sides; fixed before fits. No rotation/flipping chosen by predictions.'),target)
if __name__=='__main__':main()
