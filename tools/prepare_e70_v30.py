"""Prepare manually reviewed E70 sources; split author/date before fitting."""
import hashlib,json,sys,re,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import read_image
from carvision.segmentation import InstanceSegmenter,tight_box,isolate
from carvision.semantic_make import GeneralVisionEncoder,visual_inputs
from carvision.dino_features import DinoEncoder,dino_input,descriptor

def main():
    torch.set_num_threads(3);folder=ROOT/'datasets/e70_truth_v30';out=ROOT/'models/truth_v30';out.mkdir(exist_ok=True);cache=ROOT/'cache/truth_v30';meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    groups=sorted({r['author']+'|'+r['date'][:10] for r in meta},key=lambda g:hashlib.sha256(g.encode()).hexdigest());assert len(groups)>=8
    split={g:'validation' if i<3 else 'test' if i<5 else 'train' for i,g in enumerate(groups)}
    (out/'e70_source_review.json').write_text(json.dumps(dict(reviewed_indices=list(range(len(meta))),basis='All13 source photographs visually reviewed on contact sheet; consistent E70 exterior, source descriptions and author identities preserved. Own first sedan independently compared by badge/grille/lights. No predicted labels used.',groups=split,owner_first_two='Known error diagnostics only; not added to training or validation in v30.'),ensure_ascii=False,indent=2),encoding='utf-8')
    segmenter=InstanceSegmenter(torch.load(ROOT/'outputs/CarVision-2.9-Test/CarVision/models/segmenter.pt',weights_only=True));rows=[]
    for i,r in enumerate(meta):
        im=read_image(folder/r['image']);work=im.copy();work.thumbnail((1600,1600));records,_=segmenter.instances(work,.5,'fast');cars=[x for x in records if x.get('label')==3];assert cars,(i,r['title']);target=max(cars,key=lambda x:int(x['mask'].sum()));box=tight_box(target['mask']);crop=work.crop(box);mask=target['mask'][box[1]:box[3],box[0]:box[2]];crop.thumbnail((640,640));mask=np.asarray(Image.fromarray(mask).resize(crop.size,Image.Resampling.NEAREST)).copy();group=r['author']+'|'+r['date'][:10]
        rows.append(dict(image='e70_truth_v30/'+r['image'],brand='Dongfeng',family='Aeolus E70',group='e70:'+group,split=split[group],body_rgb=torch.from_numpy(np.asarray(crop).copy()),body_mask=torch.from_numpy(mask),source=r));print('CROP',i,split[group],flush=True)
    torch.save(dict(rows=rows),cache/'e70_crops.pt');sheet=Image.new('RGB',(1200,300*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        im=ImageOps.contain(isolate(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy()),(295,260));x=i%4*300;y=i//4*300;sheet.paste(im,(x,y+25));draw.text((x+3,y+3),f"{i} Dongfeng E70 {r['split']}",fill='black')
    sheet.save(out/'e70_crop_review.jpg',quality=95)
    del segmenter
    clip=GeneralVisionEncoder().eval();clip.load_state_dict(torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True)['encoder_state']);dino=DinoEncoder().eval();dino.load_state_dict(torch.load(ROOT/'cache/dino_v28/dinov2_vits14_pretrain.pth',weights_only=True));features=[]
    with torch.inference_mode():
        for k in ('whole','phone'):
            for r in rows:
                im=Image.fromarray(r['body_rgb'].numpy());mask=r['body_mask'].numpy();c=torch.nn.functional.normalize(clip(visual_inputs(im,mask,k)[:1])[0],dim=0);d=descriptor(dino(dino_input(im,mask,k).unsqueeze(0)))[0];features.append(torch.nn.functional.normalize(torch.cat([d,c]),dim=0))
    torch.save(dict(x=torch.stack(features)),cache/'e70_features.pt');(out/'e70_dataset.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('body_rgb','body_mask')} for r in rows],ensure_ascii=False,indent=2),encoding='utf-8');print('E70 FEATURES COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
