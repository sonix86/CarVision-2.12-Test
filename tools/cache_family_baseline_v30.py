"""Measure the actual legacy family head, not a previous classifier's score."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageEnhance,ImageFilter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.vehicle_model import VehicleModelClassifier
from carvision.visual_parts import VisualReferences
from carvision.segmentation import isolate
from carvision.badge_evidence import fuse_badge

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--test',action='store_true');args=parser.parse_args();torch.set_num_threads(3);cache=ROOT/'cache/truth_v30';out=ROOT/'models/truth_v30';assert json.loads((out/'joint_validation.json').read_text())['passed'] if args.test else not (out/'training_once.json').exists();target=cache/('legacy_model_test.pt' if args.test else 'legacy_model_validation.pt');assert not target.exists()
    data=torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True);rows=data['rows'];extra=torch.load(cache/'e70_crops.pt',weights_only=True)['rows'];allrows=rows+extra;metadata=json.loads((cache/'rows.json').read_text(encoding='utf-8'))+json.loads((out/'e70_dataset.json').read_text(encoding='utf-8'))
    for r,m in zip(allrows,metadata):m.update(mask_fraction=float(r['body_mask'].float().mean()),mask_pixels=int(r['body_mask'].sum()),shape=list(r['body_rgb'].shape))
    (cache/'all_rows.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    modeldir=ROOT/'outputs/CarVision-2.9-Test/CarVision/models';ck=torch.load(modeldir/'vehicle_model.pt',weights_only=True);model=VehicleModelClassifier(ck);model.references=VisualReferences(torch.load(modeldir/'visual_references.pt',weights_only=True));pairs=list(zip(model.brands,model.labels))+[('Dongfeng','Aeolus E70')]
    known=lambda r:(r['brand'],r.get('family')) in pairs and not r['image'].startswith('make_scope_v23/')
    selected=[i for i,r in enumerate(allrows) if r['split']==('test' if args.test else 'validation') and known(r)];result={};started=time.time()
    with torch.inference_mode():
        for kind in ('whole','phone'):
            ps=[]
            for j,index in enumerate(selected):
                r=allrows[index]
                if r['brand']=='Dongfeng':p=torch.zeros(len(pairs))
                else:
                    im=Image.fromarray(r['body_rgb'].numpy());mask=r['body_mask'].numpy()
                    if kind=='phone':im.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));im=ImageEnhance.Brightness(im).enhance(.55).filter(ImageFilter.GaussianBlur(.35))
                    p=model.probabilities(im,None,None)*.5+model.probabilities(isolate(im,mask),None,None)*.5
                    if model.badge_evidence is not None:p=fuse_badge(p,list(zip(model.brands,model.labels)),model.badge_evidence.predict(im,mask))
                    p=torch.cat([model.references.calibrated(p),torch.zeros(1)])
                ps.append(p)
                if j%35==0:print('LEGACY FAMILY',kind,j,len(selected),round(time.time()-started,1),flush=True)
            result[kind]=torch.stack(ps)
    torch.save(dict(selected=selected,probabilities=result,pairs=pairs,threshold=float(model.references.threshold)),target);print('LEGACY FAMILY CACHE COMPLETE',len(selected),flush=True)
if __name__=='__main__':main()
