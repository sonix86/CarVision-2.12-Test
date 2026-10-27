"""Keep supplied corrections separate from predictions; cache frozen image features."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageOps,ImageEnhance
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.dino_make import DinoMakeExpert
from carvision.segmentation import decode_mask

def main():
    torch.set_num_threads(3);out=ROOT/'models/phone_v32';out.mkdir(exist_ok=True);cache=ROOT/'cache/phone_v32';cache.mkdir(exist_ok=True)
    app=ROOT/'outputs/CarVision-2.10-Test/CarVision';records=[]
    for prefix,items in [
        ('photo_2026-10-27_19-49-23',[(2,'Zeekr','007','grey','User correction and clearly visible ZEEKR rectangular emblem')]),
        ('photo_2026-10-26_15-59-53',[(1,'Leapmotor',None,'black','User correction; original hood emblem LP, compared with official Leapmotor identification sheet'),(2,'Chevrolet','Spark','white','Visible Chevrolet bowtie on original')]),
        ('photo_2026-10-26_15-53-32',[(1,'Voyah',None,'black','Visible Voyah hood emblem; exact model not used'),(2,'Chevrolet','Tracker','white','Readable TRACKER text on rear, original image'),(3,'BYD','Yuan Up','black','Readable BYD and YUAN UP text on rear, original image')])]:
        f=max((app/'outputs').glob(prefix+'*.json'),key=lambda p:p.stat().st_mtime);d=json.loads(f.read_text(encoding='utf-8'))
        for ident,brand,family,colour,source in items:
            c=next(c for c in d['cars'] if c['id']==ident);r=dict(image=d['image'],original_sha256=hashlib.sha256(Path(d['image']).read_bytes()).hexdigest(),box=c['box'],segmentation=c['segmentation'],brand=brand,family=family,colour=colour,label_source=source,role='TRAIN correction after user failure; never independent accuracy',group='owner-front-yard-oct27' if brand=='Zeekr' else 'owner-garage-oct26');records.append(r)
    policy=dict(max_rounds=2,epochs=80,configs=[dict(lr=.00005,own_weight=12),dict(lr=.0001,own_weight=16)],scope='Supervised make-head adaptation only; general encoders and family head frozen. Known new errors become TRAIN. Public validation chooses candidate. Reused public test is development regression, not independent accuracy.',selection='Do not lower confidence gates by hand. Require validation routed precision >= baseline - .002, correct confirmations >= baseline on both normal and phone views; raw top1 >= baseline - .005. Require all original new TRAIN corrections accepted correctly. No further candidate searches after these two.',excluded=[dict(image='C:/Users/Miku/Downloads/photo_2026-10-26_11-51-34.jpg',reason='Visible AIQAR badge; missing from82-class vocabulary. AIQAR Arrizo e official source located. Keep excluded from make training until own class has sufficient source-group validation; do not relabel as Chery/Hyundai from body similarity.')])
    for name,data in [('truth.json',records),('policy.json',policy)]:
        p=out/name
        if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==data
        else:p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    featurefile=cache/'features.pt'
    if featurefile.exists():print('Already cached',flush=True);return
    ck=torch.load(ROOT/'models/truth_v31/scoped_candidate.pt',weights_only=True);clip=torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True);expert=DinoMakeExpert(ck,clip);features=[];labels=[];meta=[]
    with torch.inference_mode():
        for i,r in enumerate(records):
            image=Image.open(r['image']).convert('RGB').crop(r['box']);image.thumbnail((640,640));mask=np.asarray(Image.fromarray(decode_mask(r['segmentation'])).resize(image.size,Image.Resampling.NEAREST))
            for kind in ('whole','phone','flip','dark','bright','neutral'):
                im=image;ma=mask
                if kind=='flip':im=ImageOps.mirror(im);ma=mask[:,::-1].copy()
                elif kind=='phone':
                    size=tuple(max(1,round(x*min(1,96/max(image.size)))) for x in image.size);im=image.resize(size,Image.Resampling.BILINEAR);ma=np.asarray(Image.fromarray(mask).resize(size,Image.Resampling.NEAREST))
                elif kind in ('dark','bright'):im=ImageEnhance.Brightness(image).enhance(.65 if kind=='dark' else 1.25)
                elif kind=='neutral':im=ImageOps.autocontrast(image,cutoff=1)
                features.append(expert.feature(im,ma));labels.append(expert.brands.index(r['brand']));meta.append(dict(index=i,kind=kind,size=list(im.size),brand=r['brand']))
            print('NEW PHONE FEATURES',i+1,len(records),r['brand'],flush=True)
    torch.save(dict(x=torch.stack(features),y=torch.tensor(labels),meta=meta,brands=expert.brands),featurefile)
    print('New known failures cached as TRAIN',len(records),len(features),flush=True)
if __name__=='__main__':main()
