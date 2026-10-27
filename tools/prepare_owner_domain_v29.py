"""Reviewed owner labels and inference-identical crops, TRAIN-only augmentation.

Previously held development errors explicitly become TRAIN in this new phase.
Their later fit is never reported as an independent accuracy estimate.
"""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import read_image
from carvision.segmentation import decode_mask,isolate

torch.set_num_threads(3);out=ROOT/'models/owner_domain_v29';out.mkdir(exist_ok=True)
audit=ROOT/'outputs/mashinki-v29';run=json.loads((audit/'summary.json').read_text(encoding='utf-8'));assert run['complete'] and run['processed']==99
manifest=json.loads((ROOT/'data/owner_oct25/manifest.json').read_text(encoding='utf-8'));by_name={Path(r['source']).name:r for r in manifest}
by_sha={r['sha256']:r for r in manifest}
for current in json.loads((audit/'manifest.json').read_text(encoding='utf-8')):
    # A user renamed photo1 to11223344; identify the source by pixels, not name.
    prior=by_sha[current['sha256']];by_name[Path(current['source']).name]=dict(prior,source=current['source'])
base=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True);brands=base['brands'];old_labels=[json.loads(s) for s in (ROOT/'data/owner_oct25/labels.jsonl').read_text(encoding='utf-8').splitlines()]+json.loads((ROOT/'models/ev_views_v26/owner_additions.json').read_text(encoding='utf-8'))
existing_images={r['image'] for r in base['rows'] if r['image'].startswith('owner_oct25/')}
labels=[dict(r,provenance='Existing manually reviewed own annotation; crop refreshed to match runtime') for r in old_labels if 'owner_oct25/'+r['image'] in existing_images]
for label in labels:
    if Path(label['source']).name=='photo_21_2026-10-25_20-44-40.jpg':
        label.pop('box',None);label['point']=(850,600);label['provenance']='Corrected geometry after full-resolution review: labelled Kia foreground, not neighbouring Chevrolet'
    if Path(label['source']).name=='photo_22_2026-10-25_20-44-40.jpg':
        label['brand']='Chevrolet';label['provenance']='CORRECTION: previous Kia label was wrong; gold Chevrolet bowtie and Tracker lettering visible in original. Previous files retained.'
new=[(1,'GAC Aion','white',None),(2,'GAC Aion','white',None),(3,'Haval','grey',None),(4,'BYD','white',(600,680)),(5,'BYD','black',None),
     (11,'Dongfeng','white',None),(12,'Dongfeng','white',None),(13,'Dongfeng','white',None),(14,'Dongfeng','white',None),(16,'Toyota','black',None),
     (27,'BYD','grey',(600,650)),(28,'Chevrolet','black',None),(38,'Tesla','white',(540,890)),(39,'Tesla','white',(500,850)),
     (44,'Chevrolet','white',None),(46,'Chevrolet','white',None),(58,'Chevrolet','white',None),
     (72,'Deepal','grey',(450,600)),(74,'Kia','white',(450,520)),(77,'BYD','grey',None),(78,'Kia','white',None),
     (81,'Li Auto','grey',None),(82,'BYD','black',(550,680)),(84,'Leapmotor','white',None),(87,'Deepal','grey',(350,400)),
     (88,'BYD','grey',(500,450)),(91,'BYD','grey',(450,400)),(92,'BMW','blue',(650,250))]
for number,brand,colour,point in new:
    name='11223344.jpg' if number==1 else f'photo_{number}_2026-10-25_20-44-40.jpg';m=by_name[name]
    labels.append(dict(image=m['image'],source=m['source'],original_sha256=m['sha256'],brand=brand,colour=colour,point=point,group='owner-phone-outing-2026-10-25',split='train',provenance='User make/model correction in chat; colour visually reviewed by assistant' if number<=5 else 'Assistant full-resolution visible-emblem review; not independently certified',model='unknown'))

def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=w*h
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

by_source={r['source']:r for r in run['records']};proposals=[];annotations=[]
for label in labels:
    assert label['brand'] in brands;source=Path(label['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==label['original_sha256']
    case=by_source[str(source)];prediction=json.loads((audit/'baseline'/f"{case['index']:03d}.json").read_text(encoding='utf-8'));cars=prediction['cars'];assert cars
    if label.get('box'):target=max(cars,key=lambda r:iou(r['box'],label['box']));assert iou(target['box'],label['box'])>.2
    elif label.get('point'):
        x,y=label['point'];matches=[r for r in cars if r['box'][0]<=x<=r['box'][2] and r['box'][1]<=y<=r['box'][3]]
        assert matches,(source.name,label['point']);target=max(matches,key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]))
    else:target=max(cars,key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]))
    im=read_image(source).crop(target['box']);im.thumbnail((640,640));mask=np.asarray(Image.fromarray(decode_mask(target['segmentation'])).resize(im.size,Image.Resampling.NEAREST)).copy()
    record=dict(label,box=target['box'],source=str(source),role='TRAIN only, same outing group; never independent evaluation',previous_known_error_now_training=source.name in ['11223344.jpg']+[f'photo_{i}_2026-10-25_20-44-40.jpg' for i in (2,3,4,5,11,16,27)])
    annotations.append(record);proposals.append(dict(image='owner_oct25/'+label['image'],brand=label['brand'],colour=label['colour'],family='unknown',group=label['group'],split='train',historical=False,owner=True,y=brands.index(label['brand']),body_rgb=torch.from_numpy(np.asarray(im).copy()),body_mask=torch.from_numpy(mask),source=record))
torch.save(dict(rows=proposals,brands=brands),out/'proposals.pt');(out/'annotations.json').write_text(json.dumps(annotations,indent=2,ensure_ascii=False),encoding='utf-8')
for first in range(0,len(proposals),16):
    sheet=Image.new('RGB',(1200,1100),'white');draw=ImageDraw.Draw(sheet)
    for j,r in enumerate(proposals[first:first+16]):
        image=ImageOps.contain(isolate(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy()),(290,235));x=j%4*300;y=j//4*275;sheet.paste(image,(x+(300-image.width)//2,y+25));draw.text((x+5,y+5),f"{first+j}: {r['brand']} {r['colour']}",fill='black');draw.text((x+5,y+258),Path(r['source']['source']).name.split('_2026')[0],fill='black')
    sheet.save(out/f'review_{first//16:02d}.jpg',quality=95)
report=dict(proposed=len(proposals),replacing=len(existing_images),new=len(proposals)-len(existing_images),brands=dict(collections.Counter(r['brand'] for r in proposals)),all_train=True,independent_accuracy_claim_allowed=False,originals_unchanged=True)
(out/'proposal_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
