"""Incremental exterior proposals, preserving all historical splits and pixels."""
import collections,hashlib,json,re,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import read_image
from carvision.segmentation import InstanceSegmenter,tight_box,isolate
from carvision.scene import manual_foreground
from tools.prepare_ev_training import dhash,group
torch.set_num_threads(3);out=ROOT/'models/ev_views_v26';out.mkdir(exist_ok=True);cache=ROOT/'cache/ev_views_v26_crops';cache.mkdir(exist_ok=True)
base_path=ROOT/'cache/multimodal_v24_data.pt';data=torch.load(base_path,weights_only=True);base=data['rows'];brands=data['brands'];hashes=[dhash(Image.fromarray(r['body_rgb'].numpy())) for r in base];locks=collections.defaultdict(set)
for r in base:locks[r['group']].add(r['split'])
def source_group(r):
    if re.sub(r'[^0-9]','',r.get('date','')):return group(r)
    return 'undated:'+hashlib.sha256(r['author'].strip().casefold().encode()).hexdigest()[:20] if r.get('author','').strip() else None
source=ROOT/'datasets/ev_views_v26/manifest.json';manifest=json.loads(source.read_text(encoding='utf-8')) if source.exists() else [];proposals=[];rejected={}
network=None
for r in manifest:
    name='ev_views_v26/'+r['image'];g=source_group(r)
    if r['brand'] not in brands or g is None or len(locks[g])>1:rejected[name]='unknown taxonomy/source group conflict';continue
    if re.search(r'interior|engine|dashboard|concept|render|scale model|diecast|\btoy\b|miniature|汽车模型|概念车',r['title']+' '+r.get('description',''),re.I):rejected[name]='not real exterior';continue
    path=ROOT/'datasets'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];target=cache/(r['sha256']+'-full.pt')
    if target.exists():c=torch.load(target,weights_only=True)
    else:
        if network is None:network=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=False)
        im=read_image(path);im.thumbnail((640,640));d=network.raw(im);ids=[j for j in range(len(d['scores'])) if int(d['labels'][j]) in (3,8) and float(d['scores'][j])>=.8];ids.sort(key=lambda j:float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])),reverse=True);c=dict(rejected=True)
        if ids:
            a=[float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])) for j in ids]
            if a[0]>=im.width*im.height*.2 and (len(a)<2 or a[1]<a[0]*.5):
                mask=d['masks'][ids[0],0].numpy()>.5;b=tight_box(mask)
                if b and min(b[2]-b[0],b[3]-b[1])>=70:
                    crop=im.crop(b);crop.thumbnail((320,320));local=np.asarray(Image.fromarray(mask[b[1]:b[3],b[0]:b[2]]).resize(crop.size,Image.Resampling.NEAREST)).copy();c=dict(rgb=torch.from_numpy(np.asarray(crop).copy()),mask=torch.from_numpy(local))
        torch.save(c,target)
    if c.get('rejected'):rejected[name]='no isolated dominant exterior';continue
    h=dhash(Image.fromarray(c['rgb'].numpy()))
    if any((h^old).bit_count()<=5 for old in hashes):rejected[name]='near duplicate';continue
    hashes.append(h);token=int(hashlib.sha256(('ev26:'+g).encode()).hexdigest()[:8],16)%20;split=next(iter(locks[g])) if locks[g] else 'train' if token<14 else 'validation' if token<17 else 'test';locks[g].add(split)
    proposals.append(dict(image=name,brand=r['brand'],family='unknown',requested_family=r['requested_family'],family_verified=False,group=g,split=split,body_rgb=c['rgb'],body_mask=c['mask'],historical=False,y=brands.index(r['brand']),source=r))
for r in json.loads((out/'owner_additions.json').read_text(encoding='utf-8')):
    # Reviewed boxes retain the whole body. GrabCut on dark phone photos
    # removed roofs/doors and kept only a bright bonnet; do not teach that crop.
    image='owner_oct25/'+r['image'];assert all(b['image']!=image for b in base);im=read_image(ROOT/'data'/image).crop(r['box']);im.thumbnail((320,320));m=np.ones((im.height,im.width),dtype=bool)
    proposals.append(dict(image=image,brand=r['brand'],family='unknown',group=r['group'],split='train',body_rgb=torch.from_numpy(np.asarray(im).copy()),body_mask=torch.from_numpy(m),historical=False,y=brands.index(r['brand']),owner=True,colour=r['colour'],source=r))
torch.save(dict(rows=proposals,brands=brands),out/'proposals.pt')
for start in range(0,len(proposals),20):
    sheet=Image.new('RGB',(1000,((min(20,len(proposals)-start)+4)//5)*235),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(proposals[start:start+20]):
        x=(i%5)*200;y=(i//5)*235;im=Image.fromarray(r['body_rgb'].numpy());im=ImageOps.contain(isolate(im,r['body_mask'].numpy()),(195,195));sheet.paste(im,(x+(200-im.width)//2,y+20));draw.text((x+3,y+2),f'{start+i} {r["brand"]}',fill='black');draw.text((x+3,y+216),r['split']+' '+r['image'].split('/')[-1][:12],fill='black')
    sheet.save(out/f'review_{start//20:02d}.jpg',quality=94)
report=dict(source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,base_dataset_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),proposed=len(proposals),rejected=rejected,entries=[dict(index=i,image=r['image'],brand=r['brand'],requested_family=r.get('requested_family'),group=r['group'],split=r['split']) for i,r in enumerate(proposals)],reviewed=False,trained=False)
(out/'proposal_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(proposed=len(proposals),rejected=len(rejected),splits=dict(collections.Counter(r['split'] for r in proposals)))),flush=True)
