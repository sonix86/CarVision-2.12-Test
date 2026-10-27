"""Prepare a reviewed, grouped make corpus; every historical crop stays immutable."""
import collections,hashlib,json,re,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from torchvision.ops import box_iou
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import read_image
from carvision.segmentation import InstanceSegmenter,tight_box
from tools.prepare_ev_training import dhash,group
from tools.make_taxonomy_v23 import canonical_make

torch.set_num_threads(3);OUT=ROOT/'models/make_scope_v23';OUT.mkdir(exist_ok=True)
CACHE=ROOT/'cache/make_scope_v23_crops';CACHE.mkdir(exist_ok=True)
base=torch.load(ROOT/'cache/ev_patch_v22.pt',weights_only=True);rows=[]
metadata=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
metadata+=json.loads((ROOT/'models/ev_parts_candidate/dataset.json').read_text(encoding='utf-8'))['new_rows']
metadata+=json.loads((ROOT/'models/ev_patch_v22/dataset.json').read_text(encoding='utf-8'))['new_rows']
lookup={r['image']:r for r in metadata};locks=collections.defaultdict(set)
for r in metadata:locks[r['group']].add(r['split'])
def source_group(row):
    if re.sub(r'[^0-9]','',row.get('date','')):return group(row)
    author=row.get('author','').strip()
    if not author:return None
    return 'undated:'+hashlib.sha256(author.casefold().encode()).hexdigest()[:20]
for row in metadata:
    source=row.get('source') if isinstance(row.get('source'),dict) else row
    g=source_group(source)
    if g and g.startswith('undated:'):locks[g].add(row['split'])
for r in base['rows']:
    brand,model=base['pairs'][r['y']]
    g=r.get('group',lookup.get(r['image'],{}).get('group','historical:'+r['image']))
    rows.append(dict(image=r['image'],brand=brand,family=model,group=g,split=r['split'],
                     body_rgb=r['body_rgb'],body_mask=r['body_mask'],historical=True,
                     source=r.get('source',lookup.get(r['image'],dict(source='Historical project corpus; original source metadata retained in the full project',image=r['image'])))))
hashes=[dhash(Image.fromarray(r['body_rgb'].numpy())) for r in rows]
segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True)
reviewfile=OUT/'review_exclusions.json';review=json.loads(reviewfile.read_text(encoding='utf-8')) if reviewfile.exists() else {}
relabel_file=OUT/'review_relabels.json';relabels=json.loads(relabel_file.read_text(encoding='utf-8')) if relabel_file.exists() else {}
manual_file=OUT/'manual_boxes.json';manual=json.loads(manual_file.read_text(encoding='utf-8')) if manual_file.exists() else {}
manifests={folder:ROOT/'datasets'/folder/'manifest.json' for folder in ['make_scope_v23','make_flickr_v23']}
source=[];source_hashes={}
for folder,manifest in manifests.items():
    if manifest.exists():
        raw=manifest.read_bytes();source_hashes[folder]=hashlib.sha256(raw).hexdigest();source += [(folder,r) for r in json.loads(raw)]
manifest_sha=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
accepted=[];rejected=collections.Counter()
for i,(folder,r) in enumerate(source):
    image=folder+'/'+r['image']
    if image in review:rejected['manual_review']+=1;continue
    if re.search(r'retouched|changed numberplates|IMG\d+crop|\bmod\b|bearb\s*Sp',r['title'],re.I):
        rejected['edited_duplicate_risk']+=1;continue
    if re.search(r'interior|engine|dashboard|concept|render|scale model|diecast|\btoy\b',r['title'],re.I):rejected['not_exterior']+=1;continue
    if re.search(r'die.?cast|scale model|miniature|\b1:(?:18|24|43|64|87)\b|汽车模型|概念车',r.get('description',''),re.I):rejected['model_or_concept_metadata']+=1;continue
    g=source_group(r)
    if g is None:rejected['untracked_source_metadata']+=1;continue
    if len(locks[g])>1:rejected['historical_group_conflict']+=1;continue
    path=ROOT/'datasets'/image;digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==r['sha256']
    cache=CACHE/(digest+'.pt')
    cached=torch.load(cache,weights_only=True) if cache.exists() else None
    if image in manual:
        im=read_image(path);box=manual[image]['box'];assert len(box)==4 and 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height
        crop=im.crop(box);crop.thumbnail((320,320))
        geometry=dict(rgb=torch.from_numpy(np.asarray(crop).copy()),mask=torch.ones(crop.height,crop.width,dtype=torch.bool))
    elif cached is not None and (not cached.get('rejected') or (cached.get('vehicle_labels')==[3,8] and cached.get('proposal_dedup'))):geometry=cached
    else:
        im=read_image(path);im.thumbnail((640,640))
        with torch.inference_mode():d=segmenter.raw(im)
        ids=[j for j in range(len(d['scores'])) if int(d['labels'][j]) in (3,8) and float(d['scores'][j])>.8
             and not (int(d['labels'][j])==8 and float(d['boxes'][j,2]-d['boxes'][j,0])/max(1,float(d['boxes'][j,3]-d['boxes'][j,1]))>3.5)]
        distinct=[]
        for j in sorted(ids,key=lambda j:float(d['scores'][j]),reverse=True):
            if any(int(d['labels'][j])!=int(d['labels'][k]) and float(box_iou(d['boxes'][j:j+1],d['boxes'][k:k+1])[0,0])>.7 for k in distinct):continue
            distinct.append(j)
        ids=distinct
        ids.sort(key=lambda j:float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])),reverse=True)
        geometry=dict(rejected='no_dominant_car')
        if ids:
            areas=[float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])) for j in ids]
            if areas[0]>=im.width*im.height*.2 and (len(areas)<2 or areas[1]<areas[0]*.5):
                m=d['masks'][ids[0],0].numpy()>.5;b=tight_box(m)
                if b and min(b[2]-b[0],b[3]-b[1])>=70:
                    crop=im.crop(b);mask=m[b[1]:b[3],b[0]:b[2]];crop.thumbnail((320,320));mask=np.asarray(Image.fromarray(mask).resize(crop.size,Image.Resampling.NEAREST)).copy()
                    geometry=dict(rgb=torch.from_numpy(np.asarray(crop).copy()),mask=torch.from_numpy(mask))
        geometry.update(vehicle_labels=[3,8],proposal_dedup=True);torch.save(geometry,cache)
    if geometry.get('rejected'):rejected[geometry['rejected']]+=1;continue
    h=dhash(Image.fromarray(geometry['rgb'].numpy()))
    if any((h^v).bit_count()<=5 for v in hashes):rejected['near_duplicate']+=1;continue
    hashes.append(h)
    corrected=relabels.get(image,{})
    accepted.append(dict(image=image,brand=canonical_make(corrected.get('brand',r['brand'])),requested_brand=r['brand'],family=corrected.get('family',r['requested_family']),group=g,source=r,manual_label_review=corrected or None,
                         body_rgb=geometry['rgb'],body_mask=geometry['mask'],historical=False))
    if i%50==0:print('PREPARED',i,len(source),'accepted',len(accepted),flush=True)
# Only absent legacy makes are supplemented from exact Stanford vehicle annotations.
# These makes were not in the body expert's training taxonomy. No colour pseudo-label is used.
missing={'Audi','Ford','Mercedes-Benz','Volkswagen','Volvo','Tesla'}
supplement=[];public=ROOT/'data/public_prototype'
for r in [json.loads(s) for s in (public/'labels.jsonl').read_text(encoding='utf-8').splitlines()]:
    if r['brand'] not in missing:continue
    token=int(hashlib.sha256(('make23:'+r['group']).encode()).hexdigest()[:8],16)%10
    split='train' if token<6 else 'validation' if token<8 else 'test'
    supplement.append((hashlib.sha256(r['group'].encode()).hexdigest(),split,r))
counts=collections.Counter()
for _,split,r in sorted(supplement):
    key=(r['brand'],split)
    if counts[key]>=dict(train=60,validation=20,test=30)[split]:continue
    im=read_image(public/r['image']).crop(r['box']);im.thumbnail((320,320));h=dhash(im)
    if any((h^v).bit_count()<=5 for v in hashes):continue
    hashes.append(h);counts[key]+=1
    rows.append(dict(image='public_prototype/'+r['image'],brand=r['brand'],family=None,group=r['group'],split=split,
                     body_rgb=torch.from_numpy(np.asarray(im).copy()),body_mask=torch.ones(im.height,im.width,dtype=torch.bool),
                     historical=False,source=dict(source=r['source'],original_sha256=r['original_sha256'],annotation='Original Stanford brand and box only; colour labels ignored')))
# Complete author/date sessions receive one immutable split, across makes/models.
for brand in sorted({r['brand'] for r in accepted},key=lambda b:sum(r['brand']==b for r in accepted)):
    groups=sorted({r['group'] for r in accepted if r['brand']==brand},key=lambda g:hashlib.sha256(('make23:'+g).encode()).hexdigest())
    if len(groups)<3:
        for g in groups:
            if not locks[g]:locks[g]={'train'}
    counts=collections.Counter(next(iter(locks[g])) for g in groups if locks[g])
    for g in groups:
        if locks[g]:continue
        split=max(['train','validation','test'],key=lambda s:dict(train=.55,validation=.25,test=.2)[s]*len(groups)-counts[s])
        locks[g]={split};counts[split]+=1
for r in accepted:r['split']=next(iter(locks[r['group']]))
probe_file=OUT/'family_probe.json'
if probe_file.exists():
    probe=json.loads(probe_file.read_text(encoding='utf-8'))
    assert probe['source_manifest_sha256']==manifest_sha
    reserved=set(probe['groups'])
    assert not reserved&{r['group'] for r in rows},'New-family probe must not overlap historical or supplementary rows'
    for r in accepted:
        if r['group'] in reserved:r['split']='family_test'
for first in range(0,len(accepted),40):
    sheet=Image.new('RGB',(1000,1120),'white');draw=ImageDraw.Draw(sheet)
    for idx,r in enumerate(accepted[first:first+40],first):
        im=Image.fromarray(r['body_rgb'].numpy());im.thumbnail((190,103));x=(idx-first)%5*200;y=(idx-first)//5*140
        sheet.paste(im,(x,y));draw.text((x,y+104),str(idx)+' '+r['brand'],fill='black');draw.text((x,y+119),r['family'],fill='black')
    sheet.save(OUT/f'review_{first//40}.jpg')
rows+=accepted;brands=sorted({r['brand'] for r in rows});eligible=[];excluded_brands=[]
for brand in brands:
    subset=[r for r in rows if r['brand']==brand];counts=collections.Counter(r['split'] for r in subset)
    if counts['train']<2:
        excluded_brands.append(brand);continue
    eligible+=subset
brands=sorted({r['brand'] for r in eligible})
for r in eligible:r['y']=brands.index(r['brand'])
groups=collections.defaultdict(set)
for r in eligible:groups[r['group']].add(r['split'])
assert not any(len(v)>1 for v in groups.values())
torch.save(dict(rows=eligible,brands=brands),ROOT/'cache/make_scope_v23.pt')
meta=lambda r:{k:v for k,v in r.items() if k not in ['body_rgb','body_mask']}
info=dict(brands=brands,counts={b:dict(collections.Counter(r['split'] for r in eligible if r['brand']==b)) for b in brands},
          provisional_train_only_brands=[b for b in brands if not any(r['brand']==b and r['split']=='validation' for r in eligible) or not any(r['brand']==b and r['split'] in ['test','family_test'] for r in eligible)],
          source_manifest_sha256=manifest_sha,source_manifest_hashes=source_hashes,
          new_rows=[meta(r) for r in accepted],excluded_brands=excluded_brands,rejected=dict(rejected),reviewed=False,
          historical_prefix_crops_preserved=True,models_added_to_model_head=0,
          new_session_target_fractions=dict(train=.55,validation=.25,test=.2),
          supplementary_public_split='Predeclared SHA groups on absent makes only; not the official Stanford benchmark split.',
          limitations='Author/date is approximate session identity; undated photos grouped by author. Rare makes with two or more training photos may have no heldout evidence and must never be confirmed by the calibrated gate. New family names are source metadata, not certified model labels.')
(OUT/'dataset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',json.dumps(info['counts']),flush=True)
