"""Prepare licensed new sessions without changing historical split membership."""
import sys,json,hashlib,collections,re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image,ImageDraw
from carvision.core import ROOT,read_image
from carvision.segmentation import InstanceSegmenter,tight_box
from tools.prepare_ev_training import dhash,group

torch.set_num_threads(3)
out=ROOT/'models/ev_patch_v22';out.mkdir(exist_ok=True)
cropcache=ROOT/'cache/patch_v22_crops';cropcache.mkdir(exist_ok=True)
base=torch.load(ROOT/'cache/body_expert_v20.pt',weights_only=True)
pairs=[tuple(p) for p in base['pairs']];rows=base['rows']
oldmeta=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
oldmeta+=json.loads((ROOT/'models/ev_parts_candidate/dataset.json').read_text(encoding='utf-8'))['new_rows']
locks=collections.defaultdict(set)
for r in oldmeta:locks[r['group']].add(r['split'])
hashes=[dhash(Image.fromarray(r['body_rgb'].numpy())) for r in rows]
seen_sources={r.get('source',{}).get('source','') for r in oldmeta if isinstance(r.get('source'),dict)}
segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True)
accepted=[];rejected=collections.Counter()
review_file=out/'review_exclusions.json'
review=json.loads(review_file.read_text(encoding='utf-8')) if review_file.exists() else {}
prior_review=json.loads((ROOT/'models/ev_annotation_review.json').read_text(encoding='utf-8'))['excluded_sources']
for folder in ['ev_patch_v22','ev_patch_flickr_v22']:
    source=ROOT/'datasets'/folder;manifest=source/'manifest.json'
    if not manifest.exists():continue
    for n,r in enumerate(json.loads(manifest.read_text(encoding='utf-8'))):
        if r['source'] in prior_review:rejected['prior_manual_review']+=1;continue
        # Dolphin Mini is the regional Seagull name, not the larger Dolphin.
        if r['brand']=='BYD' and re.search(r'dolphin\s+mini',r['title'],re.I):
            r=dict(r,model='Seagull',taxonomy_note='Source title Dolphin Mini; regional name of Seagull family')
        if folder+'/'+r['image'] in review:rejected['manual_visual_review']+=1;continue
        if r['source'] in seen_sources:rejected['source_duplicate']+=1;continue
        seen_sources.add(r['source'])
        g=group(r)
        if len(locks[g])>1:rejected['historical_group_conflict']+=1;continue
        if re.search(r'interior|engine|dashboard|concept|render|scale model',r['title'],re.I):rejected['non_exterior']+=1;continue
        path=source/r['image'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==r['sha256'],'Source photo checksum changed'
        cached=cropcache/(digest+'.pt')
        if cached.exists():
            geometry=torch.load(cached,weights_only=True)
            if geometry.get('rejected'):rejected[geometry['rejected']]+=1;continue
            crop=Image.fromarray(geometry['rgb'].numpy());m=geometry['mask'].numpy()
        else:
            im=read_image(path);im.thumbnail((640,640))
            with torch.inference_mode():d=segmenter.raw(im)
            ids=[j for j in range(len(d['scores'])) if int(d['labels'][j])==3 and float(d['scores'][j])>.75]
            ids.sort(key=lambda j:float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])),reverse=True)
            if not ids:
                torch.save(dict(rejected='no_car'),cached);rejected['no_car']+=1;continue
            areas=[float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])) for j in ids]
            if areas[0]<im.width*im.height*.18 or (len(areas)>1 and areas[1]>.5*areas[0]):
                torch.save(dict(rejected='ambiguous_scene'),cached);rejected['ambiguous_scene']+=1;continue
            m=d['masks'][ids[0],0].numpy()>.5;b=tight_box(m)
            if b is None or min(b[2]-b[0],b[3]-b[1])<70:
                torch.save(dict(rejected='small'),cached);rejected['small']+=1;continue
            crop=im.crop(b);m=m[b[1]:b[3],b[0]:b[2]]
            torch.save(dict(rgb=torch.from_numpy(np.asarray(crop).copy()),mask=torch.from_numpy(m.copy())),cached)
        h=dhash(crop)
        if any((h^v).bit_count()<=5 for v in hashes):rejected['near_duplicate']+=1;continue
        hashes.append(h)
        crop.thumbnail((320,320));m=np.asarray(Image.fromarray(m).resize(crop.size,Image.Resampling.NEAREST)).copy()
        accepted.append(dict(image=folder+'/'+r['image'],pair=(r['brand'],r['model']),group=g,source=r,split=next(iter(locks[g])) if locks[g] else None,body_rgb=torch.from_numpy(np.asarray(crop).copy()),body_mask=torch.from_numpy(m)))
        print('ACCEPT',len(accepted),r['title'],flush=True)
# Review sheets carry explicit source labels; screenshots/user diagnostics are absent.
for first in range(0,len(accepted),40):
    sheet=Image.new('RGB',(1000,8*140),'white');draw=ImageDraw.Draw(sheet)
    for idx,r in enumerate(accepted[first:first+40],first):
        im=Image.fromarray(r['body_rgb'].numpy());im.thumbnail((190,105));x=((idx-first)%5)*200;y=((idx-first)//5)*140
        sheet.paste(im,(x,y));draw.text((x,y+106),str(idx)+' '+' '.join(r['pair']),fill='black')
    sheet.save(out/f'review_{first//40}.jpg')
# Group-stratify new sessions before any training/evaluation. Reuse every old lock.
# A pure hash split can leave rare classes with no test sessions at all.
for pair in sorted({r['pair'] for r in accepted},key=lambda p:sum(r['pair']==p for r in accepted)):
    groups=sorted({r['group'] for r in accepted if r['pair']==pair},key=lambda g:hashlib.sha256(('patch22:'+g).encode()).hexdigest())
    counts=collections.Counter(next(iter(locks[g])) for g in groups if locks[g])
    for g in groups:
        if locks[g]:continue
        split=max(['train','validation','test'],key=lambda s:dict(train=.6,validation=.2,test=.2)[s]*len(groups)-counts[s])
        locks[g]={split};counts[split]+=1
for r in accepted:r['split']=next(iter(locks[r['group']]))
eligible=[]
for pair in sorted({r['pair'] for r in accepted}):
    subset=[r for r in accepted if r['pair']==pair];counts=collections.Counter(r['split'] for r in subset)
    if pair not in pairs and (len({r['group'] for r in subset})<5 or min(counts.get(s,0) for s in ['train','validation','test'])<1 or counts['train']<4):
        rejected['insufficient_independent_class']+=len(subset);continue
    eligible+=subset
newpairs=sorted({r['pair'] for r in eligible}-set(pairs));pairs+=newpairs
for r in eligible:r['y']=pairs.index(r['pair'])
expanded=rows+eligible
owner_rows=[]
owner=ROOT/'data/own_oct21'
for r in [json.loads(s) for s in (owner/'labels.jsonl').read_text(encoding='utf-8').splitlines()]:
    pair=(r['brand'],r.get('model'))
    if pair not in pairs:continue
    im=read_image(owner/r['image']).crop(r['box']);im.thumbnail((320,320))
    with torch.inference_mode():prediction=segmenter.raw(im)
    ids=[i for i in range(len(prediction['scores'])) if int(prediction['labels'][i])==3 and float(prediction['scores'][i])>.7]
    if ids:
        best=max(ids,key=lambda i:float(prediction['masks'][i,0].sum()))
        mask=prediction['masks'][best,0].numpy()>.5
    else:mask=np.ones((im.height,im.width),dtype=bool)
    owner_rows.append(dict(image='own_oct21/'+r['image']+'#'+str(r['box']),split='train',group=r['group'],y=pairs.index(pair),body_rgb=torch.from_numpy(np.asarray(im).copy()),body_mask=torch.from_numpy(mask.copy()),source=dict(annotation=r,license='User supplied; explicitly authorized for local project training',sha256=hashlib.sha256((owner/r['image']).read_bytes()).hexdigest())))
expanded+=owner_rows
assert all(len(v)==1 for g,v in locks.items() if any(r['group']==g for r in eligible))
torch.save(dict(rows=expanded,pairs=pairs),ROOT/'cache/ev_patch_v22.pt')
metadata=[{k:v for k,v in r.items() if k not in ['body_rgb','body_mask']} for r in accepted]
info=dict(pairs=pairs,new_pairs=newpairs,new_rows=metadata,eligible_images=[r['image'] for r in eligible],owner_training_rows=[{k:v for k,v in r.items() if k not in ['body_rgb','body_mask']} for r in owner_rows],rejected=dict(rejected),counts={b+' '+m:{s:sum(r['y']==i and r['split']==s for r in expanded) for s in ['train','validation','test']} for i,(b,m) in enumerate(pairs)},historical_splits_unchanged=True,source_groups='author/date session; physical vehicle identities unavailable',new_night_diagnostic_used=False,previous_owner_outing_train_only=True)
(out/'dataset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',json.dumps(info['counts']),flush=True)
