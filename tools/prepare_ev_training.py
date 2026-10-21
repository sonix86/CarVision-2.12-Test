"""Prepare independently category-labelled EV photos; preserve all previous splits.

The generic segmenter proposes the crop, never its make/model label. Ambiguous
multi-car scenes are excluded. Public scenes remain unlabelled review material.
"""
import collections,hashlib,json,re,sys,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image
from carvision.core import ROOT,read_image
from carvision.segmentation import InstanceSegmenter,tight_box
from tools.ev_group_audit import excluded_legacy_images

def dhash(im):
 a=np.asarray(im.resize((9,8)).convert('L'));return int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big')

def group(row):
 date=re.sub(r'[^0-9]','',row.get('date',''))[:8]
 session=(row['author'],date or re.sub(r'\d+|front|rear|left|right','',row['title'].lower()))
 return hashlib.sha256(str(session).encode()).hexdigest()[:20]

def main():
 torch.set_num_threads(3);src=ROOT/'datasets/ev_commons_v19';out=ROOT/'models/ev_parts_candidate';out.mkdir(exist_ok=True)
 manifest=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
 review=json.loads((ROOT/'models/ev_annotation_review.json').read_text(encoding='utf-8'))
 extra=src/'thumb_data/manifest.json'
 if extra.exists():manifest+=json.loads(extra.read_text(encoding='utf-8'))
 previous=[json.loads(l) for l in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
 previous_groups=collections.defaultdict(set)
 for r in previous:previous_groups[r['group']].add(r['split'])
 locked={g:next(iter(s)) for g,s in previous_groups.items() if len(s)==1};conflicting={g for g,s in previous_groups.items() if len(s)>1};hashes=[int(r['dhash'],16) for r in previous]
 demo=[dhash(read_image(p)) for p in (ROOT/'samples/images').glob('*.jpg')]
 old=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);old_pairs=list(zip(old['brands'],old['labels']))
 segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True);rows=[];rejected=collections.Counter()
 with torch.inference_mode():
  for i,row in enumerate(manifest):
   if row['source'] in review['excluded_sources']:rejected['manual_annotation_review']+=1;continue
   if row['brand']=='Haval' and row['model']=='H6 III':row=dict(row,model='H6')
   if not row['brand']:rejected['unlabelled_scene']+=1;continue
   if group(row) in conflicting:rejected['old_session_split_conflict']+=1;continue
   im=read_image(src/row['image']);h=dhash(im)
   if any((h^v).bit_count()<=8 for v in demo):rejected['city_example_similarity']+=1;continue
   im.thumbnail((384,384));d=segmenter.raw(im)
   ids=[j for j in range(len(d['scores'])) if int(d['labels'][j]) in (3,8) and float(d['scores'][j])>.7]
   ids.sort(key=lambda j:float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])),reverse=True)
   if not ids:rejected['no_confident_car']+=1;continue
   boxes=d['boxes'];areas=[float((boxes[j,2]-boxes[j,0])*(boxes[j,3]-boxes[j,1])) for j in ids]
   if areas[0]<im.width*im.height*.15 or (len(areas)>1 and areas[1]>.55*areas[0]):rejected['multiple_or_small_cars']+=1;continue
   mask=d['masks'][ids[0],0].numpy()>.5;box=tight_box(mask)
   if not box or min(box[2]-box[0],box[3]-box[1])<60:rejected['too_small_mask']+=1;continue
   crop=im.crop(box);dh=dhash(crop)
   if any((dh^v).bit_count()<=5 for v in hashes):rejected['near_duplicate']+=1;continue
   hashes.append(dh)
   m=mask[box[1]:box[3],box[0]:box[2]]
   rows.append(dict(image='ev_v19/'+row['image'],brand=row['brand'],model=row['model'],group=group(row),ratio=crop.width/crop.height,rgb=torch.from_numpy(np.asarray(crop.resize((176,176),Image.Resampling.BICUBIC)).copy()),mask=torch.from_numpy(np.asarray(Image.fromarray(m).resize((176,176),Image.Resampling.NEAREST)).copy()),source=row,split=locked.get(group(row))))
   if i%20==0:print('PREPARE',i,'/',len(manifest),'accepted',len(rows),flush=True)
 # Fixed split by complete author/date sessions, including sessions already
 # allocated by earlier versions. Do not reshuffle existing heldout photographs.
 eligible=[]
 for pair in sorted({(r['brand'],r['model']) for r in rows}):
  selected=[r for r in rows if (r['brand'],r['model'])==pair];groups=collections.defaultdict(list)
  for r in selected:r['split']=locked.get(r['group']);groups[r['group']].append(r)
  old_class=pair in old_pairs
  if not old_class and (len(selected)<12 or len(groups)<5):rejected['insufficient_class_groups']+=len(selected);continue
  counts=collections.Counter(r['split'] for r in selected if r['split']);units=list(groups.values());random.Random(321019).shuffle(units)
  for unit in sorted(units,key=len,reverse=True):
   if unit[0]['split']:continue
   s=max(['train','validation','test'],key=lambda s:dict(train=.7,validation=.15,test=.15)[s]*len(selected)-counts[s]);counts[s]+=len(unit)
   for r in unit:r['split']=s
   locked[unit[0]['group']]=s
  if not old_class and any(counts[s]==0 for s in ['train','validation','test']):rejected['missing_split']+=len(selected);continue
  eligible+=selected
 pairs=old_pairs+sorted({(r['brand'],r['model']) for r in eligible}-set(old_pairs));prepared=torch.load(ROOT/'cache/expanded_parts_v18.pt',weights_only=True)['rows']
 excluded_old,group_audit=excluded_legacy_images()
 expanded=[dict(r,legacy=True) for r in prepared if r['image'] not in excluded_old]
 for r in eligible:expanded.append({k:r[k] for k in ['image','rgb','mask','split','ratio']}|dict(y=pairs.index((r['brand'],r['model'])),legacy=False))
 data=[{k:v for k,v in r.items() if k not in ['rgb','mask']} for r in eligible]
 info=dict(pairs=pairs,new_rows=data,excluded=dict(rejected),group_audit=group_audit,counts={b+' '+m:{s:sum(r['y']==j and r['split']==s for r in expanded) for s in ['train','validation','test']} for j,(b,m) in enumerate(pairs)},source_model_sha256=hashlib.sha256((ROOT/'models/vehicle_model.pt').read_bytes()).hexdigest(),split='Original files and splits immutable; eight heldout images from seven conflicting cross-brand sessions excluded from new evaluation, never moved to train. New author/date groups locked against existing sessions; whole-scene/crop near-duplicates removed. True vehicle identities unavailable.',own_student_photos=0,city_examples_used=False)
 (out/'dataset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
 torch.save(dict(rows=expanded,pairs=pairs),ROOT/'cache/ev_parts_v19.pt')
 print('COMPLETE',json.dumps(info['counts']),dict(rejected),flush=True)
if __name__=='__main__':main()
