"""Automatic vehicle crops for category-labelled public images; preserve provenance.

Ambiguous multi-brand scenes and small/no-vehicle images are excluded. This is
supplementary data with noisy category labels, not the student's own collection.
"""
import collections,hashlib,json,random,re,sys
from pathlib import Path
import numpy as np,torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_tensor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image

def main():
 torch.set_num_threads(2);root=ROOT/'datasets/regional_commons';source=json.loads((root/'manifest.json').read_text(encoding='utf-8'));out=ROOT/'data/regional';out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
 (out/'source_manifest_snapshot.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
 detector=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None).eval();detector.load_state_dict(torch.load(ROOT/'cache/hub/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',weights_only=True))
 rows=[];excluded=collections.Counter();seen=set();demo_hashes=[]
 for file in (ROOT/'samples/images').glob('*.jpg'):
  a=np.asarray(read_image(file).resize((9,8)).convert('L'));demo_hashes.append(int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big'))
 with torch.inference_mode():
  for start in range(0,len(source),8):
   batch=source[start:start+8];images=[read_image(root/r['image']) for r in batch];detections=detector([to_tensor(im) for im in images])
   for row,image,d in zip(batch,images,detections):
    title=row['title'].lower()
    a=np.asarray(image.resize((9,8)).convert('L'));full_hash=int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big')
    if any((full_hash^h).bit_count()<=8 for h in demo_hashes):excluded['similar_to_development_demo']+=1;continue
    # Labels must identify a single badge family. Rebadged Ravon/Holden/Suzuki are excluded.
    if any(x in title for x in ['ravon','holden','suzuki','pontiac',' & ',' and ','interior','seat','dashboard','door panel','motor','wheel','detail']):excluded['ambiguous_title']+=1;continue
    mask=torch.isin(d['labels'],torch.tensor([3,8]))&(d['scores']>.7);boxes=d['boxes'][mask];scores=d['scores'][mask]
    if not len(boxes):excluded['no_confident_car']+=1;continue
    area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]);order=area.argsort(descending=True);box=boxes[order[0]].tolist()
    if float(area[order[0]])<image.width*image.height*.15 or min(box[2]-box[0],box[3]-box[1])<80:excluded['too_small']+=1;continue
    if len(order)>1 and float(area[order[1]])>.55*float(area[order[0]]):excluded['multiple_comparable_cars']+=1;continue
    crop=image.crop(box);crop.thumbnail((640,640));a=np.asarray(crop.resize((9,8)).convert('L'));dh=int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),'big')
    if any((dh^other).bit_count()<=5 for other in seen):excluded['near_duplicate_crop']+=1;continue
    seen.add(dh);dest=out/'images'/row['image'];crop.save(dest,quality=94)
    session=(row['author'],re.sub(r'[^0-9]','',row['date'])[:8])
    # Same author/date/session stays together. Fallback groups file series when metadata lacks date.
    if not session[1]:session=(row['author'],re.sub(r'\d+|front|rear|left|right','',title))
    rows.append(dict(row,image='images/'+dest.name,box=[0,0,crop.width,crop.height],crop_box=box,dhash=f'{dh:016x}',group=hashlib.sha256(str(session).encode()).hexdigest()[:20],source='public Wikimedia Commons',source_url=row['source']))
   print('PREPARED',start+len(batch),'/',len(source),'accepted',len(rows),flush=True)
 groups=collections.defaultdict(list)
 for row in rows:groups[(row['brand'],row['group'])].append(row)
 brands=sorted({r['brand'] for r in rows});split_counts={}
 for brand in brands:
  units=[v for (b,g),v in groups.items() if b==brand];random.Random(13210).shuffle(units);total=sum(map(len,units));counts=collections.Counter()
  for unit in sorted(units,key=len,reverse=True):
   # Greedy grouped allocation to target ratios, deterministic shuffle breaks ties.
   split=max(['train','validation','test'],key=lambda s:dict(train=.7,validation=.15,test=.15)[s]*total-counts[s])
   for row in unit:row['split']=split
   counts[split]+=len(unit)
  split_counts[brand]=dict(counts)
 (out/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');(out/'provenance.json').write_text(json.dumps(dict(source='Wikimedia Commons explicit brand/model categories plus matching file title; detector crops; no pretrained brand weights',counts=split_counts,excluded=dict(excluded),split='Author/date sessions grouped; crop dHash distance<=5 duplicates removed; true vehicle identities unavailable',six_demo_photos_excluded=True,own_student_photos=0),indent=2),encoding='utf-8');print('COMPLETE',json.dumps(split_counts),flush=True)
if __name__=='__main__':main()
