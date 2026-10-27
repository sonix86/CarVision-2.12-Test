"""Propose geometry only for manually identified makes; never infer truth labels."""
import json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.segmentation import InstanceSegmenter,tight_box
torch.set_num_threads(2)
out=ROOT/'data/owner_oct25';manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
# Human readings of clear emblems/exteriors in the selected original photos.
labels={8:('Deepal','black'),9:('BMW','black'),11:('BMW','grey'),12:('Kia','black'),13:('Kia','black'),23:('Chevrolet','black'),54:('BYD','black'),59:('Chevrolet','grey'),69:('Kia','white'),87:('Chevrolet','white')}
segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True)
rows=[];sheet=Image.new('RGB',(1250,700),'white');draw=ImageDraw.Draw(sheet)
for pos,(i,(brand,colour)) in enumerate(labels.items()):
    source=manifest[i];original=Image.open(out/source['image']).convert('RGB');im=original.copy();im.thumbnail((640,640))
    with torch.inference_mode():d=segmenter.raw(im)
    ids=[j for j,s in enumerate(d['scores']) if float(s)>.8 and int(d['labels'][j]) in (3,8)]
    if not ids:continue
    j=max(ids,key=lambda j:float((d['boxes'][j,2]-d['boxes'][j,0])*(d['boxes'][j,3]-d['boxes'][j,1])))
    b=tight_box(d['masks'][j,0].numpy()>.5)
    if not b:continue
    scale=original.width/im.width;box=[round(v*scale) for v in b]
    crop=original.crop(box);crop.thumbnail((240,300));x=pos%5*250;y=pos//5*350;sheet.paste(crop,(x,y));draw.text((x,y+305),f'{i} {brand} {colour}',fill='black')
    rows.append(dict(image=source['image'],box=box,brand=brand,colour=colour,source=source['source'],original_sha256=source['sha256'],group='owner-phone-outing-2026-10-25',split='train',label_source='Manual visible emblem/body review; detector proposes geometry only',status='geometry_pending_review',exclude_from_independent_evaluation=True))
(out/'annotation_proposals.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
sheet.save(out/'annotation_proposals.jpg')
print('Proposals',len(rows),flush=True)
