"""Import explicitly authorized phone photos; uncertain cars are not pseudo-labelled."""
import hashlib,json,shutil
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
source=Path('C:/Users/Miku/Downloads/Telegram Desktop')
out=ROOT/'data/own_oct21';(out/'images').mkdir(parents=True,exist_ok=True)
# All photographs from this outing stay in TRAIN, including neighbouring frames
# of the same vehicle. They are never an independent evaluation set.
annotations=[(1,[247,535,899,969],'Chevrolet','Cobalt','black'),
 (1,[88,530,243,651],'Chevrolet','Tracker','white'),
 (2,[269,509,620,711],'Chevrolet','Tracker','white'),
 (5,[158,622,469,762],'Chevrolet','Cobalt','white'),
 (7,[558,646,711,768],'Chevrolet','Cobalt','white'),
 (7,[760,642,923,761],'Chevrolet','Cobalt','white'),
 (8,[112,516,693,826],'Chevrolet','Cobalt','white'),
 (10,[175,551,842,1017],'Chevrolet','Tracker','black'),
 (9,[129,368,660,888],'Kia',None,'white')]
badges={1:[710,750,785,797],2:[520,595,565,624],8:[198,658,248,686],10:[740,673,781,705],9:[340,593,435,632]}
manifest=[];rows=[]
for i in range(1,11):
 name=f'photo_{i}_2026-10-21_13-00-40.jpg';src=source/name;dst=out/'images'/name;shutil.copy2(src,dst)
 manifest.append(dict(image='images/'+name,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),group='owner-phone-outing-2026-10-21',split='train',source='User supplied phone photograph; use for local training explicitly authorized in chat',license='User-provided; not assigned a public redistribution licence',annotation_status='partial: only visually identifiable vehicles labelled; remaining objects require review'))
 for photo,box,brand,model,colour in annotations:
  if photo!=i:continue
  row=dict(image='images/'+name,box=box,brand=brand,model=model,colour=colour,source='Owner phone photos,21October2026',group=manifest[-1]['group'],split='train',label_source='Manual visual review, no classifier-generated labels',review='Model-level label omitted when not certain')
  if i in badges:row['badge_box']=badges[i] if i!=1 or model=='Cobalt' else None
  rows.append(row)
(out/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
(out/'provenance.json').write_text(json.dumps(dict(photos=manifest,labelled_cars=len(rows),model_labelled_cars=sum(r['model'] is not None for r in rows),excluded_from_independent_evaluation=True,remaining_unlabelled_cars='Not treated as negatives; not ready for detector training'),ensure_ascii=False,indent=2),encoding='utf-8')
print('Imported10photos,9manually labelled crops,8model labels,one training group')
