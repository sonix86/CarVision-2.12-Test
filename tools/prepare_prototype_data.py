"""Join public JPEGs to original human annotations using exact Git blob hashes.

No image-name guessing, pretrained predictions or student-source attribution.
Colour values are explicitly marked heuristic pseudo-labels, not evaluation truth.
"""
import collections
import hashlib
import io
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from scipy.io import loadmat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import heuristic_colour

ROOT=Path(__file__).resolve().parents[1];raw=ROOT/'datasets/stanford_cars';out=ROOT/'data/public_prototype';(out/'images').mkdir(parents=True,exist_ok=True)
tree=json.loads((raw/'original_github_tree.json').read_text(encoding='utf-8'))
lookup={r['sha']:r['path'] for r in tree['tree'] if r['path'].lower().endswith('.jpg')}
annotations={}
for split,name in [('train','cars_train_annos.mat'),('test','cars_test_annos_withlabels.mat')]:
    a=loadmat(raw/name,squeeze_me=True,struct_as_record=False)['annotations']
    annotations[split]={str(r.fname):r for r in a}
names=[str(s) for s in loadmat(raw/'cars_annos.mat',squeeze_me=True,struct_as_record=False)['class_names']]
covered={'Chevrolet','Daewoo','Hyundai','Toyota','Mercedes-Benz','BMW','Volkswagen','Nissan','Audi','Honda','Ford','Tesla','Volvo'}
records=[];counts=collections.Counter();seen=set();missing=0;conflicts=0
for split in ['train','test']:
    for file in sorted(raw.glob(split+'-*.parquet')):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                model_name=names[int(row['label'])];brand=model_name.split()[0]
                if brand not in covered:continue
                if split=='train' and counts[(split,brand)]>=600:continue
                b=row['image']['bytes'];digest=hashlib.sha256(b).hexdigest()
                if digest in seen:continue
                original=lookup.get(hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest())
                if not original:missing+=1;continue
                original_split='train' if '/cars_train/' in original else 'test'
                if original_split!=split:conflicts+=1;continue
                annotation=annotations[split][Path(original).name]
                if names[int(getattr(annotation,'class'))-1].split()[0]!=brand:conflicts+=1;continue
                im=Image.open(io.BytesIO(b)).convert('RGB');w,h=im.size;im.thumbnail((768,768))
                box=[max(0,float(annotation.bbox_x1)-1)*im.width/w,max(0,float(annotation.bbox_y1)-1)*im.height/h,min(w,float(annotation.bbox_x2))*im.width/w,min(h,float(annotation.bbox_y2))*im.height/h]
                if box[2]-box[0]<24 or box[3]-box[1]<24:continue
                image='images/stanford_'+digest[:24]+'.jpg';destination=out/image
                if not destination.exists():im.save(destination,quality=92)
                colour,_=heuristic_colour(im.crop(box))
                records.append(dict(image=image,box=box,brand=brand,colour=colour,source='public:Stanford Cars/Krause2013; exact original annotations',group='stanford:'+digest,official_split=split,original=original,original_sha256=digest,colour_annotation='heuristic pseudo-label; NOT human colour truth'))
                counts[(split,brand)]+=1;seen.add(digest)
        print(split,file.name,'cars',sum(counts.values()),flush=True)
(out/'labels.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
summary=dict(source='Stanford Cars, Krause et al. 2013',image_mirror='https://huggingface.co/datasets/tanganke/stanford_cars',annotation_mirror='https://github.com/jhpohovey/StanfordCars-Dataset',mapping='Exact Git blob SHA1 join, verified brand and official split; duplicate original SHA256 removed. Model-level disagreements within a brand do not change brand labels; brand/split conflicts excluded.',own_tashkent_photos=0,counts={s:{b:n for (sp,b),n in counts.items() if sp==s} for s in ['train','test']},unmatched_images=missing,excluded_conflicts=conflicts,colour_truth='No human colour ground truth. Stored colour is an explicit heuristic pseudo-label; do not report its self-agreement as colour accuracy.')
(out/'provenance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
