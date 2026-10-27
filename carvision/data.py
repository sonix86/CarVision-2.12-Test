import collections
import hashlib
import json
import random
from pathlib import Path
from PIL import Image
from .core import COLOURS, read_image

def load_records(file):
    file=Path(file).resolve(); rows=[];labelled_boxes={}
    for n,line in enumerate(file.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line)
        for key in ['image','box','brand','colour','source','group']:
            if key not in r: raise ValueError(f'Line {n}: missing {key}')
        for key in ['image','brand','colour','source','group']:
            if not isinstance(r[key],str) or not r[key].strip():raise ValueError(f'Line {n}: invalid {key}')
        for key in ['brand','colour','source','group']:r[key]=r[key].strip()
        r['path']=(file.parent/r['image']).resolve()
        if not r['path'].is_relative_to(file.parent): raise ValueError(f'Line {n}: image must be inside dataset directory')
        im=read_image(r['path']); w,h=im.size
        x1,y1,x2,y2=r['box']
        if not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError(f'Line {n}: invalid box')
        if r['colour'] not in COLOURS or not r['brand'].strip(): raise ValueError(f'Line {n}: invalid label')
        if not r['source'].strip() or not r['group'].strip(): raise ValueError(f'Line {n}: source/group required')
        r['hash']=hashlib.sha256(r['path'].read_bytes()).hexdigest()
        r['pixel_hash']=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        identity=(r['pixel_hash'],tuple(r['box']));label=(r['brand'],r['colour'])
        if identity in labelled_boxes and labelled_boxes[identity][0]!=label:
            raise ValueError(f'Line {n}: Conflicting labels for the same image/box (previous line {labelled_boxes[identity][1]})')
        labelled_boxes[identity]=(label,n)
        rows.append(r)
    if not rows: raise ValueError('Dataset is empty. Add and label real car photographs first.')
    return rows

def split_records(rows, seed=3210, fraction=.2):
    # Union groups, image paths and exact duplicates: never split a photo/session/car.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b): parent[find(a)]=find(b)
    for r in rows:
        union('g:'+r['group'],'h:'+r['hash'])
        if r.get('pixel_hash'):union('g:'+r['group'],'p:'+r['pixel_hash'])
    groups=collections.defaultdict(list)
    for r in rows: groups[find('g:'+r['group'])].append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys)
    if len(keys)<2: raise ValueError('Need at least 2 independent groups for train/validation.')
    n=max(1,min(len(keys)-1,round(len(keys)*fraction)))
    valkeys=set(keys[:n]); train=[]; val=[]
    for k,rs in groups.items(): (val if k in valkeys else train).extend(rs)
    missing={r['brand'] for r in val}-{r['brand'] for r in train}
    if missing: raise ValueError(f'Validation-only brands: {sorted(missing)}. Collect more independent groups or change seed.')
    if len({r['brand'] for r in train})<2: raise ValueError('Training needs at least 2 brands.')
    return train,val

def statistics(rows):
    return dict(cars=len(rows),images=len({r.get('pixel_hash',r['hash']) for r in rows}),brands=dict(collections.Counter(r['brand'] for r in rows)),colours=dict(collections.Counter(r['colour'] for r in rows)),sources=dict(collections.Counter(r['source'] for r in rows)),groups=len({r['group'] for r in rows}))

def dataset_fingerprint(rows):
    payload=sorted(json.dumps({k:r[k] for k in ['image','box','brand','colour','source','group','hash']},sort_keys=True) for r in rows)
    return hashlib.sha256('\n'.join(payload).encode('utf-8')).hexdigest()
