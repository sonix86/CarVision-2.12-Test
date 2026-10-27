"""Keep source splits, add reviewed owner TRAIN data, lock reported errors out of fitting."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import read_image
from carvision.scene import manual_foreground

data=torch.load(ROOT/'cache/make_scope_v23.pt',weights_only=True);rows=data['rows'];brands=data['brands']
# These reported errors are development controls, never fitting rows in this iteration.
pool=ROOT/'data/owner_oct25';manifest=json.loads((pool/'manifest.json').read_text(encoding='utf-8'))
controls={manifest[i]['image'] for i in (6,32,54)}
added=[]
for directory in (ROOT/'data/own_oct21',pool):
    for line in (directory/'labels.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if directory==pool and r['image'] in controls:continue
        if r['brand'] not in brands:continue
        source=directory/r['image'];im=read_image(source).crop(r['box']);im.thumbnail((320,320));mask=manual_foreground(im)
        added.append(dict(image=directory.name+'/'+r['image'],brand=r['brand'],family=r.get('model','unknown'),group=r['group'],split='train',body_rgb=torch.from_numpy(np.asarray(im).copy()),body_mask=torch.from_numpy(mask.copy()),historical=False,y=brands.index(r['brand']),owner=True,colour=r['colour'],source=dict(source='Owner photographs explicitly authorized for training',path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),label_source=r['label_source'])))
rows+=added
groups=collections.defaultdict(set)
for row in rows:groups[row['group']].add(row['split'])
assert all(len(s)==1 for s in groups.values())
out=ROOT/'models/multimodal_v24';out.mkdir(exist_ok=True);target=ROOT/'cache/multimodal_v24_data.pt';torch.save(dict(rows=rows,brands=brands),target)
report=dict(base_dataset_sha256=hashlib.sha256((ROOT/'cache/make_scope_v23.pt').read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),rows=len(rows),owner_train_crops=len(added),owner_groups=sorted({r['group'] for r in added}),owner_colours=dict(collections.Counter(r['colour'] for r in added)),locked_development_controls=sorted(controls),original_splits_preserved=True,private_pixels_distributable=False)
(out/'dataset.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=True),flush=True)
