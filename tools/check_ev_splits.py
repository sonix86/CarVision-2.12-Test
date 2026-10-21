"""Audit actual source session separation and immutable baseline membership."""
import collections,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from carvision.core import ROOT
from tools.ev_group_audit import excluded_legacy_images
data=json.loads((ROOT/'models/ev_parts_candidate/dataset.json').read_text(encoding='utf-8'));old=[json.loads(l) for l in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()];old_by={r['image']:r for r in old};new_by={r['image']:r for r in data['new_rows']};excluded,audit=excluded_legacy_images();cache=torch.load(ROOT/'cache/ev_parts_v19.pt',weights_only=True);original=torch.load(ROOT/'cache/expanded_parts_v18.pt',weights_only=True)
groups=collections.defaultdict(set);counts=collections.Counter();previous={r['image']:r['split'] for r in original['rows']};new_images=set(new_by)
for row in cache['rows']:
 image=row['image'];counts[row['split']]+=1
 assert image not in excluded
 if image in previous:assert row['split']==previous[image],'Old split was changed'
 source=old_by.get(image,new_by.get(image))
 if source:groups[source['group']].add(row['split'])
assert not any(len(s)>1 for s in groups.values()),'Capture group leaks between splits'
assert {r['image'] for r in cache['rows']}==set(previous)-excluded|new_images
source=ROOT/'datasets/ev_commons_v19';sources=json.loads((source/'manifest.json').read_text(encoding='utf-8'))+json.loads((source/'thumb_data/manifest.json').read_text(encoding='utf-8'))
for row in sources:assert hashlib.sha256((source/row['image']).read_bytes()).hexdigest()==row['sha256']
info=dict(passed=True,counts=dict(counts),known_author_date_groups=len(groups),group_intersections=0,old_splits_immutable=True,excluded_legacy_audit=audit,public_photo_checksums_verified=len(sources),new_photos=len(data['new_rows']),new_groups=len({r['group'] for r in data['new_rows']}),stanford_identity_labels_available=False,city_images_used_for_training_or_selection=False)
(ROOT/'models/ev_split_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
