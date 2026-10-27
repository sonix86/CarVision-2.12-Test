"""Audit splits, provenance and explicit visual review before the make fit."""
import collections,hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'models/make_scope_v23'
data=torch.load(ROOT/'cache/make_scope_v23.pt',weights_only=True)
old=torch.load(ROOT/'cache/ev_patch_v22.pt',weights_only=True)
lookup={r['image']:r for r in data['rows'] if r['historical']}
assert len(lookup)==len(old['rows'])
for row in old['rows']:
    current=lookup[row['image']]
    assert current['split']==row['split'] and current['brand']==old['pairs'][row['y']][0]
    assert torch.equal(current['body_rgb'],row['body_rgb']) and torch.equal(current['body_mask'],row['body_mask'])
groups=collections.defaultdict(set);photos=collections.defaultdict(set)
review_file=OUT/'reviewed_images.json';review=json.loads(review_file.read_text(encoding='utf-8')) if review_file.exists() else {}
unreviewed=[];checked=0
for row in data['rows']:
    assert data['brands'][row['y']]==row['brand']
    groups[row['group']].add(row['split'])
    if row['image'].startswith(('make_scope_v23/','make_flickr_v23/')):
        source=ROOT/'datasets'/row['image'];r=row['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r['sha256']
        assert r['license'] and r['source'] and r['license_url'] is not None
        photos[r['sha256']].add(row['split']);checked+=1
        if row['image'] not in review:unreviewed.append(row['image'])
assert not any(len(v)>1 for v in groups.values())
assert not any(len(v)>1 for v in photos.values())
report=dict(data_integrity_passed=True,ready_for_training=not unreviewed,unreviewed_images=unreviewed,
            historical_crops_preserved=len(old['rows']),new_source_crops_checked=checked,
            brands=len(data['brands']),rows=len(data['rows']),groups=len(groups),
            split_counts=dict(collections.Counter(r['split'] for r in data['rows'])),
            dataset_sha256=hashlib.sha256((ROOT/'cache/make_scope_v23.pt').read_bytes()).hexdigest(),
            limitation='Declared source-session independence, not verified physical-vehicle identity. The integrity audit is not an accuracy test.')
(OUT/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='unreviewed_images'}))
