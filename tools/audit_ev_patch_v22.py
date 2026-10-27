"""Verify the immutable historical prefix and new source/session separation."""
import collections
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
old = torch.load(ROOT / 'cache/body_expert_v20.pt', weights_only=True)
new = torch.load(ROOT / 'cache/ev_patch_v22.pt', weights_only=True)
assert new['pairs'][:len(old['pairs'])] == old['pairs']
for a, b in zip(old['rows'], new['rows']):
    assert (a['image'], a['split'], a['y']) == (b['image'], b['split'], b['y'])
    assert torch.equal(a['body_rgb'], b['body_rgb'])
    assert torch.equal(a['body_mask'], b['body_mask'])
groups = collections.defaultdict(set)
sources = collections.defaultdict(set)
metadata = [json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
metadata += json.loads((ROOT/'models/ev_parts_candidate/dataset.json').read_text(encoding='utf-8'))['new_rows']
active_images = {r['image'] for r in new['rows']}
inactive_metadata = [r for r in metadata if r['image'] not in active_images]
for r in [r for r in metadata if r['image'] in active_images] + new['rows'][len(old['rows']):]:
    groups[r['group']].add(r['split'])
    if isinstance(r.get('source'), dict) and r['source'].get('source'):
        sources[r['source']['source']].add(r['split'])
assert not any(len(v) > 1 for v in groups.values())
assert not any(len(v) > 1 for v in sources.values())
for r in new['rows'][len(old['rows']):]:
    if r['image'].startswith('own_oct21/'):
        assert r['split'] == 'train'
        continue
    source = ROOT/'datasets'/r['image']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == r['source']['sha256']
    assert r['source']['license']
report = dict(passed=True, historical_rows_preserved=len(old['rows']),
              total_rows=len(new['rows']), families=len(new['pairs']),
              new_rows=len(new['rows'])-len(old['rows']),
              groups=len(groups), source_urls=len(sources), inactive_historical_metadata=len(inactive_metadata),
              split_counts=dict(collections.Counter(r['split'] for r in new['rows'])),
              dataset_sha256=hashlib.sha256((ROOT/'cache/ev_patch_v22.pt').read_bytes()).hexdigest(),
              limitations='Session grouping and near-duplicate filtering do not prove physical vehicle independence.')
(ROOT/'models/ev_patch_v22/data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
