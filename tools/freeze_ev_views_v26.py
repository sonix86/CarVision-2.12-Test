"""Freeze only explicitly reviewed exterior proposals; forbid split drift."""
import collections,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'models/ev_views_v26';target=ROOT/'cache/ev_views_v26_data.pt'
assert not target.exists(),'Do not overwrite a frozen training corpus'
approval=json.loads((out/'review_approval.json').read_text(encoding='utf-8'));proposals=torch.load(out/'proposals.pt',weights_only=True);report=json.loads((out/'proposal_report.json').read_text());source=ROOT/'datasets/ev_views_v26/manifest.json';assert report['source_manifest_sha256']==hashlib.sha256(source.read_bytes()).hexdigest(),'Collection changed after preparation'
assert approval['proposal_sha256']==hashlib.sha256((out/'proposals.pt').read_bytes()).hexdigest(),'Reviewed pixels changed'
assert set(approval['reviewed_images'])=={r['image'] for r in proposals['rows']}
base_path=ROOT/'cache/multimodal_v24_data.pt';data=torch.load(base_path,weights_only=True);assert hashlib.sha256(base_path.read_bytes()).hexdigest()==report['base_dataset_sha256'];added=[r for r in proposals['rows'] if r['image'] not in approval['exclusions']];rows=data['rows']+added;groups=collections.defaultdict(set)
for r in rows:groups[r['group']].add(r['split'])
assert all(len(v)==1 for v in groups.values()),'Source leakage across splits'
# A historical photograph may contain two annotated cars. Preserve those
# distinct original crops, while forbidding new rows reusing an existing ID.
assert len({r['image'] for r in added})==len(added)
assert not ({r['image'] for r in added}&{r['image'] for r in data['rows']})
torch.save(dict(rows=rows,brands=data['brands']),target)
summary=dict(dataset_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),base_dataset_sha256=report['base_dataset_sha256'],source_manifest_sha256=report['source_manifest_sha256'],base_rows=len(data['rows']),new_rows=len(added),new_splits=dict(collections.Counter(r['split'] for r in added)),new_makes=dict(collections.Counter(r['brand'] for r in added)),rows=len(rows),splits=dict(collections.Counter(r['split'] for r in rows)),groups_disjoint=True,original_rows_unchanged=True,private_additional_rows=sum(bool(r.get('owner')) for r in added),requested_families_not_exact_model_ground_truth=True)
(out/'dataset.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
