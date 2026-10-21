"""Preserve rejected research honestly; unchanged selected neural weights."""
import json,hashlib,shutil
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];out=root/'models/body_expert_candidate';owner=root/'models/owner_candidate'
info=json.loads((owner/'training.json').read_text(encoding='utf-8'))
assert not info['selected'],'A selected owner candidate requires its own calibrated evaluation first'
badge=torch.load(owner/'badge.pt',weights_only=True);assert not badge['enabled']
ck=torch.load(out/'vehicle_model.pt',weights_only=True)
ck['badge_evidence']=dict(badge,brand_thresholds={})
ck['owner_adaptation_status']='12epoch experiment rejected by validation; original selected body expert retained; badge influence disabled'
torch.save(ck,out/'vehicle_model.pt');sha=hashlib.sha256((out/'vehicle_model.pt').read_bytes()).hexdigest()
ref=torch.load(out/'visual_references.pt',weights_only=True);ref['source_model_sha256']=sha;torch.save(ref,out/'visual_references.pt')
evaluation=json.loads((out/'evaluation.json').read_text(encoding='utf-8'));evaluation['pre_badge_metadata_sha256']=evaluation['candidate_sha256'];evaluation['candidate_sha256']=sha
evaluation['owner_experiment']='Own FC adaptation rejected; embedded badge evidence disabled. Neural probabilities and evaluated model weights unchanged.'
(out/'evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
shutil.copy2(owner/'training.json',root/'models/owner_adaptation_training.json')
print('Kept selected expert; rejected owner adaptation and badge acceptance; matched source/ref',sha)
