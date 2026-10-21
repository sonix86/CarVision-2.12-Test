"""Bounded validation candidate combining two project-trained heads, not a new training round."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from carvision.core import ROOT
source=ROOT/'models/ev_parts_candidate';out=ROOT/'models/hybrid_parts_candidate';out.mkdir(exist_ok=True)
assert (source/'training.json').exists(),'Complete the second bounded neural round first'
ck=torch.load(source/'vehicle_model.pt',weights_only=True);legacy=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);reference=torch.load(ROOT/'models/visual_references.pt',weights_only=True)
assert list(zip(legacy['brands'],legacy['labels']))==list(zip(ck['brands'],ck['labels']))[:len(legacy['labels'])]
ck.update(legacy_model=legacy,legacy_references=reference,legacy_model_sha256=hashlib.sha256((ROOT/'models/vehicle_model.pt').read_bytes()).hexdigest(),legacy_reference_sha256=hashlib.sha256((ROOT/'models/visual_references.pt').read_bytes()).hexdigest(),legacy_weight=0.,hybrid_method='Constant validation-selected mixture of two project-trained networks/reference banks; no extra neural training round')
torch.save(ck,out/'vehicle_model.pt')
for name in ['dataset.json','training.json']:shutil.copy2(source/name,out/name)
cache=ROOT/'cache/visual_parts_v19.pt'
if cache.exists():
 features=torch.load(cache,weights_only=True)
 assert features['model_sha']==hashlib.sha256((source/'vehicle_model.pt').read_bytes()).hexdigest()
 # Exact same neural state and vocabulary: only the optional legacy mixture
 # metadata was added. Rebind features to the new final container explicitly.
 features['neural_container_source_sha256']=features['model_sha'];features['model_sha']=hashlib.sha256((out/'vehicle_model.pt').read_bytes()).hexdigest();torch.save(features,ROOT/'cache/visual_parts_hybrid_v19.pt')
print('Prepared bounded hybrid candidate, same trained neural state; legacy mixture0 until validation selection',flush=True)
