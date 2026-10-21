"""Learn new classes through train-only labelled exemplars, using our body CNN.

No extra neural training round or foreign car classifier. Validation decides
whether this candidate is useful; runtime must retain its matched32-class bank.
"""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from carvision.core import ROOT
source=ROOT/'models/occlusion_parts_candidate/vehicle_model.pt';ck=torch.load(source,weights_only=True);dataset=ROOT/'models/ev_parts_candidate/dataset.json';data=json.loads(dataset.read_text(encoding='utf-8'));pairs=[tuple(p) for p in data['pairs']]
assert list(zip(ck['brands'],ck['labels']))==pairs[:len(ck['labels'])]
ck.update(neural_labels=ck['labels'],neural_brands=ck['brands'],labels=[m for b,m in pairs],brands=[b for b,m in pairs],architecture='reference-expanded-model-network-v1',expansion_method='Train-only labelled body/detail exemplar classifier; own24-class CNN body features. No neural output weights invented for new classes.',source_neural_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
out=ROOT/'models/prototype_parts_candidate';out.mkdir(exist_ok=True);torch.save(ck,out/'vehicle_model.pt');shutil.copy2(dataset,out/'dataset.json');(out/'training.json').write_text(json.dumps(dict(method=ck['expansion_method'],neural_training_rounds=0,new_images_used='Training split only; fit_prototype_references performs train-only exemplar learning and grouped validation selection.',source=str(source),test_used_for_selection=False)),encoding='utf-8')
print('Prepared candidate taxonomy',len(pairs),'neural output count',len(ck['neural_labels']),flush=True)
