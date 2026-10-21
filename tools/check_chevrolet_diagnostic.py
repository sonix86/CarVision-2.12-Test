"""Post-selection application diagnostics, never training or tuning data."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from carvision.core import ROOT,Pipeline,save_prediction
from carvision.vehicle_model import VehicleModelClassifier
from carvision.visual_parts import VisualReferences

out=ROOT/'outputs/chevrolet-diagnostic-v20';out.mkdir(exist_ok=True)
candidate=ROOT/'models/body_expert_candidate';info=json.loads((candidate/'evaluation.json').read_text(encoding='utf-8'))
ck=torch.load(candidate/'vehicle_model.pt',weights_only=True);ref=torch.load(candidate/'visual_references.pt',weights_only=True)
assert ref['source_model_sha256']==hashlib.sha256((candidate/'vehicle_model.pt').read_bytes()).hexdigest()
p=Pipeline(mode='fast');previous=p.vehicle_classifier
assert not getattr(previous,'body_expert',None),'Run before installing candidate; immutable v1.9 baseline required'
current=VehicleModelClassifier(ck);current.references=VisualReferences(ref)
credits=json.loads((ROOT/'samples/CHEVROLET_PHOTO_SOURCES.json').read_text(encoding='utf-8'))
items=[dict(path=ROOT/r['relative_file'],expected_model=r['model'],role='New Flickr research diagnostic, not a representative accuracy benchmark') for r in credits]
parking=Path('C:/Users/Miku/Downloads/Telegram Desktop/photo_5_2026-10-21_13-00-40.jpg')
if parking.exists():items.append(dict(path=parking,expected_model=None,role='User development parking photograph; never used for training or selection'))
results=[]
for item in items:
    p.vehicle_classifier=previous;before=p.predict(item['path'])
    p.vehicle_classifier=current;after=p.predict(item['path'])
    save_prediction(item['path'],before,out/'before');save_prediction(item['path'],after,out/'after')
    def main(pred):
        return max(pred['cars'],key=lambda r:r.get('mask_pixels',0),default={})
    b,a=main(before),main(after)
    row=dict(image=str(item['path']),expected_model=item['expected_model'],role=item['role'],before=b,after=a,baseline_seconds=before['seconds'],candidate_seconds=after['seconds'],cars=len(after['cars']))
    results.append(row);print(item['path'].name,'before',b.get('model_candidate'),b.get('brand'),'after',a.get('model_candidate'),a.get('brand'),flush=True)
(out/'summary.json').write_text(json.dumps(dict(selected_candidate_sha256=info['candidate_sha256'],test_used_for_selection=False,scope='Application diagnostic only; photographer sessions correlated, some public press photographs may repeat older source material. No accuracy-generalization claim.',photos=results),indent=2),encoding='utf-8')
