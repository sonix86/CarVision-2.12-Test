"""Install a matched, evaluated candidate; preserve the complete old model pair."""
import json,hashlib,shutil
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];candidate=ROOT/'models/body_expert_candidate'
evaluation=json.loads((candidate/'evaluation.json').read_text(encoding='utf-8'));training=json.loads((candidate/'training.json').read_text(encoding='utf-8'))
if not training['selected']:raise SystemExit('Validation rejected candidate')
old=ROOT/'models/vehicle_model.pt';old_sha=hashlib.sha256(old.read_bytes()).hexdigest()
if old_sha!=evaluation['baseline_sha256']:raise SystemExit('Baseline changed: refuse accidental reinstallation')
record=next(r for r in training['history'] if r['epoch']==training['best_epoch'])
if not record['guard']:raise SystemExit('Selected validation guard failed')
ck=torch.load(candidate/'vehicle_model.pt',weights_only=True);ref=torch.load(candidate/'visual_references.pt',weights_only=True)
sha=hashlib.sha256((candidate/'vehicle_model.pt').read_bytes()).hexdigest()
if sha!=evaluation['candidate_sha256'] or ref['source_model_sha256']!=sha:raise SystemExit('Candidate/reference hash mismatch')
backup=ROOT/'models/backups/v19_before_v20'
if backup.exists():raise SystemExit('Backup already exists; inspect state before any retry')
backup.mkdir(parents=True)
for name in ['vehicle_model.pt','visual_references.pt','visual_parts_training.json','model_card.json']:shutil.copy2(ROOT/'models'/name,backup/name)
for name in ['vehicle_model.pt','visual_references.pt']:shutil.copy2(candidate/name,ROOT/'models'/name)
shutil.copy2(candidate/'evaluation.json',ROOT/'models/body_expert_evaluation.json');shutil.copy2(candidate/'training.json',ROOT/'models/body_expert_training.json')
card=json.loads((ROOT/'models/model_card.json').read_text(encoding='utf-8'));card.update(release='2.0',owner='София Рахманкулова',vehicle_model_threshold=ref['confidence_threshold'])
card['historical_v19_model_metrics']={k:card[k] for k in ['vehicle_model_test_accuracy','vehicle_model_test_count','instance_model_metrics'] if k in card}
card['vehicle_model_test_accuracy']=evaluation['results']['whole']['candidate']['all']['top1'];card['vehicle_model_test_count']=416
card['model_rejection_threshold']=ref['confidence_threshold']
card['v20_body_expert']=dict(architecture='ResNet18 ImageNet generic initialization; independently trained32-model FC and layer3/4',geometry='Aspect-preserving224padding; source crop shapes restored',training_photos=779,validation_photos=161,test_photos=416,selected_epoch=training['best_epoch'],mixture_weight=record['weight'],results=evaluation['results'],source_model_sha256=old_sha,candidate_sha256=sha,city_examples_used_for_training_or_selection=False,new_photographs_used_for_training=False,limitations=evaluation['limitations'])
card['current_scope']='v2.0 body expert selected on grouped validation, priorities Chevrolet and low-resolution bodies;32families unchanged. Small Cobalt/Nexia test counts and unvalidated Damas prevent broad reliability claims. Historical18-make metrics below are not measurements of this new expert.'
card['v20_body_expert']['generic_initialization_sha256']=hashlib.sha256((ROOT/'cache/hub/checkpoints/resnet18-f37072fd.pth').read_bytes()).hexdigest()
(ROOT/'models/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
parts=json.loads((ROOT/'models/visual_parts_training.json').read_text(encoding='utf-8'));parts['model_sha256']=sha;parts['confidence_threshold']=ref['confidence_threshold'];parts['probability_temperature']=ref['probability_temperature'];parts['v20_body_expert']=dict(training='body_expert_training.json',evaluation='body_expert_evaluation.json',old_reference_descriptors_unchanged=True,reason='Embedded independently trained body expert changes the enclosing file hash; original CNN and original train-reference features remain the same.')
(ROOT/'models/visual_parts_training.json').write_text(json.dumps(parts,indent=2),encoding='utf-8')
print('Installed matched candidate',sha,'old pair saved',backup)
