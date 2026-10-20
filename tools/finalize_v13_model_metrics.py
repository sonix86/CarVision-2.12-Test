"""Publish actual selected-weight measurements; never reuse old model scores."""
import collections,json,sys,shutil
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS,read_image
from carvision.colour_model import ColourClassifier,PREPROCESS

torch.set_num_threads(3)
if '--brand-only' in sys.argv:
    info=json.loads((ROOT/'models/regional_candidate/training.json').read_text(encoding='utf-8'));assert info.get('promoted');(ROOT/'models/regional_brand_training.json').write_text(json.dumps(info,indent=2));print('Published selected regional brand metrics');raise SystemExit
comparisons=[];best=-1.;selected=None
baseline_path=ROOT/'models/backups/v12/colour_training.json'
baseline=json.loads((baseline_path if baseline_path.exists() else ROOT/'models/colour_training.json').read_text(encoding='utf-8'))
best=(baseline['best_validation_accuracy']+.6956521739130435)/2
for name in ['robust_colour_candidate','paint_colour_candidate','colour_ensemble_candidate','colour_panels_candidate']:
    folder=ROOT/'models'/name;info=json.loads((folder/'training.json').read_text(encoding='utf-8'))
    assert 'test_accuracy' in info or name=='paint_colour_candidate' and 'best' not in info or name in ('colour_ensemble_candidate','colour_panels_candidate') and 'candidates' in info,'Training has not finished: '+name
    if not info.get('best'):continue
    row=info['best'];comparisons.append(dict(candidate=name,**row))
    if row['validation_accuracy']>=.88 and row['selection_score']>best:best=row['selection_score'];selected=folder
if selected:
    backup=ROOT/'models/backups/v12';backup.mkdir(parents=True,exist_ok=True)
    if not (backup/'colour_model.pt').exists():shutil.copy2(ROOT/'models/colour_model.pt',backup/'colour_model.pt')
    if not (backup/'colour_training.json').exists():shutil.copy2(ROOT/'models/colour_training.json',backup/'colour_training.json')
    candidate=torch.load(selected/'colour_model.pt',weights_only=True)
    if candidate.get('architecture')=='colour_evidence_ensemble_v3':
        used=sorted({entry['component'] for entry in candidate['mixture']});mapping={old:new for new,old in enumerate(used)};candidate['components']=[candidate['components'][i] for i in used];candidate['mixture']=[dict(entry,component=mapping[entry['component']]) for entry in candidate['mixture']]
    torch.save(candidate,ROOT/'models/colour_model.pt')
(ROOT/'models/colour_candidate_comparison.json').write_text(json.dumps(dict(candidates=comparisons,selected=selected.name if selected else 'v1.2 baseline',selection='Mean VCoR validation and 23 separate Stanford development reviews; VCoR validation>=88%; tests never select architecture/checkpoint'),indent=2))
checkpoint=torch.load(ROOT/'models/colour_model.pt',weights_only=True)
if checkpoint.get('architecture')=='colour_cnn_avgmax_v2':
    model=ColourClassifier(checkpoint);training=json.loads((ROOT/'models/robust_colour_candidate/training.json').read_text(encoding='utf-8'));manifest=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text(encoding='utf-8'))['rows'];test=[r for r in manifest if r['split']=='test'];matrix=torch.zeros((8,8),dtype=torch.long)
    with torch.inference_mode():
        for start in range(0,len(test),64):
            rows=test[start:start+64];x=torch.stack([PREPROCESS(read_image(ROOT/'datasets/vcor/images'/r['image'])) for r in rows]);pred=model.network(x).argmax(1)
            for row,p in zip(rows,pred):matrix[COLOURS.index(row['colour']),int(p)]+=1
    # Measure original images with shipped preprocessing. CNN training used
    # a cached144px intermediate for validation; do not misreport its test number.
    metrics=dict(classes=COLOURS,train=training['train'],validation=training['validation'],test=len(test),test_accuracy=float(matrix.diag().sum()/matrix.sum()),confusion_matrix=matrix.tolist(),best_epoch=checkpoint['best_epoch'],best_validation_accuracy=training['best']['validation_accuracy'],counts={s:dict(collections.Counter(r['colour'] for r in manifest if r['split']==s)) for s in ['train','validation','test']},history=training['history'],seconds=training['seconds'],method='Own colour CNN: general ImageNet MobileNetV3 Small initialization, end-to-end fine-tuning, average/max pooling and independently learned 8-colour head; realistic crop, lighting and occlusion augmentation',architecture=checkpoint['architecture'],optimizer='AdamW backbone lr0.00008/head0.001 weight_decay0.02; cosine schedule25 epochs',batch=48,loss='Inverse-sqrt class-weighted cross entropy, label smoothing0.03',augmentation='Random resized crop55–100%, flip, brightness±35%, contrast±25%, saturation±12%, hue±0.01, erasing2–15% with probability20%',head_parameters=sum(p.numel() for p in model.network.head.parameters()),general_backbone_parameters=sum(p.numel() for p in model.network.features.parameters()),feature_dimensions=1152,input_size=128,epochs_completed=len(training['history']),timing_scope='End-to-end CNN training and validation; excludes initial data loading',ready_made_colour_model_used=False,source='VCoR: https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset',split='Original VCoR split, exact SHA256 and identical dHash duplicates removed; true vehicle identities unavailable',test_used_for_checkpoint_selection=False,selection=training['selection'],cross_domain_development_images=23,cross_domain_development_accuracy=training['best']['cross_domain_development_accuracy'])
    (ROOT/'models/colour_training.json').write_text(json.dumps(metrics,indent=2))
    print('Published actual selected colour model metrics',metrics['test_accuracy'],flush=True)
elif checkpoint.get('feature_mode')=='paint_statistics':
    checkpoint['origin']='team-trained-colour-statistics';torch.save(checkpoint,ROOT/'models/colour_model.pt');model=ColourClassifier(checkpoint);training=json.loads((ROOT/'models/paint_colour_candidate/training.json').read_text(encoding='utf-8'));manifest=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text(encoding='utf-8'))['rows']
    metrics=dict(classes=COLOURS,train=training['train'],validation=training['validation'],test=training['test'],test_accuracy=training['test_accuracy'],confusion_matrix=training['confusion_matrix'],best_epoch=checkpoint['best_epoch'],best_validation_accuracy=training['best']['validation_accuracy'],counts={s:dict(collections.Counter(r['colour'] for r in manifest if r['split']==s)) for s in ['train','validation','test']},history=training['history'],seconds=training['seconds'],method='Original joint HSV saturation/value histograms, Lab and spatial body-panel colour evidence + independently learned MLP. No pretrained colour decision/feature extractor.',architecture=checkpoint['architecture'],optimizer='AdamW lr0.0007 weight_decay0.03, cosine70 epochs, two random initializations',batch=96,loss='Inverse-sqrt class-weighted cross entropy with label smoothing0.03',head_parameters=sum(p.numel() for p in model.head.parameters()),general_backbone_parameters=0,feature_dimensions=len(model.mean),input_size=128,epochs_completed=len(training['history']),timing_scope='MLP training/validation excluding feature extraction',ready_made_colour_model_used=False,source='VCoR: https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset',split='Original VCoR split; exact SHA256/identical dHash duplicates removed; vehicle identities unavailable',test_used_for_checkpoint_selection=False,selection=training['selection'],cross_domain_development_images=23,cross_domain_development_accuracy=training['best']['cross_domain_development_accuracy'])
    (ROOT/'models/colour_training.json').write_text(json.dumps(metrics,indent=2));print('Selected body-paint evidence model',metrics['test_accuracy'],flush=True)
elif checkpoint.get('architecture')=='colour_evidence_ensemble_v3':
    model=ColourClassifier(checkpoint);training=json.loads((selected/'training.json').read_text(encoding='utf-8'));manifest=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text(encoding='utf-8'))['rows'];test=[r for r in manifest if r['split']=='test'];matrix=torch.zeros((8,8),dtype=torch.long)
    for i,row in enumerate(test):
        colour,confidence=model.predict(read_image(ROOT/'datasets/vcor/images'/row['image']));matrix[COLOURS.index(row['colour']),COLOURS.index(colour)]+=1
        if i%250==0:print('Actual ensemble test',i,'/',len(test),flush=True)
    head_parameters=backbone_parameters=0
    for component in model.components:
        if component.network is not None:head_parameters+=sum(p.numel() for p in component.network.head.parameters());backbone_parameters+=sum(p.numel() for p in component.network.features.parameters())
        else:
            head_parameters+=sum(p.numel() for p in component.head.parameters())
            if component.features is not None:backbone_parameters+=sum(p.numel() for p in component.features.parameters())
    component_training=dict(baseline=baseline,cnn=json.loads((ROOT/'models/robust_colour_candidate/training.json').read_text(encoding='utf-8')),paint=json.loads((ROOT/'models/paint_colour_candidate/training.json').read_text(encoding='utf-8')))
    metrics=dict(classes=COLOURS,train=6210,validation=training['validation'],test=len(test),test_accuracy=float(matrix.diag().sum()/matrix.sum()),confusion_matrix=matrix.tolist(),best_epoch=0,best_validation_accuracy=training['best']['validation_accuracy'],counts={s:dict(collections.Counter(r['colour'] for r in manifest if r['split']==s)) for s in ['train','validation','test']},history=[],component_training=component_training,seconds=training['seconds'],method='Собственные обученные цветовые головы: полная вырезка и участки кузова; веса видов выбраны по development. Готовые vehicle-colour classifiers не используются.',architecture=checkpoint['architecture'],optimizer='Головы: AdamW; CNN backbone0.00008/head0.001, original MLP0.001, paint MLP0.0007. Смесь: сетка весов0.25 по development, без градиентного обучения.',batch='48 / 64 / 96 (головы)',loss='Weighted CE, smoothing0.03 в головах; смесь выбирается по development',head_parameters=head_parameters,general_backbone_parameters=backbone_parameters,feature_dimensions='1152 / 1413 / 929 (головы)',input_size=128,epochs_completed=0,timing_scope='Mixture evidence extraction/selection only; actual component histories/timings are in component_training',ready_made_colour_model_used=False,source='VCoR: https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset',split='Original VCoR split; exact SHA256/identical dHash duplicates removed; vehicle IDs unavailable',test_used_for_checkpoint_selection=False,selection=training['selection'],cross_domain_development_images=23,cross_domain_development_accuracy=training['best']['cross_domain_development_accuracy'],mixture=checkpoint['mixture'])
    (ROOT/'models/colour_training.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8');print('Selected colour evidence mixture test',metrics['test_accuracy'],flush=True)
if selected:
    metrics=json.loads((ROOT/'models/colour_training.json').read_text(encoding='utf-8'));metrics.update(baseline_test_accuracy=baseline['test_accuracy'],baseline_cross_domain_development_accuracy=.6956521739130435,candidate_comparison=comparisons);(ROOT/'models/colour_training.json').write_text(json.dumps(metrics,indent=2))
if (ROOT/'models/regional_candidate/training.json').exists():
    regional=json.loads((ROOT/'models/regional_candidate/training.json').read_text(encoding='utf-8'))
    if regional.get('promoted'):
        (ROOT/'models/regional_brand_training.json').write_text(json.dumps(regional,indent=2));print('Published regional brand training metrics',regional['regional_test_accuracy'],flush=True)
