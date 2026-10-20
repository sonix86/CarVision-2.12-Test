"""Attach the new held-out VCoR benchmark without changing the frozen brand-test result."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from carvision.core import ROOT,read_image
from carvision.colour_model import ColourClassifier
torch.set_num_threads(4)
checkpoint=torch.load(ROOT/'models/colour_model.pt',weights_only=True);model=ColourClassifier(checkpoint)
metrics=json.loads((ROOT/'models/colour_training.json').read_text(encoding='utf-8'))
if checkpoint.get('architecture') not in ('colour_cnn_avgmax_v2','colour_evidence_ensemble_v3') and checkpoint.get('feature_mode')!='paint_statistics':
    metrics.update(head_parameters=sum(p.numel() for p in model.head.parameters()),general_backbone_parameters=sum(p.numel() for p in model.features.parameters()),feature_dimensions=len(checkpoint['mean']),input_size=128,epochs_completed=len(metrics['history']),optimizer='AdamW lr=0.001 weight_decay=0.01; cosine decay over max100 epochs; early stop after15 unimproved epochs, minimum30',batch=64,loss='Weighted cross entropy; inverse-sqrt class counts; label smoothing 0.03',timing_scope='seconds measures MLP training/validation, excluding data preparation and frozen feature extraction')
(ROOT/'models/colour_training.json').write_text(json.dumps(metrics,indent=2))
path=ROOT/'models/evaluation/public_test.json';result=json.loads(path.read_text());result.setdefault('colour_visual_review_hsv',result.get('colour_visual_review'))
refs=json.loads((ROOT/'data/public_prototype/colour_reference_review.json').read_text())['references'];scored=[]
for r in refs:
    if r['reference_colour'] is None:continue
    colour,confidence=model.predict(read_image(ROOT/'data/public_prototype'/r['image']).crop(r['box']));scored.append(dict(id=r['id'],reference=r['reference_colour'],prediction=colour,correct=colour==r['reference_colour']))
result['colour_visual_review']=dict(images=len(scored),accuracy=sum(r['correct'] for r in scored)/len(scored),annotator='Codex visual development diagnostic, not independent human ground truth; this diagnostic informed subsequent development',details=scored)
result['colour_benchmark']={k:v for k,v in metrics.items() if k!='history'}
scene=result.get('dominant_car_full_photo_check',{})
if scene.get('seconds',0)>300:scene.update(wall_timing_valid=False,timing_note='Wall time includes a long session interruption. Use per-image runtime checks rather than this elapsed wall clock.')
path.write_text(json.dumps(result,indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
matrix=metrics['confusion_matrix'];fig,ax=plt.subplots(figsize=(8,7));ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(8),metrics['classes'],rotation=45,ha='right');ax.set_yticks(range(8),metrics['classes']);ax.set(xlabel='Predicted',ylabel='Dataset reference',title=f"Own colour head: {metrics['test_accuracy']:.1%}, {metrics['test']} VCoR test photos")
for i in range(8):
    for j in range(8):ax.text(j,i,str(matrix[i][j]),ha='center',va='center',color='white' if matrix[i][j]>170 else 'black')
fig.tight_layout();fig.savefig(ROOT/'models/evaluation/colour_confusion.png',dpi=150);plt.close(fig)
print('VCoR test accuracy:',metrics['test_accuracy'],'additional visual diagnostic:',result['colour_visual_review']['accuracy'])
