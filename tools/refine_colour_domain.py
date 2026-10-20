"""Compare feature choices on VCoR validation + separate Stanford DEV colour review only."""
import collections,json,random,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS,read_image
from carvision.colour_model import ColourClassifier,colour_statistics,colour_head
torch.set_num_threads(4);torch.manual_seed(531);random.seed(531)
rows=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text())['rows'];raw=torch.load(ROOT/'cache/vcor_features.pt',weights_only=True)['x'];y=torch.tensor([COLOURS.index(r['colour']) for r in rows]);indices={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
dev=[r for r in json.loads((ROOT/'data/public_prototype/colour_development_review.json').read_text())['references'] if r['reference_colour'] is not None]
dev_images=[read_image(ROOT/'data/public_prototype'/r['image']).crop(r['box']) for r in dev];dev_y=torch.tensor([COLOURS.index(r['reference_colour']) for r in dev])
original=torch.load(ROOT/'models/colour_model.pt',weights_only=True);baseline=ColourClassifier(original);baseline_dev=np.mean([baseline.predict(im)[0]==r['reference_colour'] for im,r in zip(dev_images,dev)])
base_metrics=json.loads((ROOT/'models/colour_training.json').read_text());best_score=.5*base_metrics['best_validation_accuracy']+.5*baseline_dev;chosen=None;history=[];train=indices['train'];validation=indices['validation'];data=raw[:,-261:];mean=data[train].mean(0);std=data[train].std(0).clamp_min(.03);x=(data-mean)/std;dx=(torch.stack([colour_statistics(im) for im in dev_images])-mean)/std
head=colour_head(261);optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.03);weights=torch.bincount(y[train],minlength=8).float().rsqrt();weights/=weights.mean();criterion=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
for epoch in range(80):
    head.train()
    for batch in train[torch.randperm(len(train))].split(64):
        optimizer.zero_grad();loss=criterion(head(x[batch]),y[batch]);loss.backward();optimizer.step()
    head.eval()
    with torch.inference_mode():accuracy=float((head(x[validation]).argmax(1)==y[validation]).float().mean());dev_accuracy=float((head(dx).argmax(1)==dev_y).float().mean())
    score=.5*accuracy+.5*dev_accuracy;row=dict(epoch=epoch+1,validation_accuracy=accuracy,cross_domain_dev_accuracy=dev_accuracy,selection_score=score);history.append(row)
    if accuracy>=.85 and score>best_score:
        best_score=score;chosen=dict(original,feature_mode='statistics',head={k:v.clone() for k,v in head.state_dict().items()},mean=mean,std=std,best_epoch=epoch+1,architecture='original_spatial_rgb_hsv_statistics_mlp_v1')
        chosen_row=row
if chosen is not None:
    torch.save(original,ROOT/'outputs/colour-combined-baseline.pt');torch.save(chosen,ROOT/'models/colour_model.tmp');(ROOT/'models/colour_model.tmp').replace(ROOT/'models/colour_model.pt')
    final=ColourClassifier(chosen)
    with torch.inference_mode():head.load_state_dict(chosen['head']);head.eval();pred=head(x[indices['test']]).argmax(1)
    matrix=torch.zeros((8,8),dtype=torch.long)
    for truth,p in zip(y[indices['test']],pred):matrix[int(truth),int(p)]+=1
    base_metrics.update(best_epoch=chosen['best_epoch'],best_validation_accuracy=chosen_row['validation_accuracy'],test_accuracy=float((pred==y[indices['test']]).float().mean()),confusion_matrix=matrix.tolist(),method='Original spatial RGB/HSV statistics + independently trained MLP; no pretrained colour decision or feature weights are used in colour inference',feature_mode='statistics',cross_domain_development_images=len(dev),cross_domain_development_accuracy=chosen_row['cross_domain_dev_accuracy'],selection='Equal-weight VCoR validation and 23 AI-reviewed Stanford development crops, VCoR validation >=85%; test not used in selection',domain_refinement_history=history)
    (ROOT/'models/colour_training.json').write_text(json.dumps(base_metrics,indent=2));print('Selected statistics head',json.dumps(chosen_row),'test',base_metrics['test_accuracy'],flush=True)
else:print('Combined feature model retained; baseline dev accuracy',baseline_dev,flush=True)
(ROOT/'models/colour_domain_comparison.json').write_text(json.dumps(dict(baseline_dev_accuracy=float(baseline_dev),baseline_validation_accuracy=base_metrics.get('best_validation_accuracy'),statistics_selected=chosen is not None,history=history),indent=2))
