"""Train a colour network on real VCoR labels with lighting/crop/occlusion augmentation.

Checkpoint selection uses VCoR validation + explicitly separate Stanford development
review. Six city examples and both public test sets never select the checkpoint.
Candidate promotion requires both validation safeguards; no forced city colours.
"""
import copy,json,random,sys,time
from pathlib import Path
import numpy as np,torch
from torchvision import transforms
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS,read_image
from carvision.colour_model import ColourCNN,ColourClassifier,PREPROCESS

def main():
 torch.set_num_threads(3);torch.manual_seed(13210);random.seed(13210);np.random.seed(13210)
 out=ROOT/'models/robust_colour_candidate';out.mkdir(exist_ok=True)
 rows=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text())['rows'];root=ROOT/'datasets/vcor/images'
 augment=transforms.Compose([transforms.RandomResizedCrop(128,scale=(.55,1.),ratio=(.7,1.45)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.35,contrast=.25,saturation=.12,hue=.01),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225]),transforms.RandomErasing(p=.2,scale=(.02,.15),value=0)])
 images=[read_image(root/r['image']).resize((144,144)) for r in rows];targets=torch.tensor([COLOURS.index(r['colour']) for r in rows]);indices={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
 validation=torch.stack([PREPROCESS(images[i]) for i in indices['validation']]);vy=targets[indices['validation']]
 dev=[r for r in json.loads((ROOT/'data/public_prototype/colour_development_review.json').read_text())['references'] if r['reference_colour'] is not None];dev_images=[read_image(ROOT/'data/public_prototype'/r['image']).crop(r['box']) for r in dev];dx=torch.stack([PREPROCESS(im) for im in dev_images]);dy=torch.tensor([COLOURS.index(r['reference_colour']) for r in dev])
 baseline=ColourClassifier(torch.load(ROOT/'models/colour_model.pt',weights_only=True));base_dev=np.mean([baseline.predict(im)[0]==r['reference_colour'] for im,r in zip(dev_images,dev)]);base_val=json.loads((ROOT/'models/colour_training.json').read_text())['best_validation_accuracy'];del baseline
 model=ColourCNN();model.features.load_state_dict(torch.load(ROOT/'models/generic_brand_backbone.pt',weights_only=True)['state'])
 optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=.00008),dict(params=model.head.parameters(),lr=.001)],weight_decay=.02);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,25)
 counts=torch.bincount(targets[indices['train']],minlength=8).float();weights=counts.rsqrt();weights/=weights.mean();loss_fn=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
 best=-1;best_row=None;history=[];started=time.perf_counter()
 def accuracy(x,y):
  with torch.inference_mode():return float((torch.cat([model(b) for b in x.split(64)]).argmax(1)==y).float().mean())
 for epoch in range(25):
  model.train();loss_sum=0.;correct=0;epoch_start=time.perf_counter()
  for batch in indices['train'][torch.randperm(len(indices['train']))].split(48):
   x=torch.stack([augment(images[int(i)]) for i in batch]);y=targets[batch];optimizer.zero_grad();p=model(x);loss=loss_fn(p,y);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(batch);correct+=int((p.argmax(1)==y).sum())
  scheduler.step();model.eval();val=accuracy(validation,vy);cross=accuracy(dx,dy);score=(val+cross)/2
  row=dict(epoch=epoch+1,loss=loss_sum/len(indices['train']),train_accuracy=correct/len(indices['train']),validation_accuracy=val,cross_domain_development_accuracy=cross,selection_score=score,seconds=time.perf_counter()-epoch_start);history.append(row);print(json.dumps(row),flush=True)
  if val>=.88 and score>best:
   best=score;best_row=row.copy();torch.save(dict(origin='team-trained-colour-head-imagenet-backbone',architecture='colour_cnn_avgmax_v2',classes=COLOURS,state=model.state_dict(),source='VCoR labels; general ImageNet initialization; own end-to-end training',ready_made_colour_model_used=False,best_epoch=epoch+1),out/'colour_model.pt')
  (out/'training.json').write_text(json.dumps(dict(history=history,baseline_validation=base_val,baseline_cross_domain_development=base_dev,best=best_row,selection='Mean of VCoR validation and 23 development reviews, validation >=88%; tests excluded',seconds=time.perf_counter()-started),indent=2))
 if best_row:
  ck=torch.load(out/'colour_model.pt',weights_only=True);model.load_state_dict(ck['state']);model.eval();test_x=torch.stack([PREPROCESS(images[i]) for i in indices['test']]);test_accuracy=accuracy(test_x,targets[indices['test']]);promotion=best>(base_val+base_dev)/2 and best_row['cross_domain_development_accuracy']>base_dev
  info=json.loads((out/'training.json').read_text());info.update(test_accuracy=test_accuracy,promoted=bool(promotion),train=len(indices['train']),validation=len(indices['validation']),test=len(indices['test']),test_used_for_selection=False);(out/'training.json').write_text(json.dumps(info,indent=2))
  if promotion:
   backup=ROOT/'models/backups/v12';backup.mkdir(parents=True,exist_ok=True)
   import shutil
   shutil.copy2(ROOT/'models/colour_model.pt',backup/'colour_model.pt');shutil.copy2(ROOT/'models/colour_training.json',backup/'colour_training.json');shutil.copy2(out/'colour_model.pt',ROOT/'models/colour_model.pt')
  print('FINAL',json.dumps(info['best']),test_accuracy,'promoted',promotion,flush=True)
 else:print('No candidate passed the validation safeguard; baseline retained.',flush=True)
if __name__=='__main__':main()
