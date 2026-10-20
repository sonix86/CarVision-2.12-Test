"""Independent learned paint-statistics candidate; selection uses development only.

No rules force grey/black for particular filenames, brands or samples. Public
test labels are inspected only after the candidate's checkpoint is selected.
"""
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS,read_image
from carvision.colour_model import paint_statistics,colour_head

def main():
 torch.set_num_threads(1);out=ROOT/'models/paint_colour_candidate';out.mkdir(exist_ok=True);rows=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text())['rows'];cache=ROOT/'cache/paint_colour_features.pt';fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
 if cache.exists():
  ck=torch.load(cache,weights_only=True);assert ck['fingerprint']==fingerprint;x=ck['x']
 else:
  features=[]
  for i,r in enumerate(rows):
   features.append(paint_statistics(read_image(ROOT/'datasets/vcor/images'/r['image'])))
   if i%1000==0:print('FEATURES',i,'/',len(rows),flush=True)
  x=torch.stack(features);torch.save(dict(x=x,fingerprint=fingerprint),cache)
 y=torch.tensor([COLOURS.index(r['colour']) for r in rows]);ids={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']};train=ids['train'];val=ids['validation'];mean=x[train].mean(0);std=x[train].std(0).clamp_min(.025);x=(x-mean)/std
 dev=[r for r in json.loads((ROOT/'data/public_prototype/colour_development_review.json').read_text())['references'] if r['reference_colour'] is not None];dx=(torch.stack([paint_statistics(read_image(ROOT/'data/public_prototype'/r['image']).crop(r['box'])) for r in dev])-mean)/std;dy=torch.tensor([COLOURS.index(r['reference_colour']) for r in dev]);counts=torch.bincount(y[train],minlength=8).float();weights=counts.rsqrt();weights/=weights.mean();criterion=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
 best=-1.;history=[];started=time.perf_counter()
 for seed in [13210,13211]:
  torch.manual_seed(seed);random.seed(seed);model=colour_head(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.03);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,70)
  for epoch in range(70):
   model.train()
   for batch in train[torch.randperm(len(train))].split(96):
    optimizer.zero_grad();loss=criterion(model(x[batch]),y[batch]);loss.backward();optimizer.step()
   scheduler.step();model.eval()
   with torch.inference_mode():a=float((model(x[val]).argmax(1)==y[val]).float().mean());b=float((model(dx).argmax(1)==dy).float().mean())
   score=(a+b)/2;row=dict(seed=seed,epoch=epoch+1,validation_accuracy=a,cross_domain_development_accuracy=b,selection_score=score);history.append(row)
   if a>=.88 and score>best:
    best=score;best_row=row;torch.save(dict(origin='team-trained-colour-statistics',architecture='original_body_joint_colour_statistics_mlp_v2',feature_mode='paint_statistics',head=model.state_dict(),classes=COLOURS,mean=mean,std=std,source='Own spatial RGB/HSV/Lab paint evidence; independently trained on VCoR labels; no pretrained colour decisions/features used',best_epoch=epoch+1,ready_made_colour_model_used=False),out/'colour_model.pt')
  print('SEED',seed,'best',best,flush=True)
 info=dict(history=history,seconds=time.perf_counter()-started,selection='Mean VCoR validation and separate 23 Stanford development reviews; validation >=88%; no tests/city samples used',feature_dimensions=x.shape[1],train=len(train),validation=len(val),test=len(ids['test']))
 if (out/'colour_model.pt').exists():
  ck=torch.load(out/'colour_model.pt',weights_only=True);model.load_state_dict(ck['head']);model.eval()
  with torch.inference_mode():pred=model(x[ids['test']]).argmax(1)
  matrix=torch.zeros((8,8),dtype=torch.long)
  for truth,p in zip(y[ids['test']],pred):matrix[int(truth),int(p)]+=1
  info.update(best=best_row,test_accuracy=float((pred==y[ids['test']]).float().mean()),confusion_matrix=matrix.tolist())
 (out/'training.json').write_text(json.dumps(info,indent=2));print('FINAL',json.dumps({k:v for k,v in info.items() if k!='history'}),flush=True)
if __name__=='__main__':main()
