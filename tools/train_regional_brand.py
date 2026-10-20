"""Fine-tune our own brand network on independently sourced regional vehicle variants."""
import collections,copy,json,random,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Classifier,read_image,make_preprocess

def main():
 torch.set_num_threads(3);torch.manual_seed(13210);random.seed(13210);np.random.seed(13210)
 old=torch.load(ROOT/'models/classifier.pt',weights_only=True);root=ROOT/'data/public_prototype';regional=ROOT/'data/regional';out=ROOT/'models/regional_candidate';out.mkdir(exist_ok=True)
 public=[json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if s.strip()];extra=[json.loads(s) for s in (regional/'labels.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
 # Prototype coverage requires examples in all grouped splits. New-class counts
 # are deliberately reported; 3–5 test images do not establish street accuracy.
 counts=collections.Counter(r['brand'] for r in extra if r['split']=='train');eligible={b for b,n in counts.items() if n>=15 and all(sum(r['brand']==b and r['split']==s for r in extra)>=3 for s in ['validation','test'])}
 extra=[r for r in extra if r['brand'] in eligible];brands=sorted(set(old['brands'])|eligible);oldsplit=json.loads((ROOT/'models/public_split.json').read_text())
 tr=[(root,r) for r in public if r['image'] in set(oldsplit['train'])];va=[(root,r) for r in public if r['image'] in set(oldsplit['validation'])];rt=[(regional,r) for r in extra if r['split']=='train'];rv=[(regional,r) for r in extra if r['split']=='validation'];test=[(regional,r) for r in extra if r['split']=='test']
 if len(rt)<40 or len(rv)<10:raise ValueError('Not enough supplementary training/validation data')
 model=Classifier(len(brands));state=model.state_dict()
 for key,value in old['state'].items():
  if not key.startswith('brand.'):state[key]=value
 for i,b in enumerate(old['brands']):state['brand.weight'][brands.index(b)]=old['state']['brand.weight'][i];state['brand.bias'][brands.index(b)]=old['state']['brand.bias'][i]
 model.load_state_dict(state)
 if (out/'classifier.pt').exists():(out/'classifier.pt').replace(out/'classifier.previous.pt')
 def cache(pairs):return [read_image(folder/r['image']).crop(r['box']).resize((176,176),Image.Resampling.BILINEAR) for folder,r in pairs],torch.tensor([brands.index(r['brand']) for folder,r in pairs])
 train_images,train_y=cache(tr+rt);val_images,val_y=cache(va);regional_images,regional_y=cache(rv);preprocess=make_preprocess(160,16);vx=torch.stack([preprocess(im) for im in val_images]);rx=torch.stack([preprocess(im) for im in regional_images])
 augment=transforms.Compose([transforms.RandomResizedCrop(160,scale=(.55,1.),ratio=(.75,1.35)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.2,.2,.15,.01),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225]),transforms.RandomErasing(p=.3,scale=(.03,.25),value=0)])
 # Regional classes are sparse; replay old data to limit forgetting.
 sampling=torch.ones(len(train_y));category_counts=collections.Counter(r['vehicle_category'] for folder,r in rt)
 for i,(folder,r) in enumerate(rt,len(tr)):sampling[i]=3*min(3.,(max(category_counts.values())/category_counts[r['vehicle_category']])**.5)
 class_counts=torch.bincount(train_y,minlength=len(brands)).float();weights=class_counts.clamp_min(1).pow(-.25);weights/=weights.mean();criterion=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.05)
 optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=.00006),dict(params=list(model.neck.parameters())+list(model.brand.parameters()),lr=.0003)],weight_decay=.015);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,18)
 def predictions(x):
  with torch.inference_mode():return torch.cat([model(b)[0] for b in x.split(64)])
 best=-1.;history=[];started=time.perf_counter()
 for epoch in range(18):
  model.train();epoch_start=time.perf_counter();loss_sum=correct=0;ids=torch.multinomial(sampling,len(train_y),replacement=True)
  for indices in ids.split(32):
   if len(indices)==1:continue
   x=torch.stack([augment(train_images[int(i)]) for i in indices]);y=train_y[indices];optimizer.zero_grad();p,_=model(x);loss=criterion(p,y);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(indices);correct+=int((p.argmax(1)==y).sum())
  scheduler.step();model.eval();a=float((predictions(vx).argmax(1)==val_y).float().mean());b=float((predictions(rx).argmax(1)==regional_y).float().mean());score=(a+b)/2;row=dict(epoch=epoch+1,loss=loss_sum/len(ids),train_accuracy=correct/len(ids),stanford_validation_accuracy=a,regional_validation_accuracy=b,selection_score=score,seconds=time.perf_counter()-epoch_start);history.append(row);print(json.dumps(row),flush=True)
  if a>=.80 and score>best:
   best=score;torch.save(dict(old,state=model.state_dict(),brands=brands,source='public Stanford Cars + category-labelled Commons regional variants; no own Tashkent photos',regional_training=True,validation_accuracy=a,regional_validation_accuracy=b),out/'classifier.pt')
  info=dict(history=history,brands=brands,train_stanford=len(tr),train_regional=len(rt),validation_stanford=len(va),validation_regional=len(rv),test_regional=len(test),augmentation='Random resized crop with 55–100% visible area, horizontal flip, mild colour jitter, random erasing up to 25%',initialization='Our previously trained brand classifier; original general ImageNet initialization; new brand rows randomly initialized',selection='Mean Stanford and regional development accuracy; Stanford development >=80%; tests and six demo photos excluded',seconds=time.perf_counter()-started)
  (out/'training.json').write_text(json.dumps(info,indent=2))
 if not (out/'classifier.pt').exists():raise RuntimeError('No candidate passed validation; production model retained')
 candidate=torch.load(out/'classifier.pt',weights_only=True);model.load_state_dict(candidate['state']);model.eval()
 # Calibrate confidence on development logits only, after selecting the network.
 logits=torch.cat([predictions(vx),predictions(rx)]).detach();truth=torch.cat([val_y,regional_y]);log_temperature=torch.tensor(0.,requires_grad=True);calibration=torch.optim.LBFGS([log_temperature],lr=.1,max_iter=40)
 def closure():
  calibration.zero_grad();loss=torch.nn.functional.cross_entropy(logits/log_temperature.exp().clamp(.5,5.),truth);loss.backward();return loss
 calibration.step(closure);candidate['temperature']=float(log_temperature.detach().exp().clamp(.5,5.));torch.save(candidate,out/'classifier.pt');info['temperature']=candidate['temperature']
 def runtime_probabilities(x):return ((predictions(x)/candidate['temperature']).softmax(1)+(predictions(x.flip(3))/candidate['temperature']).softmax(1))/2
 tx,ty=cache(test);test_x=torch.stack([preprocess(im) for im in tx]);probs=runtime_probabilities(test_x);tp=probs.argmax(1);accepted=probs.max(1).values>=.5;info.update(regional_test_accuracy=float((tp==ty).float().mean()),regional_test_coverage_at_05=float(accepted.float().mean()),regional_test_accepted_accuracy_at_05=float((tp[accepted]==ty[accepted]).float().mean()) if accepted.any() else None,best_validation_stanford=candidate['validation_accuracy'],best_validation_regional=candidate['regional_validation_accuracy'],test_used_for_selection=False,regional_test_runtime_policy='Mean original/mirror probabilities with development-calibrated temperature, matching shipped pipeline')
 matrix=torch.zeros((len(brands),len(brands)),dtype=torch.long)
 for truth,pred in zip(ty,tp):matrix[int(truth),int(pred)]+=1
 info['regional_test_confusion_matrix']=matrix.tolist();info['regional_test_per_brand']={brand:dict(count=int(matrix[i].sum()),accuracy=float(matrix[i,i]/matrix[i].sum())) for i,brand in enumerate(brands) if matrix[i].sum()>0}
 # Evaluate deterministic partial crops after selecting the checkpoint.
 partial=[]
 for im in tx:
  w,h=im.size;partial.append(preprocess(im.crop((0,0,round(w*.65),h))))
 partial_x=torch.stack(partial);info['regional_left_65_percent_test_accuracy']=float((runtime_probabilities(partial_x).argmax(1)==ty).float().mean())
 halves=torch.stack([preprocess(im.crop((0,0,im.width//2,im.height))) for im in tx]+[preprocess(im.crop((im.width//2,0,im.width,im.height))) for im in tx]);half_y=torch.cat([ty,ty]);info['regional_half_vehicle_test_accuracy']=float((runtime_probabilities(halves).argmax(1)==half_y).float().mean());info['partial_test_scope']='Deterministic left/right50% crops of held-out vehicle boxes; synthetic visibility benchmark, not real street occlusions'
 (out/'training.json').write_text(json.dumps(info,indent=2));print('FINAL',json.dumps({k:v for k,v in info.items() if k not in ('history','regional_test_confusion_matrix')}),flush=True)
 # Promotion is assessed separately against baseline on identical validation data.
 base=Classifier(len(old['brands']));base.load_state_dict(old['state']);base.eval()
 with torch.inference_mode():
  baseline_pred=torch.cat([base(b)[0] for b in rx.split(64)]).argmax(1)
  baseline_test=torch.cat([base(b)[0] for b in test_x.split(64)]).argmax(1)
  baseline_partial=torch.cat([base(b)[0] for b in partial_x.split(64)]).argmax(1)
  baseline_halves=torch.cat([base(b)[0] for b in halves.split(64)]).argmax(1)
 baseline_accuracy=np.mean([old['brands'][int(i)]==brands[int(y)] for i,y in zip(baseline_pred,regional_y)])
 if candidate['regional_validation_accuracy']>baseline_accuracy and candidate['validation_accuracy']>=.80:
  import shutil
  backup=ROOT/'models/backups/v12';backup.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'models/classifier.pt',backup/'classifier.pt');shutil.copy2(out/'classifier.pt',ROOT/'models/classifier.pt');info['promoted']=True
 else:info['promoted']=False
 info['baseline_regional_validation_accuracy']=float(baseline_accuracy);info['baseline_regional_test_accuracy']=float(np.mean([old['brands'][int(i)]==brands[int(y)] for i,y in zip(baseline_test,ty)]));info['baseline_regional_left_65_percent_test_accuracy']=float(np.mean([old['brands'][int(i)]==brands[int(y)] for i,y in zip(baseline_partial,ty)]));info['baseline_regional_half_vehicle_test_accuracy']=float(np.mean([old['brands'][int(i)]==brands[int(y)] for i,y in zip(baseline_halves,half_y)]));(out/'training.json').write_text(json.dumps(info,indent=2));print('PROMOTION',info['promoted'],'baseline regional validation',baseline_accuracy,flush=True)
if __name__=='__main__':main()
