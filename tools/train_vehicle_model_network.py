"""Train a fresh model-name head on grouped regional labels, never on demo photos."""
import sys,json,copy,time,collections,re,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from PIL import ImageOps
from torchvision import transforms
from carvision.core import ROOT,Classifier,read_image,make_preprocess
from carvision.vehicle_model import pooled_features

torch.set_num_threads(3);torch.manual_seed(321014);random.seed(321014);started=time.perf_counter()
ck=torch.load(ROOT/'models/classifier.pt',weights_only=True);network=Classifier(len(ck['brands'])).eval();network.load_state_dict(ck['state'])
rows=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
def label(r):
    s=r['vehicle_category']
    if s=='VAZ-2101':return '2101'
    return re.sub(r' \(.*?\)','',s).removeprefix(r['brand']+' ').removesuffix(' sedan')
for r in rows:r['model']=label(r)
labels=sorted(set((r['brand'],r['model']) for r in rows));names=[v for b,v in labels];brands=[b for b,v in labels]
pre=make_preprocess(160,16)
augment=transforms.Compose([transforms.RandomResizedCrop(160,scale=(.65,1.),ratio=(.8,1.25)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.12,.12,.08),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

model=Classifier(len(labels));state=model.state_dict()
for k,v in ck['state'].items():
    if not k.startswith('brand.'):state[k]=v
model.load_state_dict(state)
images={split:[read_image(ROOT/'data/regional'/r['image']).crop(r['box']).resize((176,176)) for r in rows if r['split']==split] for split in ['train','validation','test']}
y={split:torch.tensor([labels.index((r['brand'],r['model'])) for r in rows if r['split']==split]) for split in images}
x={split:torch.stack([pre(im) for im in images[split]]+[pre(ImageOps.mirror(im)) for im in images[split]]) for split in ['validation','test']}
counts=torch.bincount(y['train'],minlength=len(labels)).float();sampling=counts[y['train']].pow(-.5)
optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=.00012),dict(params=list(model.neck.parameters())+list(model.brand.parameters()),lr=.001)],weight_decay=.02)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40);history=[];best=-1
@torch.inference_mode()
def logits(split):return torch.cat([model(v)[0] for v in x[split].split(48)])
for epoch in range(40):
    model.train();ids=torch.multinomial(sampling,len(y['train'])*2,replacement=True)
    for batch in ids.split(32):
        if len(batch)<2:continue
        optimizer.zero_grad();pred=model(torch.stack([augment(images['train'][int(i)]) for i in batch]))[0];loss=nn.functional.cross_entropy(pred,y['train'][batch],label_smoothing=.08);loss.backward();optimizer.step()
    scheduler.step();model.eval();v=logits('validation');n=len(y['validation']);p=v.softmax(1);p=(p[:n]+p[n:])/2;accuracy=float((p.argmax(1)==y['validation']).float().mean());nll=float(nn.functional.nll_loss(p.log(),y['validation']));score=accuracy-.01*nll
    if score>best:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    history.append(dict(epoch=epoch+1,validation_accuracy=accuracy,nll=nll));print(json.dumps(history[-1]),flush=True)
model.load_state_dict(state);model.eval();v=logits('validation');truth=y['validation'].repeat(2)
temperature=min([.5,.75,1.,1.25,1.5,2.,3.],key=lambda t:float(nn.functional.cross_entropy(v/t,truth)))
n=len(y['validation']);vp=(v/temperature).softmax(1);vp=(vp[:n]+vp[n:])/2;vy=y['validation'];threshold=.65
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
    accept=vp.max(1).values>=t
    if int(accept.sum())>=15 and float((vp.argmax(1)[accept]==vy[accept]).float().mean())>=.9:threshold=t;break
p=(logits('test')/temperature).softmax(1);n=len(y['test']);p=(p[:n]+p[n:])/2;truth=y['test'];accept=p.max(1).values>=threshold
info=dict(origin='project-trained-regional-model-head',architecture='MobileNetV3 Small avg/max + 256 hidden + independently trained 16-model head; fine-tuned features',labels=names,brands=brands,counts={b+' '+m:dict(collections.Counter(r['split'] for r in rows if (r['brand'],r['model'])==(b,m))) for b,m in labels},best_epoch=best_epoch,temperature=temperature,threshold=threshold,validation_accuracy=float((vp.argmax(1)==vy).float().mean()),test_count=len(truth),test_accuracy=float((p.argmax(1)==truth).float().mean()),test_coverage=float(accept.float().mean()),test_accepted_accuracy=float((p.argmax(1)[accept]==truth[accept]).float().mean()) if accept.any() else None,history=history,seconds=time.perf_counter()-started,limitations='Public category labels; sparse Damas/Nexia/Cobalt samples; no own photographs. Existing grouped train/validation/test preserved. Backbone initialized from own brand network trained on regional training only. No open-set model benchmark. Demo photos excluded.',optimizer='AdamW features 0.00012, neck/head 0.001; weight decay .02; cosine 40 epochs',augmentation='65-100% crop, flip, mild colour jitter; balanced replacement sampling')
torch.save(dict(origin=info['origin'],architecture='finetuned-model-network-v1',labels=names,brands=brands,state=state,temperature=temperature,threshold=threshold),ROOT/'models/vehicle_model.pt')
(ROOT/'models/vehicle_model_training.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8');print('FINAL',json.dumps({k:v for k,v in info.items() if k!='history'},ensure_ascii=False),flush=True)
