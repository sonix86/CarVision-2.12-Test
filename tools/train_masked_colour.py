"""Train from source colour labels and generic COCO foreground masks; no city example labels."""
import sys,json,time,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image
from carvision.core import ROOT,COLOURS,read_image
from carvision.segmentation import InstanceSegmenter,tight_box,isolate
from carvision.masked_colour import features,head
from carvision.colour_model import ColourClassifier
started=time.perf_counter();torch.set_num_threads(3);torch.manual_seed(321015)
out=ROOT/'models/masked_colour_candidate';out.mkdir(exist_ok=True);cache=ROOT/'cache/masked_colour_v15.pt';meta=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text());selected=[]
for split,count in [('train',80),('validation',20),('test',30)]:
 for colour in COLOURS:
  selected+=sorted([r for r in meta['rows'] if r['split']==split and r['colour']==colour],key=lambda r:hashlib.sha256(r['image'].encode()).hexdigest())[:count]
net=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True);baseline=ColourClassifier(torch.load(ROOT/'models/colour_model.pt',weights_only=True));rows=[]
if cache.exists():rows=torch.load(cache,weights_only=True)['rows'];assert all(a['image']==b['image'] for a,b in zip(rows,selected))
for r in selected[len(rows):]:
 im=read_image(ROOT/'datasets/vcor/images'/r['image']);im.thumbnail((384,384));d=net.raw(im);indices=[i for i in range(len(d['scores'])) if int(d['labels'][i]) in [3,8] and float(d['scores'][i])>.3]
 if indices:
  i=max(indices,key=lambda i:int((d['masks'][i,0]>.5).sum()));mask=d['masks'][i,0].numpy()>.5
 else:
  import cv2
  a=np.asarray(im);m=np.zeros(a.shape[:2],dtype=np.uint8);cv2.grabCut(a,m,(2,2,max(1,im.width-4),max(1,im.height-4)),None,None,2,cv2.GC_INIT_WITH_RECT);mask=(m==1)|(m==3)
 if int(mask.sum())<20:mask=np.ones((im.height,im.width),dtype=bool)
 box=tight_box(mask);crop=im.crop(box);mask=mask[box[1]:box[3],box[0]:box[2]];x=features(crop,mask);old=baseline.probabilities(isolate(crop,mask)) if r['split']!='train' else torch.zeros(8)
 rows.append(dict(image=r['image'],split=r['split'],colour=r['colour'],x=x,baseline=old,mask_source='generic COCO instance mask' if indices else 'GrabCut fallback'))
 if len(rows)%25==0 or len(rows)==len(selected):torch.save(dict(rows=rows),cache);print('MASK FEATURES',len(rows),'/',len(selected),'seconds',round(time.perf_counter()-started),flush=True)
del net;data={s:dict(x=torch.stack([r['x'] for r in rows if r['split']==s]),y=torch.tensor([COLOURS.index(r['colour']) for r in rows if r['split']==s]),baseline=torch.stack([r['baseline'] for r in rows if r['split']==s])) for s in ['train','validation','test']}
mean=data['train']['x'].mean(0);std=data['train']['x'].std(0).clamp_min(.01)
for d in data.values():d['x']=(d['x']-mean)/std
network=head(len(mean));optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.03);best=-1;history=[]
for epoch in range(160):
 network.train()
 for ids in torch.randperm(len(data['train']['y'])).split(64):
  optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(network(data['train']['x'][ids]),data['train']['y'][ids],label_smoothing=.03);loss.backward();optimizer.step()
 network.eval()
 with torch.no_grad():p=network(data['validation']['x']).softmax(1);a=float((p.argmax(1)==data['validation']['y']).float().mean());score=a-.01*float(torch.nn.functional.nll_loss(p.log(),data['validation']['y']))
 if score>best:best=score;state=copy.deepcopy(network.state_dict());best_epoch=epoch+1
 history.append(dict(epoch=epoch+1,validation_accuracy=a))
network.load_state_dict(state);network.eval()
with torch.no_grad():logits=network(data['validation']['x']);truth=data['validation']['y']
temperature=min([.5,.75,1.,1.5,2.],key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,truth)));p=(logits/temperature).softmax(1);best=-1
for w in [0.,.25,.5,.75,1.]:
 q=p*w+data['validation']['baseline']*(1-w);a=float((q.argmax(1)==truth).float().mean());score=a-.01*float(torch.nn.functional.nll_loss(q.clamp_min(1e-8).log(),truth))
 if score>best:best=score;weight=w;vp=q
threshold=.8
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
 accept=vp.max(1).values>=t
 if int(accept.sum())>=40 and float((vp.argmax(1)[accept]==truth[accept]).float().mean())>=.95:threshold=t;break
with torch.no_grad():test=(network(data['test']['x'])/temperature).softmax(1)*weight+data['test']['baseline']*(1-weight);ty=data['test']['y'];accept=test.max(1).values>=threshold
info=dict(origin='project-trained-masked-paint-head',architecture='Own visible-pixel HSV/Lab distribution features, 96 hidden, eight source-labelled colours',classes=COLOURS,train=len(data['train']['y']),validation=len(truth),test=len(ty),best_epoch=best_epoch,temperature=temperature,weight=weight,threshold=threshold,validation_accuracy=float((vp.argmax(1)==truth).float().mean()),test_accuracy=float((test.argmax(1)==ty).float().mean()),test_coverage=float(accept.float().mean()),test_accepted_accuracy=float((test.argmax(1)[accept]==ty[accept]).float().mean()) if accept.any() else None,baseline_masked_test_accuracy=float((data['test']['baseline'].argmax(1)==ty).float().mean()),seconds=time.perf_counter()-started,mask_sources={s:sum(r['mask_source']==s for r in rows) for s in ['generic COCO instance mask','GrabCut fallback']},selection='Source train/validation/test preserved; deterministic balanced subsets, original VCoR labels; head epoch, temperature, fusion and rejection from validation only; city examples excluded',optimizer='AdamW .001, wd .03, 160 epochs, batch64, label smoothing .03',history=history)
checkpoint=dict(origin=info['origin'],classes=COLOURS,mean=mean,std=std,state=state,temperature=temperature,weight=weight,threshold=threshold)
torch.save(checkpoint,out/'masked_colour.pt');(out/'training.json').write_text(json.dumps(info,indent=2));print('FINAL',json.dumps({k:v for k,v in info.items() if k!='history'}),flush=True)
if weight>0:torch.save(checkpoint,ROOT/'models/masked_colour.pt');(ROOT/'models/masked_colour_training.json').write_text(json.dumps(info,indent=2));print('PROMOTED',flush=True)
