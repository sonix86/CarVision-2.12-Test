"""Select visible-instance evidence weights on grouped development only."""
import sys,json,time,hashlib,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image,ImageOps
from carvision.core import ROOT,Pipeline,read_image
from carvision.segmentation import tight_box,isolate
started=time.perf_counter();torch.set_num_threads(3);p=Pipeline(mode='fast');p.segmenter.network.transform.min_size=(256,);p.segmenter.network.transform.max_size=384;p.segmenter.network.roi_heads.detections_per_img=3;p.segmenter.network.rpn._pre_nms_top_n['testing']=200;p.segmenter.network.rpn._post_nms_top_n['testing']=50
rows=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()];selected=[r for r in rows if r['split']!='train'];train=[r for r in rows if r['split']=='train'];labels=list(zip(p.vehicle_classifier.brands,p.vehicle_classifier.labels));cache=ROOT/'cache/visible_vehicle_evidence.pt';data=[]
def model_name(r):return '2101' if r['vehicle_category']=='VAZ-2101' else re.sub(r' \(.*?\)','',r['vehicle_category']).removeprefix(r['brand']+' ').removesuffix(' sedan')
@torch.inference_mode()
def brand(image):return (p.classifier(torch.stack([p.preprocess(image),p.preprocess(ImageOps.mirror(image))]))[0]/p.brand_temperature).softmax(1).mean(0)
@torch.inference_mode()
def evidence(image,mask):
 focus=isolate(image,mask);return dict(brand_raw=brand(image),brand_mask=brand(focus),model_raw=p.vehicle_classifier.probabilities(image,p.classifier,p.preprocess),model_mask=p.vehicle_classifier.probabilities(focus,p.classifier,p.preprocess))
if cache.exists():data=torch.load(cache,weights_only=True)['rows'];assert all(a['image']==b['image'] for a,b in zip(data,selected))
for i,r in enumerate(selected[len(data):],len(data)):
 image=read_image(ROOT/'data/regional'/r['image']).crop(r['box']);image.thumbnail((384,384));d=p.segmenter.raw(image);ids=[k for k in range(len(d['scores'])) if int(d['labels'][k]) in (3,8)]
 mask=d['masks'][max(ids,key=lambda k:int((d['masks'][k,0]>.5).sum())),0].numpy()>.5 if ids else np.ones((image.height,image.width),dtype=bool)
 if int(mask.sum())<20:mask=np.ones((image.height,image.width),dtype=bool)
 b=tight_box(mask);image=image.crop(b);mask=mask[b[1]:b[3],b[0]:b[2]];clean=evidence(image,mask)
 donor=read_image(ROOT/'data/regional'/train[int(hashlib.sha256(r['image'].encode()).hexdigest()[:8],16)%len(train)]['image']).resize(image.size);contaminated=image.copy();block=(0,round(image.height*.4),round(image.width*.45),image.height) if i%2 else (round(image.width*.55),round(image.height*.4),image.width,image.height);contaminated.paste(donor.crop(block),block);partial=mask.copy();partial[block[1]:block[3],block[0]:block[2]]=False;occluded=evidence(contaminated,partial)
 data.append(dict(image=r['image'],split=r['split'],brand_truth=p.brands.index(r['brand']),model_truth=labels.index((r['brand'],model_name(r))),clean=clean,occluded=occluded))
 if len(data)%20==0 or len(data)==len(selected):torch.save(dict(rows=data),cache);print('VEHICLE EVIDENCE',len(data),'/',len(selected),flush=True)
def measure(split,head,weight):
 rs=[r for r in data if r['split']==split];ys=torch.tensor([r[head+'_truth'] for r in rs]);result={}
 for condition in ['clean','occluded']:
  prob=torch.stack([r[condition][head+'_raw']*(1-weight)+r[condition][head+'_mask']*weight for r in rs]);result[condition]=dict(accuracy=float((prob.argmax(1)==ys).float().mean()),top3_recall=float((prob.topk(3).indices==ys[:,None]).any(1).float().mean()),nll=float(torch.nn.functional.nll_loss(prob.clamp_min(1e-8).log(),ys)))
 return result
info=dict(selection='Grouped Commons validation only; optimize mean clean and synthetic-occlusion accuracy, clean accuracy floor baseline minus2 percentage points. Temperature and source classifiers already project-trained. Donor images come only from training groups. Tests and city samples do not select fusion.',validation_count=sum(r['split']=='validation' for r in data),test_count=sum(r['split']=='test' for r in data),synthetic_scope='One lower left/right rectangle covering27% image area replaced by a different training vehicle; removed from visible mask. Artificial benchmark, not a real street occlusion test.',test_used_for_selection=False)
for h in ['brand','model']:
 baseline=measure('validation',h,0);best=-1;history=[]
 for w in [0.,.25,.5,.75,1.]:
  a=measure('validation',h,w);score=(a['clean']['accuracy']+a['occluded']['accuracy'])/2-.005*(a['clean']['nll']+a['occluded']['nll']);history.append(dict(weight=w,**a))
  if a['clean']['accuracy']>=baseline['clean']['accuracy']-.02 and score>best:best=score;weight=w
 info[h+'_mask_weight']=weight;info[h+'_development_candidates']=history;info[h+'_test']=measure('test',h,weight);info[h+'_baseline_test']=measure('test',h,0)
info['seconds']=time.perf_counter()-started;(ROOT/'models/mask_fusion.json').write_text(json.dumps(info,indent=2));print('FINAL',json.dumps({k:v for k,v in info.items() if 'candidates' not in k}),flush=True)
