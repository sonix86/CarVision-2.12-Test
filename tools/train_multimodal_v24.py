"""Two finite supervised fits, validation selection, no test access or promotion."""
import collections,copy,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cache_multimodal_v24 import token

torch.set_num_threads(3);torch.manual_seed(240026);random.seed(240026)
out=ROOT/'models/multimodal_v24';assert not (out/'training.json').exists(),'Do not repeat completed training'
data_path=ROOT/'cache/multimodal_v24_data.pt';data=torch.load(data_path,weights_only=True);rows=data['rows'];brands=data['brands']
cache=torch.load(ROOT/'cache/multimodal_v24_features.pt',weights_only=True);text=torch.load(ROOT/'cache/vehicle_text_v24.pt',weights_only=True)
policy=json.loads((out/'policy.json').read_text(encoding='utf-8'));assert policy['max_training_rounds']==2
legacy=json.loads((ROOT/'models/make_scope_v23/training.json').read_text(encoding='utf-8'))['validation_baseline_body_head']
train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];results=[];best=None;started=time.time()

def matrix(rs,kind,with_text):
    xs=[]
    for r in rs:
        x=cache['items'][token(r,kind)]
        if with_text:
            key=hashlib.sha256(r['body_rgb'].numpy().tobytes()+r['body_mask'].numpy().tobytes()).hexdigest()
            # Explicit no-readable-text phone stress view; never reuse high-res text there.
            tx=text['items'][key]['features'] if kind=='whole' else torch.zeros(512)
            x=torch.cat([x,tx])
        xs.append(x)
    return torch.stack(xs)

def metrics(p,y,rs):
    good=p.argmax(1)==y;hist=torch.tensor([r['historical'] for r in rs]);classes=y.unique()
    return dict(top1=float(good.float().mean()),macro=float(torch.tensor([float(good[y==j].float().mean()) for j in classes]).mean()),historical=float(good[hist].float().mean()),count=len(rs))

for round_id,with_text in enumerate((False,True),1):
    x=torch.cat([matrix(train,k,with_text) for k in ('whole','phone')]);y=torch.tensor([r['y'] for r in train]*2)
    mean=x.mean(0);std=x.std(0).clamp_min(.025);x=(x-mean)/std
    vx={k:(matrix(val,k,with_text)-mean)/std for k in ('whole','phone')};vy=torch.tensor([r['y'] for r in val])
    head=torch.nn.Linear(x.shape[1],len(brands));optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
    counts=torch.bincount(y,minlength=len(brands)).float();class_weights=counts.clamp_min(1).pow(-.35);class_weights/=class_weights.mean()
    owner_weight=torch.tensor([4. if r.get('owner') else 1. for r in train]*2)
    history=[];best_score=-1;chosen=None
    for epoch in range(150):
        head.train()
        for ids in torch.randperm(len(x)).split(128):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[ids]),y[ids],weight=class_weights,label_smoothing=.02,reduction='none');loss=(loss*owner_weight[ids]).sum()/owner_weight[ids].sum();loss.backward();optimizer.step()
        head.eval()
        with torch.inference_mode():p={k:head(a).softmax(1) for k,a in vx.items()}
        m={k:metrics(p[k],vy,val) for k in p};score=.7*m['whole']['macro']+.3*m['phone']['macro']
        guard=all(m[k]['historical']>=legacy[k]['top1']-.02 for k in m)
        record=dict(epoch=epoch+1,score=score,guard=guard,metrics=m);history.append(record)
        if guard and score>best_score:best_score=score;chosen=dict(head=copy.deepcopy(head.state_dict()),epoch=epoch+1,metrics=m)
        if epoch%25==0:print('ROUND',round_id,json.dumps(record),flush=True)
    result=dict(round=round_id,with_text=with_text,selected_epoch=chosen['epoch'] if chosen else None,score=best_score,history=history);results.append(result)
    if chosen and (best is None or best_score>best['score']):best=dict(chosen,score=best_score,mean=mean,std=std,with_text=with_text,vx=vx,round=round_id)
    (out/'training_progress.json').write_text(json.dumps(dict(rounds=results)),encoding='utf-8')
assert best is not None,'No candidate retained old validation quality; no release'
head=torch.nn.Linear(len(best['mean']),len(brands)).eval();head.load_state_dict(best['head'])
with torch.inference_mode():logits=torch.cat([head(best['vx'][k]) for k in ('whole','phone')])
labels=vy.repeat(2);group_names=sorted({r['group'] for r in val});groups=torch.tensor([group_names.index(r['group']) for r in val]*2)
geometry=[]
for kind in ('whole','phone'):
    for r in val:
        height,width=r['body_rgb'].shape[:2];scale=min(1,96/max(width,height)) if kind=='phone' else 1
        geometry.append(min(round(width*scale),round(height*scale))>=40)
geometry=torch.tensor(geometry,dtype=torch.bool)
temperature=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,labels)))
p=(logits/temperature).softmax(1);values,ids=p.topk(2,dim=1);thresholds={};calibration={};accepted=torch.zeros(len(p),dtype=torch.bool)
for j,b in enumerate(brands):
    options=[]
    for gate in (.5,.6,.7,.8,.9,.95,.98):
        eligible=(ids[:,0]==j)&(values[:,0]>=gate)&((values[:,0]-values[:,1])>=.06)&geometry
        if len(groups[eligible].unique())>=3 and float((labels[eligible]==j).float().mean())>=.9:options.append((int(eligible.sum()),gate))
    _,gate=max(options,default=(0,1.));thresholds[b]=gate
    eligible=(ids[:,0]==j)&(values[:,0]>=gate)&((values[:,0]-values[:,1])>=.06)&geometry if gate<1 else torch.zeros(len(p),dtype=torch.bool)
    accepted|=eligible;calibration[b]=dict(threshold=gate,views=int(eligible.sum()),groups=len(groups[eligible].unique()))
weights=torch.load(ROOT/'cache/imagenet/convnext_tiny-983f1562.pth',weights_only=True);weights={k:v for k,v in weights.items() if not k.startswith('classifier.2.')}
checkpoint=dict(origin='project-trained-multimodal-make-v24',brands=brands,encoder_state=weights,head=best['head'],mean=best['mean'],std=best['std'],with_text=best['with_text'],temperature=temperature,thresholds=thresholds,margin=.06,dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),encoder_source_sha256=cache['source_sha256'])
torch.save(checkpoint,out/'multimodal_make.pt')
report=dict(selected_round=best['round'],selected_epoch=best['epoch'],selected_metrics=best['metrics'],rounds=results,calibration=calibration,temperature=temperature,accepted_validation_views=int(accepted.sum()),accepted_validation_precision=float((ids[accepted,0]==labels[accepted]).float().mean()) if accepted.any() else None,confirmed_make_count=sum(t<1 for t in thresholds.values()),dataset_sha256=checkpoint['dataset_sha256'],source_encoder_sha256=cache['source_sha256'],seconds=time.time()-started,test_used_for_selection=False,policy=policy)
(out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('TRAINING COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('rounds','calibration','policy')}),flush=True)
