"""One finite make-only fit; model-name weights stay unchanged. Validation selection only."""
import argparse,collections,copy,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.make_expert import make_head
from tools.make_scope_features_v23 import build,key,TRAIN_VIEWS,EVAL_VIEWS
from tools.make_taxonomy_v23 import DISPLAY,ALIASES
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
torch.set_num_threads(3);torch.manual_seed(321023);random.seed(321023)
parser=argparse.ArgumentParser();parser.add_argument('--fine-tune',action='store_true');args=parser.parse_args()
OUT=ROOT/'models/make_scope_v23';dataset=ROOT/'cache/make_scope_v23.pt'
OUT.mkdir(exist_ok=True)
assert not (OUT/'training.json').exists(),'A completed make fit already exists; do not silently overwrite it'
if not dataset.exists():dataset=ROOT/'Development/TrainingInputs/make_scope_v23.pt'
review_path=OUT/'review_approval.json'
if not review_path.exists():review_path=ROOT/'Development/TrainingInputs/review_approval.json'
review=json.loads(review_path.read_text(encoding='utf-8'))
assert review['dataset_sha256']==hashlib.sha256(dataset.read_bytes()).hexdigest() and review['reviewed']
data=torch.load(dataset,weights_only=True);rows=data['rows'];brands=data['brands'];n=len(brands)
source=ROOT/'models/ev_patch_v22/vehicle_model.pt'
if not source.exists():source=ROOT/'models/vehicle_model.pt'
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert review.get('source_model_sha256',source_sha)==source_sha
checkpoint=torch.load(source,weights_only=True)
started=time.perf_counter()
encoder,cache=build(rows,checkpoint,source_sha,ROOT/'cache/make_scope_v23_features.pt')

def tensors(split,kind='whole'):
    rs=[r for r in rows if r['split']==split]
    return torch.stack([cache['items'][key(r,kind)]['feature'] for r in rs]),torch.tensor([r['y'] for r in rs]),rs
val={kind:tensors('validation',kind) for kind in EVAL_VIEWS}
train=[r for r in rows if r['split']=='train'];x=torch.stack([cache['items'][key(r,k)]['feature'] for r in train for k in TRAIN_VIEWS]);y=torch.tensor([r['y'] for r in train for k in TRAIN_VIEWS])
historic_train=torch.tensor([r['historical'] for r in train for k in TRAIN_VIEWS])
teacher_rows=torch.stack([cache['items'][key(r,k)]['old'] for r in train for k in TRAIN_VIEWS])
teacher=torch.zeros(len(x),n)
for j,b in enumerate(checkpoint['brands']):teacher[:,brands.index(b)]+=teacher_rows[:,j]
old_brands=set(checkpoint['brands']);historic=torch.tensor([r['historical'] for r in val['whole'][2]])
def measure(p,y,idx=None):
    idx=torch.ones(len(y),dtype=torch.bool) if idx is None else idx
    good=p.argmax(1)==y
    return dict(count=int(idx.sum()),top1=float(good[idx].float().mean()),macro=float(torch.tensor([float(good[(y==j)&idx].float().mean()) for j in y[idx].unique()]).mean()))
baseline={}
for kind,(vx,vy,rs) in val.items():
    old=torch.stack([cache['items'][key(r,kind)]['old'] for r in rs]);p=torch.zeros(len(rs),n)
    for j,b in enumerate(checkpoint['brands']):p[:,brands.index(b)]+=old[:,j]
    baseline[kind]=measure(p,vy,historic)
head=make_head(n);optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.03)
counts=torch.bincount(y,minlength=n).float();weights=counts.clamp_min(1).pow(-.65);weights/=weights.mean()
history=[];best=-1.;best_epoch=0
for epoch in range(120):
    head.train();indices=torch.randperm(len(x))
    for ids in indices.split(128):
        optimizer.zero_grad();z=head(x[ids]);loss=torch.nn.functional.cross_entropy(z,y[ids],weight=weights,label_smoothing=.04)
        old=historic_train[ids]
        if old.any():loss=loss+.3*torch.nn.functional.kl_div(z[old].log_softmax(1),teacher[ids[old]],reduction='batchmean')
        loss.backward();optimizer.step()
    head.eval()
    with torch.inference_mode():
        probs={kind:head(v[0]).softmax(1) for kind,v in val.items()}
    metrics={kind:dict(all=measure(p,val[kind][1]),historical=measure(p,val[kind][1],historic)) for kind,p in probs.items()}
    score=sum(metrics[kind]['all']['macro']*dict(whole=.5,phone=.25,upper=.25)[kind] for kind in metrics)
    guard=all(metrics[k]['historical']['top1']>=baseline[k]['top1']-.02 for k in metrics)
    if guard and score>best:
        best=score;best_epoch=epoch+1;best_state=copy.deepcopy(head.state_dict())
    record=dict(epoch=epoch+1,score=score,guard=guard,metrics=metrics);history.append(record)
    if epoch%10==0:print(json.dumps(record),flush=True)
    (OUT/'training_progress.json').write_text(json.dumps(dict(best_epoch=best_epoch,history=history)),encoding='utf-8')
if not best_epoch:raise SystemExit('All make-head candidates failed historical validation guards; do not release.')
head.load_state_dict(best_state);head.eval()
second_round=dict(epochs=0,selected=False)
if args.fine_tune:
    from tools.finetune_make_scope_v23 import refine
    encoder,head,val,second_round=refine(encoder,head,rows,cache,val,baseline,measure,weights,best,OUT)
vx,vy,rs=val['whole']
with torch.inference_mode():logits=torch.cat([head(val[k][0]) for k in EVAL_VIEWS])
calibration_y=vy.repeat(len(EVAL_VIEWS));photo_ids=torch.arange(len(vy)).repeat(len(EVAL_VIEWS))
group_names=sorted({r['group'] for r in rs});group_ids=torch.tensor([group_names.index(r['group']) for r in rs]).repeat(len(EVAL_VIEWS))
temperatures=[.5,.75,1.,1.5,2.,3.]
temperature=min(temperatures,key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,calibration_y)))
p=(logits/temperature).softmax(1);values,indices=p.topk(2,dim=1);confidence=values[:,0];margin=values[:,0]-values[:,1];guess=indices[:,0]
options=[]
for gate in [.5,.6,.7,.75,.8,.85,.9,.95,.98]:
    for m in [.05,.1,.15,.2]:
        idx=(confidence>=gate)&(margin>=m)
        if len(group_ids[idx].unique())>=20 and float((guess[idx]==calibration_y[idx]).float().mean())>=.95:options.append((int(idx.sum()),gate,m))
_,global_gate,margin_gate=max(options,default=(0,1.,.2))
thresholds={};calibration={}
for j,b in enumerate(brands):
    choices=[]
    for gate in sorted({global_gate,.75,.8,.85,.9,.95,.98}):
        if gate<global_gate:continue
        idx=(guess==j)&(confidence>=gate)&(margin>=margin_gate)
        if len(group_ids[idx].unique())>=3 and float((calibration_y[idx]==j).float().mean())>=.9:choices.append((int(idx.sum()),gate))
    count,threshold=max(choices,default=(0,1.));thresholds[b]=threshold
    if threshold>=1.:count=0
    accepted=(guess==j)&(confidence>=threshold)&(margin>=margin_gate) if threshold<1 else torch.zeros(len(guess),dtype=torch.bool)
    calibration[b]=dict(accepted_validation_views=count,accepted_validation_photos=len(photo_ids[accepted].unique()),accepted_validation_groups=len(group_ids[accepted].unique()),threshold=threshold,total_validation_photos=int((vy==j).sum()))
result=dict(origin='project-trained-make-expert-v1',brands=brands,encoder=encoder.state_dict(),head=head.state_dict(),display_names=DISPLAY,aliases=ALIASES,
            temperature=temperature,thresholds=thresholds,margin=margin_gate,source_body_sha256=source_sha,training_dataset_sha256=review['dataset_sha256'])
torch.save(result,OUT/'make_expert.pt')
report=dict(selected=True,best_epoch=best_epoch,epochs=120,validation_baseline_body_head=baseline,history=history,
            second_round=second_round,
            recipe_sha256={name:hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest() for name in ['train_make_scope_v23.py','finetune_make_scope_v23.py','make_scope_features_v23.py']},
            predeclared_release_policy_sha256=hashlib.sha256((OUT/'predeclared_release_policy.json').read_bytes()).hexdigest(),
            calibration=calibration,global_gate=global_gate,margin=margin_gate,temperature=temperature,
            source_body_sha256=source_sha,dataset_sha256=review['dataset_sha256'],seconds=time.perf_counter()-started,
            test_used_for_selection=False,limitation='Project-trained body encoder, make-only fit, at most two rounds; no new model-name classes. Public session heldouts are not independent phone field validation.')
(OUT/'training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',best_epoch,flush=True)
