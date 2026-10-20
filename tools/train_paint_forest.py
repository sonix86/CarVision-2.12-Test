"""Own foreground-colour tree ensemble, validation-selected; sklearn used only to train."""
import sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from carvision.core import ROOT,COLOURS
from carvision.masked_colour import MaskedColourClassifier
started=time.perf_counter();torch.set_num_threads(2);old=torch.load(ROOT/'models/masked_colour.pt',weights_only=True);baseline=MaskedColourClassifier(old);rows=torch.load(ROOT/'cache/masked_colour_v15.pt',weights_only=True)['rows'];data={s:dict(x=torch.stack([r['x'] for r in rows if r['split']==s]),y=np.array([COLOURS.index(r['colour']) for r in rows if r['split']==s]),base=torch.stack([r['baseline'] for r in rows if r['split']==s]).numpy()) for s in ['train','validation','test']};mean=data['train']['x'].mean(0);std=data['train']['x'].std(0).clamp_min(.01)
x={s:((v['x']-mean)/std).numpy() for s,v in data.items()};y=data['validation']['y'];own=baseline.feature_probabilities(data['validation']['x']);original=np.stack([baseline.blend(torch.tensor(b),p).numpy() for b,p in zip(data['validation']['base'],own)]);floor=np.mean(original.argmax(1)==y);best=floor-.001*np.mean(-np.log(np.maximum(original[np.arange(len(y)),y],1e-8)));selected=None;history=[]
for depth in [5,9,None]:
 for leaf in [3,8]:
  forest=ExtraTreesClassifier(n_estimators=64,max_depth=depth,min_samples_leaf=leaf,max_features=.5,n_jobs=2,random_state=321015);forest.fit(x['train'],data['train']['y']);paint=forest.predict_proba(x['validation'])
  for weight in [.25,.5,.75,1.]:
   for gate in [None,dict(base_ceiling=.5,paint_floor=.7,weight=1.)]:
    q=paint*weight+data['validation']['base']*(1-weight)
    if gate:
     choose=(data['validation']['base'].max(1)<gate['base_ceiling'])&(paint.max(1)>=gate['paint_floor']);q[choose]=paint[choose]
    acc=np.mean(q.argmax(1)==y);score=acc-.001*np.mean(-np.log(np.maximum(q[np.arange(len(y)),y],1e-8)));history.append(dict(depth=depth,leaf=leaf,weight=weight,gate=gate,validation_accuracy=acc))
    if acc>=floor and score>best:best=score;selected=copy.deepcopy(forest);configuration=dict(depth=depth,leaf=leaf,weight=weight,gate=gate);vp=q
if selected is None:
 (ROOT/'models/paint_forest_experiment.json').write_text(json.dumps(dict(promoted=False,baseline_validation_accuracy=float(floor),candidates=history),indent=2));print('NO VALIDATION IMPROVEMENT',flush=True);raise SystemExit(0)
threshold=.9
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
 accept=vp.max(1)>=t
 if accept.sum()>=40 and np.mean(vp.argmax(1)[accept]==y[accept])>=.95:threshold=t;break
trees=[]
for estimator in selected.estimators_:
 t=estimator.tree_;value=t.value[:,0,:];value=value/value.sum(1,keepdims=True).clip(1e-8)
 trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),probability=value.astype('float32').tolist()))
checkpoint=dict(origin=old['origin'],architecture='masked_paint_extra_trees_v1',classes=COLOURS,mean=mean,std=std,trees=trees,weight=configuration['weight'],gate=configuration['gate'],threshold=threshold,temperature=1.)
model=MaskedColourClassifier(checkpoint);pred=model.feature_probabilities(data['test']['x']);q=torch.stack([model.blend(torch.tensor(b),p) for b,p in zip(data['test']['base'],pred)]).numpy();y=data['test']['y'];accept=q.max(1)>=threshold
info=dict(promoted=True,configuration=configuration,validation_accuracy=float(np.mean(vp.argmax(1)==data['validation']['y'])),baseline_validation_accuracy=float(floor),test_count=len(y),test_accuracy=float(np.mean(q.argmax(1)==y)),test_coverage=float(accept.mean()),test_accepted_accuracy=float(np.mean(q.argmax(1)[accept]==y[accept])),threshold=threshold,trees=64,train=640,validation=160,feature_count=len(mean),seconds=time.perf_counter()-started,selection='Coarse 6 own ExtraTrees ensembles and8 fixed fusion settings selected on validation only; no validation accuracy regression vs own MLP. No test or city example chooses weights.',test_used_for_selection=False,candidates=history)
backup=ROOT/'models/backups/v15_mlp';backup.mkdir(parents=True,exist_ok=True);torch.save(old,backup/'masked_colour.pt');torch.save(checkpoint,ROOT/'models/masked_colour.pt');(ROOT/'models/paint_forest_experiment.json').write_text(json.dumps(info,indent=2));print('PROMOTED',json.dumps({k:v for k,v in info.items() if k!='candidates'}),flush=True)
