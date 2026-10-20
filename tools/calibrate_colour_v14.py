"""Select confidence fusion and rejection on development, then measure untouched VCoR test."""
import sys,json,itertools,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from carvision.core import ROOT,COLOURS,read_image
from carvision.colour_model import ColourClassifier
torch.set_num_threads(2)
d=torch.load(ROOT/'cache/colour_panel_probabilities.pt',weights_only=True);x=d['probabilities'];y=d['truth'];n=d['validation_count']
base=torch.load(ROOT/'models/colour_model.pt',weights_only=True)
weights=torch.tensor([sum(e['weight'] for e in base['mixture'] if e['component']==v['component'] and e['region']==v['region']) for v in d['descriptions']]);p=(x*weights[:,None,None]).sum(0);minimum=float((p[:n].argmax(1)==y[:n]).float().mean());best_score=-1
for counts in itertools.product(range(5),repeat=len(x)):
 if sum(counts)!=4:continue
 w=torch.tensor(counts,dtype=torch.float32)/4
 for power in [0.,1.,2.,4.]:
  effective=w[:,None]*x.max(2).values.pow(power);p=(x*effective[:,:,None]).sum(0)/effective.sum(0)[:,None].clamp_min(1e-8);a=float((p[:n].argmax(1)==y[:n]).float().mean());b=float((p[n:].argmax(1)==y[n:]).float().mean());score=(a+b)/2
  if a+1e-8>=minimum and score>best_score:best_score=score;best=dict(weights=w.tolist(),confidence_power=power,validation_accuracy=a,development_accuracy=b);best_p=p
base['mixture']=[dict(desc,weight=w) for desc,w in zip(d['descriptions'],best['weights']) if w>0];base['confidence_power']=best['confidence_power']
# Only first two own heads occur in this experiment, matching existing component order.
torch.save(base,ROOT/'models/colour_model.pt')
threshold=.95
for t in [v/100 for v in range(50,96,5)]:
 accept=best_p[:n].max(1).values>=t
 if int(accept.sum())>=100 and float((best_p[:n].argmax(1)[accept]==y[:n][accept]).float().mean())>=.97:threshold=t;break
accept=best_p[:n].max(1).values>=threshold
info=dict(best=best,threshold=threshold,validation_count=n,validation_reliable_coverage=float(accept.float().mean()),validation_reliable_accuracy=float((best_p[:n].argmax(1)[accept]==y[:n][accept]).float().mean()),selection='Full/body panel mixture, confidence power in 0/1/2/4; VCoR validation accuracy must not fall below v1.3. Maximize mean validation and declared 23-image development accuracy. No city photo or test used for selection.',test_used_for_selection=False)
model=ColourClassifier(base);rows=[r for r in json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text())['rows'] if r['split']=='test'];pred=[]
for r in rows:pred.append(model.probabilities(read_image(ROOT/'datasets/vcor/images'/r['image'])))
p=torch.stack(pred);y=torch.tensor([COLOURS.index(r['colour']) for r in rows]);accept=p.max(1).values>=threshold
matrix=torch.zeros((8,8),dtype=torch.long)
for truth,prediction in zip(y,p.argmax(1)):matrix[int(truth),int(prediction)]+=1
info['test_confusion_matrix']=matrix.tolist()
info.update(test_count=len(y),test_accuracy=float((p.argmax(1)==y).float().mean()),test_reliable_coverage=float(accept.float().mean()),test_reliable_accuracy=float((p.argmax(1)[accept]==y[accept]).float().mean()))
(ROOT/'models/colour_calibration.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
