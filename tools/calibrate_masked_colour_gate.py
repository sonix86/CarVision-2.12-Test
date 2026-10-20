"""Confidence fusion selected only on the declared masked-paint development split."""
import sys,json,copy,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS
from carvision.masked_colour import MaskedColourClassifier
torch.set_num_threads(2);checkpoint=torch.load(ROOT/'models/masked_colour.pt',weights_only=True);model=MaskedColourClassifier(checkpoint);rows=torch.load(ROOT/'cache/masked_colour_v15.pt',weights_only=True)['rows']
@torch.inference_mode()
def data(split):
 r=[r for r in rows if r['split']==split];x=torch.stack([r['x'] for r in r]);return (model.head((x-model.mean)/model.std)/model.temperature).softmax(1),torch.stack([r['baseline'] for r in r]),torch.tensor([COLOURS.index(r['colour']) for r in r])
p,b,y=data('validation');base=p*model.weight+b*(1-model.weight);minimum=float((base.argmax(1)==y).float().mean());best=minimum-.001*float(torch.nn.functional.nll_loss(base.clamp_min(1e-8).log(),y));selected=None;vp=base
for ceiling in [.5,.6,.7]:
 for floor in [.6,.75,.9]:
  for weight in [.5,.75,1.]:
   gate=(b.max(1).values<ceiling)&(p.max(1).values>=floor);q=base.clone();q[gate]=b[gate]*(1-weight)+p[gate]*weight;a=float((q.argmax(1)==y).float().mean());score=a-.001*float(torch.nn.functional.nll_loss(q.clamp_min(1e-8).log(),y))
   if a>=minimum and score>best:best=score;selected=dict(base_ceiling=ceiling,paint_floor=floor,weight=weight);vp=q
threshold=.9
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9]:
 accept=vp.max(1).values>=t
 if int(accept.sum())>=40 and float((vp.argmax(1)[accept]==y[accept]).float().mean())>=.95:threshold=t;break
checkpoint['gate']=selected;checkpoint['threshold']=threshold;model=MaskedColourClassifier(checkpoint);p,b,y=data('test');q=torch.stack([model.blend(base,paint) for base,paint in zip(b,p)]);accept=q.max(1).values>=threshold
info=dict(gate=selected,validation_accuracy=float((vp.argmax(1)==data('validation')[2]).float().mean()),threshold=threshold,test_accuracy=float((q.argmax(1)==y).float().mean()),test_count=len(y),test_coverage=float(accept.float().mean()),test_accepted_accuracy=float((q.argmax(1)[accept]==y[accept]).float().mean()),selection='Fixed coarse confidence gates: choose on160 validation only, no validation accuracy regression vs fixed fusion. Test240 and city scenes excluded from gate selection.',test_used_for_selection=False)
torch.save(checkpoint,ROOT/'models/masked_colour.pt');(ROOT/'models/masked_colour_gate.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
