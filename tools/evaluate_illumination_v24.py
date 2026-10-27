"""Single heldout paint evaluation of the validation-frozen calibrated mixture."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.illumination_colour import IlluminationColour
from carvision.core import COLOURS
out=ROOT/'models/multimodal_v24';assert not (out/'colour_evaluation.json').exists(),'Do not repeat heldout paint evaluation'
training=json.loads((out/'colour_training.json').read_text(encoding='utf-8'))
assert training['selected'],'No validation-approved colour candidate to test'
checkpoint=out/'illumination_colour.pt';expert=IlluminationColour(torch.load(checkpoint,weights_only=True))
rows=[r for r in torch.load(ROOT/'cache/illumination_v24.pt',weights_only=True)['items'].values() if r['split']=='test'];records=[]
torch.set_num_threads(2)
with torch.inference_mode():
    for r in rows:
        entry=dict(truth=r['colour'],group=r['group'])
        for kind in ('whole','warm','shadow'):
            v=r['variants'][kind];p=(expert.network((v['x']-expert.mean)/expert.std)/expert.temperature).softmax(0);q=expert.blend(v['baseline'],p)
            entry[kind]=dict(baseline=COLOURS[int(v['baseline'].argmax())],prediction=COLOURS[int(q.argmax())],confidence=float(q.max()))
        records.append(entry)
def metric(rs,kind):return dict(count=len(rs),baseline=sum(r[kind]['baseline']==r['truth'] for r in rs)/len(rs),candidate=sum(r[kind]['prediction']==r['truth'] for r in rs)/len(rs))
metrics={k:metric(records,k) for k in ('whole','warm','shadow')};base_stress=(metrics['warm']['baseline']+metrics['shadow']['baseline'])/2;new_stress=(metrics['warm']['candidate']+metrics['shadow']['candidate'])/2
checks=dict(whole=metrics['whole']['candidate']>=metrics['whole']['baseline']-.02,stress=new_stress>=base_stress+.03)
report=dict(passed=all(checks.values()),checks=checks,metrics=metrics,per_colour={c:{k:metric([r for r in records if r['truth']==c],k) for k in metrics} for c in COLOURS},records=records,candidate_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),test_used_for_selection=False,limitations='Existing VCoR source split, foreground masks. Warm/shadow are controlled synthetic views of the same photos, not independent Uzbekistan field photos.')
(out/'colour_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ('records','per_colour')}))
