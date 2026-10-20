"""Record Codex visual review of the fixed contact sheets, with ambiguous items excluded.

This is a small AI-reviewed diagnostic, not a substitute for human ground truth.
No predictions from the colour algorithm were consulted to choose these reference labels.
"""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'outputs/colour-review/references.pending.json').read_text())
labels=['white','red','blue',None,'grey',None,'brown','grey','white','white','white','grey','grey','blue','green',None,'black','blue','white','grey',None,None,'grey','grey','red','brown','white','red','red','white','white','grey','red','white','blue','red','grey','black','red','white','grey','grey','blue','black','red','white','blue','red','white','red',None,'red']
reasons={4:'Monochrome photograph: original paint colour cannot be established.',6:'Ambiguous brown/burgundy paint and lighting.',16:'Ambiguous mint blue/green.',21:'Purple/brown shade does not map unambiguously to the eight required classes.',22:'Strong warm light makes grey/brown ambiguous.',51:'Blue/green/grey cast makes the eight-class reference uncertain.'}
assert len(rows)==len(labels)
for r,label in zip(rows,labels):
    r['reference_colour']=label;r['annotator']='OpenAI Codex visual review of contact sheets, not student or independent human annotation'
    if label is None:r['excluded_reason']=reasons[r['id']]
result=dict(purpose='Small fixed held-out colour diagnostic, not a statistically representative human-labelled colour benchmark',reference_selection='Four official-test images per brand, fixed seed 510, selected before viewing algorithm predictions',references=rows)
(root/'data/public_prototype/colour_reference_review.json').write_text(json.dumps(result,indent=2))
