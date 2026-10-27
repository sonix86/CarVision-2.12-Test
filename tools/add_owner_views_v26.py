"""Explicitly reviewed additional own-photo boxes; no guessed model labels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
pool=ROOT/'data/owner_oct25';manifest=json.loads((pool/'manifest.json').read_text(encoding='utf-8'));original=[json.loads(s) for s in (pool/'labels.jsonl').read_text(encoding='utf-8').splitlines()]
# These full-resolution images were viewed, not labelled from model output.
# Unclear emblems (including photo70 and blurred photo85) are deliberately absent.
review=[(75,'GAC Aion','white',[98,210,887,929],'Clear GAC emblem; Aion S exterior, exact variant not certified'),
        (77,'Li Auto','grey',[371,462,960,1020],'Li emblem on bonnet; foreground SUV only'),
        (80,'Leapmotor','white',[259,218,875,803],'Clear Leapmotor emblem on bonnet'),
        (88,'BYD','grey',[178,142,679,729],'BYD lettering across front trim'),
        (91,'BYD','black',[377,340,960,1100],'BYD lettering on rear; different vehicle from locked photo5'),
        (97,'BYD','white',[177,330,842,925],'BYD rear lettering in night photo; motion blur retained')]
labels=[]
for i,brand,colour,box,evidence in review:
    r=manifest[i];assert i not in (6,32,54)
    labels.append(dict(image=r['image'],box=box,brand=brand,colour=colour,source=r['source'],original_sha256=r['sha256'],group='owner-phone-outing-2026-10-25',split='train',label_source='Assistant full-resolution visible-emblem review; not independently human-certified',status='additional_v26_reviewed_make_only',exclude_from_independent_evaluation=True,model='unknown',review=evidence))
out=ROOT/'models/ev_views_v26';out.mkdir(exist_ok=True)
with (out/'owner_additions.json').open('x',encoding='utf-8') as f:json.dump(labels,f,ensure_ascii=False,indent=2)
print('Saved',len(labels),'additional TRAIN-only annotations; original labels and photos unchanged')
