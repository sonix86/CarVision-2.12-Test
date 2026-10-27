"""Predeclare unseen-family groups before any make-head fitting or evaluation."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'models/make_scope_v23'
assert not (OUT/'training.json').exists()
assert not (OUT/'family_probe.json').exists(),'Do not redefine the heldout family probe'
info=json.loads((OUT/'dataset.json').read_text(encoding='utf-8'))
old=json.loads((ROOT/'models/ev_patch_v22/dataset.json').read_text(encoding='utf-8'))
family_key=lambda name:name.split()[0].lower()
old_pairs={(b,family_key(m)) for b,m in old['pairs']};rows=info['new_rows']
legacy=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
legacy+=json.loads((ROOT/'models/ev_parts_candidate/dataset.json').read_text(encoding='utf-8'))['new_rows']+old['new_rows']
locked={r['group'] for r in legacy};selected=[];groups=set()
for brand in sorted({r['brand'] for r in rows}):
    if brand in {'Audi','Ford','Mercedes-Benz','Volkswagen','Volvo','Tesla'}:continue
    subset=[r for r in rows if r['brand']==brand]
    # The collection was reviewed for make, not exact family. Broad Commons
    # categories sometimes contain sibling bodies: do not call them unseen
    # families without a separate verified model annotation.
    if not all(r.get('family_verified',False) for r in subset):continue
    if any(r['family'].startswith('(') for r in subset):continue
    families={family_key(r['family']) for r in subset}
    if len(families)<3:continue
    candidates=[]
    for family in families:
        if (brand,family) in old_pairs:continue
        these={r['group'] for r in subset if family_key(r['family'])==family};others={r['group'] for r in subset if family_key(r['family'])!=family}
        if these&locked or len(these)<3 or len(others-these)<6:continue
        candidates.append((hashlib.sha256((brand+':'+family).encode()).hexdigest(),family,these))
    if candidates:
        _,family,reserved=min(candidates)
        selected.append(dict(brand=brand,family_prefix=family,source_families=sorted({r['family'] for r in subset if family_key(r['family'])==family}),groups=sorted(reserved)));groups|=reserved
report=dict(source_manifest_sha256=info['source_manifest_sha256'],families=selected,groups=sorted(groups),
            family_annotation_requirement='Explicit family_verified labels required. Make-only reviews cannot establish unseen-family generalization.',
            policy='Conservative family groups by first model-name token, so sibling trims are not separately called unseen. Six supplementary Stanford makes excluded because their exact training families are unavailable. All new author/date sessions containing a chosen group held out across brands. No historical split moved. Fixed before fitting, never chosen by test accuracy.')
(OUT/'family_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(families=len(selected),groups=len(groups))))
