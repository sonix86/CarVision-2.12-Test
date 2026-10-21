"""Immutable clean evaluation view; no former heldout image is moved to train."""
import collections,json
import torch
from carvision.core import ROOT

def excluded_legacy_images():
 rows=[json.loads(l) for l in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()];groups=collections.defaultdict(set)
 for r in rows:groups[r['group']].add(r['split'])
 bad={g for g,splits in groups.items() if len(splits)>1}
 excluded={r['image'] for r in rows if r['group'] in bad and r['split']!='train'}
 info=dict(conflicting_sessions=len(bad),excluded_validation=sum(r['image'] in excluded and r['split']=='validation' for r in rows),excluded_test=sum(r['image'] in excluded and r['split']=='test' for r in rows),excluded_images=sorted(excluded),policy='Keep old files/splits immutable. Exclude non-training members of cross-brand author/date sessions from all new validation/test views. No old heldout photo is moved into training.',remaining_old_counts=dict(train=642,validation=131,test=389),limitations='Author/date is an approximate session. True vehicle identity labels are unavailable for these public sources.')
 return excluded,info

def clean_v18_baseline(split='validation'):
 """Recompute fixed v1.8 validation on the exact cleaned cohort, not a new fit."""
 from carvision.visual_parts import VisualReferences
 cache=torch.load(ROOT/'cache/visual_parts_v18.pt',weights_only=True)
 ck=torch.load(ROOT/'models/visual_references.pt',weights_only=True)
 if cache['model_sha']!=ck['source_model_sha256']:raise ValueError('Fixed baseline feature/weight mismatch; do not recalibrate using candidate/test')
 bank=VisualReferences(ck);excluded,audit=excluded_legacy_images()
 source={r['image']:r for r in [json.loads(l) for l in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]}
 for r in json.loads((ROOT/'models/expanded_dataset.json').read_text(encoding='utf-8'))['new_rows']:source['stanford/'+r['image']]={'box':r['box']}
 scopes={'whole':(1,1),'upper55':(1,.55),'upper35':(1,.35)};result={}
 for kind,(width,height) in scopes.items():
  rows=[r for r in cache[split] if r['kind']==kind and r['image'] not in excluded];probs=[];targets=[]
  for a,b in zip(rows[::2],rows[1::2]):
   box=source[a['image']]['box'];ratio=(box[2]-box[0])/(box[3]-box[1])*width/height
   pa=a['prob']*(1-bank.weight)+bank.probabilities(a['feature'],ratio)*bank.weight
   pb=b['prob']*(1-bank.weight)+bank.probabilities(b['feature'],ratio)*bank.weight
   probs.append(pa*.75+pb*.25);targets.append(a['y'])
  p=torch.stack(probs);y=torch.tensor(targets)
  result[kind]=dict(count=len(y),top1=float((p.argmax(1)==y).float().mean()),top3=float((p.topk(3).indices==y[:,None]).any(1).float().mean()))
 name='clean_legacy_baseline.json' if split=='validation' else 'clean_legacy_test_baseline.json'
 (ROOT/'models'/name).write_text(json.dumps({split:result,'audit':audit,'parameters':'Fixed shipped1.8 reference parameters, no candidate or test tuning'},indent=2),encoding='utf-8')
 return result
