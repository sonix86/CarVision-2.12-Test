"""Read-only audit of truth provenance; predictions never become labels."""
import collections,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import embedding_features
from tools.cache_semantic_v25 import token

def main():
    torch.set_num_threads(3);out=ROOT/'outputs/user-error-audit-v30';out.mkdir(exist_ok=True);cache=ROOT/'cache/truth_v30';cache.mkdir(exist_ok=True)
    d=torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True);rows=d['rows'];cf=torch.load(ROOT/'cache/ev_views_v26_features.pt',weights_only=True)['items'];df=torch.load(ROOT/'cache/dino_v28/features.pt',weights_only=True)['items'];own=torch.load(ROOT/'cache/owner_domain_v29/features.pt',weights_only=True);cf.update(own['clip']);df.update(own['dino'])
    ck=torch.load(ROOT/'outputs/CarVision-2.9-Test/CarVision/models/vehicle_model.pt',weights_only=True);pairs=list(zip(ck['brands'],ck['labels']));meta=[]
    for r in rows:
        m={k:v for k,v in r.items() if k not in ('body_rgb','body_mask')};m['shape']=list(r['body_rgb'].shape);m['model_pair_supported']=(r['brand'],r.get('family')) in pairs;meta.append(m)
    (cache/'rows.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    grouped=collections.defaultdict(collections.Counter)
    for r in rows:grouped[str((r['brand'],r.get('family'))) ][r['split']]+=1
    report=dict(pairs=pairs,counts=dict(grouped),supported_rows=sum(r['model_pair_supported'] for r in meta),total=len(rows),source_examples=[{k:v for k,v in r.items() if k not in ('body_rgb','body_mask')} for r in rows[:5]],label_provenance='Source metadata/manual review, not classifier output. Owner v29 family unknown. Exact-family metadata needs separate review.')
    (out/'family_truth_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    features=torch.cat([torch.stack([torch.nn.functional.normalize(torch.cat([df[token(r,k)],embedding_features(cf[token(r,k)],False)]),dim=0) for r in rows]) for k in ('whole','phone')])
    torch.save(dict(x=features,brands=d['brands'],pairs=pairs),cache/'joint.pt')
    print(json.dumps(dict(total=len(rows),supported=report['supported_rows'],counts={k:v for k,v in grouped.items() if any(s in k for s in ('Dongfeng','Song','Han','Seagull','Dolphin'))},examples=report['source_examples']),ensure_ascii=True),flush=True)
if __name__=='__main__':main()
