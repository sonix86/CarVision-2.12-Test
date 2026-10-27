"""Extract only permitted split features from a general, unlabelled backbone."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.dino_features import DinoEncoder,dino_input,descriptor
from tools.cache_semantic_v25 import token

def main():
    p=argparse.ArgumentParser();p.add_argument('--include-test',action='store_true');args=p.parse_args()
    torch.set_num_threads(3);out=ROOT/'models/dino_v28';cache=ROOT/'cache/dino_v28'
    if args.include_test:assert json.loads((out/'validation.json').read_text())['passed']
    assert json.loads((out/'backbone_parity.json').read_text())['passed']
    policy=out/'policy.json';assert policy.exists()
    data_path=ROOT/'cache/ev_views_v26_data.pt';data=torch.load(data_path,weights_only=True)
    rows=[r for r in data['rows'] if args.include_test or r['split'] in ('train','validation')]
    source=json.loads((cache/'source.json').read_text());impl=hashlib.sha256((ROOT/'carvision/dino_features.py').read_bytes()).hexdigest()
    path=cache/'features.pt';features=torch.load(path,weights_only=True) if path.exists() else dict(source_sha256=source['sha256'],implementation_sha256=impl,items={})
    assert features['source_sha256']==source['sha256'] and features['implementation_sha256']==impl
    encoder=DinoEncoder().eval();encoder.load_state_dict(torch.load(cache/'dinov2_vits14_pretrain.pth',weights_only=True))
    jobs=[(r,k) for k in ('whole','phone') for r in rows if token(r,k) not in features['items']];start=time.time()
    with torch.inference_mode():
        for first in range(0,len(jobs),8):
            batch=jobs[first:first+8];x=torch.stack([dino_input(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy(),k) for r,k in batch]);z=descriptor(encoder(x))
            for i,(r,k) in enumerate(batch):features['items'][token(r,k)]=z[i].clone()
            if first%80==0:
                torch.save(features,path);progress=dict(completed=first+len(batch),total=len(jobs),seconds=time.time()-start,test_requested=args.include_test)
                (out/'feature_progress.json').write_text(json.dumps(progress),encoding='utf-8');print(json.dumps(progress),flush=True)
    torch.save(features,path)
    report=dict(complete=True,new_jobs=len(jobs),total_items=len(features['items']),test_requested=args.include_test,dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),implementation_sha256=impl,source_sha256=source['sha256'],seconds=time.time()-start)
    (out/'features.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('DINO FEATURES COMPLETE',report['seconds'],flush=True)

if __name__=='__main__':main()
