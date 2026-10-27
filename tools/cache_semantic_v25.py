"""General vision features with an immutable source binding and preserved splits."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import GeneralVisionEncoder,visual_inputs

def token(row,kind):return hashlib.sha256(row['body_rgb'].numpy().tobytes()+row['body_mask'].numpy().tobytes()+('semantic-v25-'+kind).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-test',action='store_true');args=parser.parse_args();torch.set_num_threads(3)
    out=ROOT/'models/semantic_v25';source=torch.load(ROOT/'cache/semantic_v25/image_encoder.pt',weights_only=True)
    if args.include_test:assert json.loads((out/'validation_gate.json').read_text(encoding='utf-8'))['passed'],'Validation gate must pass before test extraction'
    data_path=ROOT/'cache/multimodal_v24_data.pt';data=torch.load(data_path,weights_only=True)
    rows=[r for r in data['rows'] if args.include_test or r['split'] in ('train','validation')]
    path=ROOT/'cache/semantic_v25/features.pt';implementation_sha=hashlib.sha256((ROOT/'carvision/semantic_make.py').read_bytes()).hexdigest()
    cache=torch.load(path,weights_only=True) if path.exists() else dict(source_sha256=source['source_sha256'],implementation_sha256=implementation_sha,items={})
    assert cache['source_sha256']==source['source_sha256'] and cache['implementation_sha256']==implementation_sha
    encoder=GeneralVisionEncoder().eval();encoder.load_state_dict(source['state']);started=time.time();jobs=[(r,k) for r in rows for k in ('whole','phone') if token(r,k) not in cache['items']]
    with torch.inference_mode():
        for start in range(0,len(jobs),4):
            batch=jobs[start:start+4];x=torch.cat([visual_inputs(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy(),k) for r,k in batch]);z=encoder(x).reshape(-1,2,512)
            for j,(r,k) in enumerate(batch):cache['items'][token(r,k)]=z[j].clone()
            if start%100==0:
                torch.save(cache,path);report=dict(completed=start+len(batch),total=len(jobs),seconds=time.time()-started);(out/'feature_progress.json').write_text(json.dumps(report),encoding='utf-8');print(json.dumps(report),flush=True)
    torch.save(cache,path);(out/'features.json').write_text(json.dumps(dict(complete=True,items=len(cache['items']),test_requested=args.include_test,dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),source_sha256=source['source_sha256'],implementation_sha256=implementation_sha),indent=2),encoding='utf-8');print('SEMANTIC FEATURES COMPLETE',flush=True)

if __name__=='__main__':main()
