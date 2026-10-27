"""Reuse pixel-equivalent frozen features; extract new TRAIN/VAL only by default."""
import argparse,ast,hashlib,json,sys,time
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import GeneralVisionEncoder,visual_inputs
from tools.cache_semantic_v25 import token
parser=argparse.ArgumentParser();parser.add_argument('--include-test',action='store_true');args=parser.parse_args();torch.set_num_threads(3);out=ROOT/'models/ev_views_v26'
if args.include_test:assert json.loads((out/'validation_gate.json').read_text())['passed'],'Freeze validation decision first'
def pixel_ast(path):
    tree=ast.parse(path.read_text(encoding='utf-8'));return ast.dump(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('image_tensor','visual_inputs','embedding_features')],type_ignores=[]),include_attributes=False)
assert pixel_ast(ROOT/'carvision/semantic_make.py')==pixel_ast(ROOT/'outputs/CarVision-Update-2.5/payload/carvision/semantic_make.py'),'Do not silently reuse features after a preprocessing change'
path=ROOT/'cache/ev_views_v26_features.pt';old=ROOT/'cache/semantic_v25/features.pt';encoder_source=torch.load(ROOT/'cache/semantic_v25/image_encoder.pt',weights_only=True);cache=torch.load(path if path.exists() else old,weights_only=True);assert cache['source_sha256']==encoder_source['source_sha256'];cache['pixel_equivalence_to_sealed_v25_verified']=True;cache['implementation_sha256']=hashlib.sha256((ROOT/'carvision/semantic_make.py').read_bytes()).hexdigest()
data=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True);rows=[r for r in data['rows'] if args.include_test or r['split'] in ('train','validation')];jobs=[(r,k) for r in rows for k in ('whole','phone') if token(r,k) not in cache['items']];network=GeneralVisionEncoder().eval();network.load_state_dict(encoder_source['state']);start=time.time()
with torch.inference_mode():
    for first in range(0,len(jobs),4):
        batch=jobs[first:first+4];x=torch.cat([visual_inputs(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy(),k) for r,k in batch]);z=network(x).reshape(-1,2,512)
        for j,(r,k) in enumerate(batch):cache['items'][token(r,k)]=z[j].clone()
        if first%40==0:torch.save(cache,path);print('FEATURES',first+len(batch),len(jobs),round(time.time()-start,1),flush=True)
torch.save(cache,path);(out/'features.json').write_text(json.dumps(dict(complete=True,test_requested=args.include_test,rows=len(rows),items=len(cache['items']),new_jobs=len(jobs),pixel_equivalence_to_sealed_v25_verified=True,dataset_sha256=hashlib.sha256((ROOT/'cache/ev_views_v26_data.pt').read_bytes()).hexdigest()),indent=2),encoding='utf-8');print('FEATURES COMPLETE',flush=True)
