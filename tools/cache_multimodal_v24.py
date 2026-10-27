"""Cache generic ImageNet features, never ready-made vehicle classifier outputs."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageEnhance,ImageFilter
from torchvision import models
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.body_expert import body_tensor
from carvision.segmentation import isolate

def token(row,kind):return hashlib.sha256(row['body_rgb'].numpy().tobytes()+row['body_mask'].numpy().tobytes()+('convnext-v24-'+kind).encode()).hexdigest()

def inputs(row,kind):
    im=Image.fromarray(row['body_rgb'].numpy());mask=row['body_mask'].numpy()
    if kind=='phone':
        im.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));im=ImageEnhance.Brightness(im).enhance(.55).filter(ImageFilter.GaussianBlur(.35))
    foreground=isolate(im,mask);w,h=foreground.size
    detail=foreground.crop((round(w*.12),round(h*.25),max(round(w*.12)+1,round(w*.88)),max(round(h*.25)+1,round(h*.85))))
    return torch.stack([body_tensor(foreground),body_tensor(detail)])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-test',action='store_true');args=parser.parse_args()
    torch.set_num_threads(3);torch.manual_seed(240026)
    out=ROOT/'models/multimodal_v24';out.mkdir(exist_ok=True);weights=ROOT/'cache/imagenet/convnext_tiny-983f1562.pth'
    source_sha=hashlib.sha256(weights.read_bytes()).hexdigest();assert source_sha.startswith('983f1562')
    dataset=ROOT/'cache/multimodal_v24_data.pt'
    if not dataset.exists():dataset=ROOT/'cache/make_scope_v23.pt'
    data=torch.load(dataset,weights_only=True)
    rows=[r for r in data['rows'] if args.include_test or r['split'] in ('train','validation')]
    network=models.convnext_tiny(weights=None);network.load_state_dict(torch.load(weights,weights_only=True));network.classifier[2]=torch.nn.Identity();network.eval()
    path=ROOT/'cache/multimodal_v24_features.pt'
    cache=torch.load(path,weights_only=True) if path.exists() else dict(source_sha256=source_sha,items={})
    assert cache['source_sha256']==source_sha
    jobs=[(r,k) for r in rows for k in ('whole','phone') if token(r,k) not in cache['items']]
    started=time.time()
    with torch.inference_mode():
        for first in range(0,len(jobs),8):
            batch=jobs[first:first+8];z=network(torch.cat([inputs(r,k) for r,k in batch])).reshape(-1,2,768)
            z=torch.nn.functional.normalize(z,dim=2).reshape(-1,1536)
            for i,(r,k) in enumerate(batch):cache['items'][token(r,k)]=z[i].clone()
            if first%200==0:
                torch.save(cache,path);status=dict(completed=first+len(batch),total=len(jobs),seconds=time.time()-started)
                (out/'feature_progress.json').write_text(json.dumps(status),encoding='utf-8');print(json.dumps(status),flush=True)
    torch.save(cache,path)
    meta=dict(complete=True,source_sha256=source_sha,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),feature_items=len(cache['items']),test_features_requested=args.include_test,ready_made_car_classifier=False)
    (out/'features.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('FEATURES COMPLETE',flush=True)

if __name__=='__main__':main()
