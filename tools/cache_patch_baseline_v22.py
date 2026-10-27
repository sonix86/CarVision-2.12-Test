"""Cache validation-only baseline inference while data review completes."""
import sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image,ImageEnhance,ImageFilter
from carvision.core import ROOT
from carvision.vehicle_model import VehicleModelClassifier
from carvision.visual_parts import VisualReferences
from carvision.segmentation import isolate

def key(row,kind):
    return hashlib.sha256(row['body_rgb'].numpy().tobytes()+row['body_mask'].numpy().tobytes()+str(tuple(row['body_rgb'].shape)).encode()+kind.encode()).hexdigest()

if __name__=='__main__':
    torch.set_num_threads(3)
    sha=hashlib.sha256((ROOT/'models/vehicle_model.pt').read_bytes()).hexdigest()
    path=ROOT/'cache/ev22_baseline_validation.pt';store=torch.load(path,weights_only=True) if path.exists() else dict(source_sha=sha,values={})
    assert store['source_sha']==sha
    net=VehicleModelClassifier(torch.load(ROOT/'models/vehicle_model.pt',weights_only=True));net.references=VisualReferences(torch.load(ROOT/'models/visual_references.pt',weights_only=True))
    rows=[r for r in torch.load(ROOT/'cache/ev_patch_v22.pt',weights_only=True)['rows'] if r['split']=='validation']
    with torch.inference_mode():
        for kind in ['whole','phone']:
            for i,r in enumerate(rows):
                k=key(r,kind)
                if k in store['values']:continue
                im=Image.fromarray(r['body_rgb'].numpy());mask=r['body_mask'].numpy()
                if kind=='phone':
                    im.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST));im=ImageEnhance.Brightness(im).enhance(.48).filter(ImageFilter.GaussianBlur(.45))
                store['values'][k]=net.probabilities(im,None,None)*.75+net.probabilities(isolate(im,mask),None,None)*.25
                if i%50==0:print(kind,i,len(rows),flush=True)
    torch.save(store,path);print('COMPLETE',len(store['values']),flush=True)
