"""Source-labelled paint corpus; preserve existing splits and private train-only rows."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import read_image
from carvision.segmentation import InstanceSegmenter,tight_box,isolate
from carvision.illumination_colour import features,scene_gains
from carvision.colour_model import ColourClassifier
from carvision.masked_colour import MaskedColourClassifier

torch.set_num_threads(2);torch.manual_seed(240026)
out=ROOT/'models/multimodal_v24';path=ROOT/'cache/illumination_v24.pt';cache=torch.load(path,weights_only=True) if path.exists() else dict(items={})
source_rows=torch.load(ROOT/'cache/masked_colour_v15.pt',weights_only=True)['rows']
segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True)
old=ColourClassifier(torch.load(ROOT/'models/colour_model.pt',weights_only=True));paint=MaskedColourClassifier(torch.load(ROOT/'models/masked_colour.pt',weights_only=True))
started=time.time()
def append(key,image,mask,colour,split,group,owner=False):
    rng=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8],16));variants={}
    for kind in ('whole','warm','shadow'):
        a=np.asarray(image,dtype=np.float32)/255
        if kind=='warm':a=np.clip(a*np.array([1.17,1.02,.84],dtype=np.float32),0,1)
        if kind=='shadow':a=np.clip((a**1.2)*.58,0,1)
        im=Image.fromarray((a*255).astype(np.uint8));gains=scene_gains(im,mask)
        box=tight_box(mask)
        crop=im.crop(box) if box else im;local=mask[box[1]:box[3],box[0]:box[2]] if box else mask
        # Training uses only our paint features. Old-network predictions are
        # required for validation/test comparisons, never for fitting this head.
        base=torch.zeros(8)
        if split!='train':
            with torch.inference_mode():base=old.probabilities(isolate(crop,local));base=paint.blend(base,paint.probabilities(crop,local))
        variants[kind]=dict(x=features(im,mask,gains),baseline=base)
    cache['items'][key]=dict(colour=colour,split=split,group=group,owner=owner,variants=variants)

for i,r in enumerate(source_rows):
    key='vcor/'+r['image']
    if key not in cache['items']:
        im=read_image(ROOT/'datasets/vcor/images'/r['image']);im.thumbnail((384,384))
        with torch.inference_mode():d=segmenter.raw(im)
        ids=[j for j,s in enumerate(d['scores']) if float(s)>.3 and int(d['labels'][j]) in (3,8)]
        mask=d['masks'][max(ids,key=lambda j:int((d['masks'][j,0]>.5).sum())),0].numpy()>.5 if ids else np.ones((im.height,im.width),dtype=bool)
        append(key,im,mask,r['colour'],r['split'],key)
    if i%50==0:
        torch.save(cache,path);status=dict(completed=i,total=len(source_rows),seconds=time.time()-started);(out/'colour_progress.json').write_text(json.dumps(status),encoding='utf-8');print(json.dumps(status),flush=True)
data=torch.load(ROOT/'cache/multimodal_v24_data.pt',weights_only=True)
for r in data['rows']:
    if not r.get('owner'):continue
    key=r['image']
    if key not in cache['items']:append(key,Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy(),r['colour'],'train',r['group'],True)
torch.save(cache,path);print('PAINT FEATURES COMPLETE',len(cache['items']),flush=True)
