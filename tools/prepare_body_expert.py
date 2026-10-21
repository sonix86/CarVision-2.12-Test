"""Restore source crop geometry without altering existing labels or splits."""
import sys,json,hashlib,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
import cv2
from PIL import Image
from carvision.core import ROOT,read_image
from carvision.segmentation import InstanceSegmenter,tight_box

def recover_box(work,target):
    """Register identical source pixels, then verify the integer crop directly.

    This is image registration, not semantic inference or a new model label.
    A failed pixel check falls back to the original generic mask preparation.
    """
    source=cv2.cvtColor(np.asarray(work.resize((176,176))),cv2.COLOR_RGB2GRAY).astype('float32')/255
    template=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY).astype('float32')/255
    try:
        score,warp=cv2.findTransformECC(template,source,np.eye(2,3,dtype='float32'),cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),None,3)
    except cv2.error:return None
    if score<.94 or abs(warp[0,1])>.02 or abs(warp[1,0])>.02:return None
    box=[round(warp[0,2]*work.width/176),round(warp[1,2]*work.height/176),round((warp[0,0]*176+warp[0,2])*work.width/176),round((warp[1,1]*176+warp[1,2])*work.height/176)]
    def error(b):
        if not 0<=b[0]<b[2]<=work.width or not 0<=b[1]<b[3]<=work.height:return float('inf')
        return float(np.abs(np.asarray(work.crop(b).resize((176,176),Image.Resampling.BICUBIC)).astype('float32')-target).mean())
    best=error(box)
    for _ in range(2):
        for axis in range(4):
            initial=box.copy()
            for delta in [-2,-1,0,1,2]:
                b=initial.copy();b[axis]+=delta;e=error(b)
                if e<best:best=e;box=b
    return (best,box) if best<=1. else None

def main():
    data=torch.load(ROOT/'cache/ev_parts_v19.pt',weights_only=True)
    regional={r['image']:r for r in map(json.loads,(ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines())}
    stanford={r['image']:r for r in json.loads((ROOT/'models/expanded_dataset.json').read_text(encoding='utf-8'))['new_rows']}
    ev={r['image']:r for r in json.loads((ROOT/'models/ev_dataset.json').read_text(encoding='utf-8'))['new_rows']}
    torch.set_num_threads(3)
    cv2.setNumThreads(1);segmenter=None
    counts=collections.Counter();rows=[];alignment=[]
    for number,row in enumerate(data['rows']):
        r=dict(row);key=r['image']
        if key in regional:
            im=read_image(ROOT/'data/regional'/key);counts['original_regional_crop']+=1
        elif key.startswith('stanford/'):
            meta=stanford[key[len('stanford/'):]];im=read_image(meta['path']).crop(meta['box']);counts['original_mat_box']+=1
        else:
            # The reviewed EV crop was cached square. Its recorded physical
            # aspect restores geometry; do not invent unavailable fine detail.
            ratio=ev[key]['ratio'];w=round(224*min(ratio,1));h=round(224*min(1/ratio,1))
            im=Image.fromarray(r['rgb'].numpy()).resize((max(1,w),max(1,h)),Image.Resampling.BICUBIC);counts['ev_restored_ratio']+=1
        if not key.startswith('ev_v19/'):
            # Original caches resized the mask's tight box, not the entire
            # source annotation box. Reproduce that operation and verify the
            # cached RGB pixels before transferring its coordinates to source.
            work=im.copy();work.thumbnail((384,384))
            match=recover_box(work,r['rgb'].numpy());candidates=[match] if match else []
            if match:counts['pixel_registration_match']+=1
            else:
                if segmenter is None:segmenter=InstanceSegmenter(torch.load(ROOT/'models/segmenter.pt',weights_only=True),small=True)
                with torch.inference_mode():d=segmenter.raw(work)
                ids=[j for j in range(len(d['scores'])) if int(d['labels'][j]) in (3,8)]
                for j in ids:
                    mask=d['masks'][j,0].numpy()>.5
                    box=tight_box(mask)
                    if box is None:continue
                    a=np.asarray(work.crop(box).resize((176,176),Image.Resampling.BICUBIC)).astype('float32')
                    error=float(np.abs(a-r['rgb'].numpy()).mean());candidates.append((error,box))
            # Earlier preparation explicitly used the full image when the
            # generic model had no usable mask. Preserve this possible match.
            full=[0,0,work.width,work.height];a=np.asarray(work.resize((176,176),Image.Resampling.BICUBIC)).astype('float32');candidates.append((float(np.abs(a-r['rgb'].numpy()).mean()),full))
            error,box=min(candidates,key=lambda item:item[0])
            if error<=1.:
                scale_x=im.width/work.width;scale_y=im.height/work.height
                im=im.crop([round(box[0]*scale_x),round(box[1]*scale_y),round(box[2]*scale_x),round(box[3]*scale_y)])
                counts['verified_tight_source_box']+=1;r['geometry_source']='Source box reproduced against cached RGB'
            else:
                # Never borrow a different vehicle's source pixels, remove a
                # difficult test item or invent an aspect ratio to improve scores.
                im=Image.fromarray(r['rgb'].numpy());counts['cached_square_fallback']+=1;r['geometry_source']='Unrecoverable source box; original cached square retained'
            alignment.append(dict(image=key,split=r['split'],rgb_mean_error=error,matched=error<=1.))
        im.thumbnail((320,320));r['body_rgb']=torch.from_numpy(np.asarray(im).copy())
        r['body_mask']=torch.from_numpy(np.asarray(Image.fromarray(r['mask'].numpy()).resize(im.size,Image.Resampling.NEAREST)).copy())
        rows.append(r)
        if number%100==0:print('ALIGNED',number,'/',len(data['rows']),dict(counts),flush=True)
    out=dict(rows=rows,pairs=data['pairs'],source_cache_sha256=hashlib.sha256((ROOT/'cache/ev_parts_v19.pt').read_bytes()).hexdigest())
    torch.save(out,ROOT/'cache/body_expert_v20.pt')
    (ROOT/'models/body_expert_candidate').mkdir(exist_ok=True)
    info=dict(geometry_sources=dict(counts),alignment=alignment,counts={b+' '+m:{s:sum(r['y']==i and r['split']==s for r in rows) for s in ['train','validation','test']} for i,(b,m) in enumerate(data['pairs'])},labels_and_splits_unchanged=True,city_examples_used=False)
    (ROOT/'models/body_expert_candidate/dataset.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(info['geometry_sources']),len(rows),'rows; splits unchanged',flush=True)

if __name__=='__main__':main()
