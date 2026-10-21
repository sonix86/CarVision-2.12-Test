"""Visible-instance masks, disjoint colour evidence and compact portable geometry."""
import hashlib
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

def encode_mask(mask,origin=(0,0)):
    a=np.asarray(mask,dtype=np.uint8);flat=a.ravel()
    # Alternating zero/one run lengths, always starting with a zero run.
    boundaries=np.flatnonzero(flat[1:]!=flat[:-1])+1 if len(flat) else np.array([],dtype=int)
    counts=np.diff(np.r_[0,boundaries,len(flat)]).tolist()
    if len(flat) and flat[0]:counts.insert(0,0)
    return dict(encoding='rle-row-major-v1',size=list(a.shape),origin=list(origin),counts=counts)

def decode_mask(data):
    h,w=data['size'];counts=np.asarray(data['counts'],dtype=np.int64)
    if h<0 or w<0 or h*w>16000000 or np.any(counts<0) or int(counts.sum())!=h*w:raise ValueError('Invalid mask encoding')
    return np.repeat(np.arange(len(counts))%2,counts).reshape(h,w).astype(bool)

def contains_point(data,x,y):
    x=(x-data['origin'][0])/data.get('scale',[1,1])[0];y=(y-data['origin'][1])/data.get('scale',[1,1])[1];h,w=data['size']
    if not (0<=x<w and 0<=y<h):return False
    return bool(decode_mask(data)[int(y),int(x)])

def polygons(mask,origin=(0,0)):
    import cv2
    contours,_=cv2.findContours(np.asarray(mask,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);out=[]
    for contour in contours:
        if cv2.contourArea(contour)<12:continue
        contour=cv2.approxPolyDP(contour,max(.7,cv2.arcLength(contour,True)*.0015),True)
        if len(contour)>=3:out.append([[int(p[0])+origin[0],int(p[1])+origin[1]] for p in contour[:,0]])
    return out

def isolate(image,mask):
    a=np.asarray(image.convert('RGB'));mask=np.asarray(mask,dtype=bool)
    if mask.shape!=a.shape[:2]:mask=np.asarray(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST))
    pixels=a[mask]
    if len(pixels)<8:return image.copy()
    background=np.median(pixels,axis=0).astype('uint8');out=a.copy();out[~mask]=background
    return Image.fromarray(out)

def tight_box(mask):
    y,x=np.nonzero(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None

def resolve_instances(records):
    """Consolidate alternative proposals, retaining disconnected visible parts.

    A bridge is evidence from a SINGLE detector instance, never a proximity rule.
    Two neighbouring masks cannot merge merely because their boxes intersect.
    """
    import cv2
    records=[dict(r,mask=r['mask'].copy(),probability=r['probability'].copy(),proposals=1) for r in records]
    for bridge in sorted(records,key=lambda r:r['score'],reverse=True):
        if not any(bridge is r for r in records) or bridge['score']<.7:continue
        area=int(bridge['mask'].sum())
        _,_,stats,_=cv2.connectedComponentsWithStats(bridge['mask'].astype('uint8'))
        if sum(int(a)>=max(60,area*.1) for a in stats[1:,4])<2:continue
        matches=[]
        for other in records:
            if other is bridge:continue
            overlap=int(np.count_nonzero(bridge['mask']&other['mask']))
            fraction=overlap/max(1,int(other['mask'].sum()))
            if fraction>=.5:matches.append((other,fraction))
        # Only combine complementary proposals supported by this one instance.
        # Independent strongly overlapping masks are handled by deduplication.
        for a,fa in matches:
            for b,fb in matches:
                if a is b or max(fa,fb)<.8:continue
                common=np.count_nonzero(a['mask']&b['mask'])/max(1,min(int(a['mask'].sum()),int(b['mask'].sum())))
                if common>.15:continue
                union=bridge['mask']|a['mask']|b['mask']
                if area/int(union.sum())<.5:continue
                for part in [a,b]:
                    if any(part is r for r in records):
                        bridge['mask']|=part['mask'];bridge['probability']=np.maximum(bridge['probability'],part['probability']);bridge['proposals']+=part['proposals'];records=[r for r in records if r is not part]
                bridge['score']=max(bridge['score'],a['score'],b['score'])
                break
            else:continue
            break
    kept=[]
    for r in sorted(records,key=lambda r:r['score']*int(r['mask'].sum())**.5,reverse=True):
        for old in kept:
            overlap=int(np.count_nonzero(r['mask']&old['mask']));small=min(int(r['mask'].sum()),int(old['mask'].sum()));union=int(np.count_nonzero(r['mask']|old['mask']))
            if overlap/max(1,union)>.55 or overlap/max(1,small)>.92:
                old['mask']|=r['mask'];old['probability']=np.maximum(old['probability'],r['probability']);old['proposals']+=r['proposals'];break
        else:kept.append(r)
    return kept

class InstanceSegmenter:
    def __init__(self,checkpoint,small=False):
        if checkpoint.get('origin')!='generic-coco-instance-segmenter':raise ValueError('Expected generic COCO masks, not vehicle-brand weights')
        self.network=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=256 if small else 800,max_size=384 if small else 1333,box_score_thresh=.3,box_detections_per_img=3 if small else 80,rpn_pre_nms_top_n_test=200 if small else 1000,rpn_post_nms_top_n_test=50 if small else 300).eval()
        self.network.load_state_dict(checkpoint['state'])
    @torch.inference_mode()
    def raw(self,image):return self.network([to_tensor(image)])[0]
    @torch.inference_mode()
    def instances(self,image,threshold=.5,mode='fast'):
        from .core import detection_tiles
        # Reuse only the last actual pixel image and identical search settings.
        # Manual selection can then retain full-image occlusion context without
        # running the same expensive detector again. Never key by file name.
        key=(image.size,hashlib.sha256(image.tobytes()).digest(),float(threshold),mode)
        if key==getattr(self,'_cache_key',None):
            return [dict(r,mask=r['mask'].copy()) for r in self._cache_rows],self._cache_views
        tiles=[(0,0,*image.size)] if mode=='fast' else detection_tiles(*image.size);views=[(tile,False) for tile in tiles];records=[];people=np.zeros((image.height,image.width),dtype=np.float32)
        if mode=='detail':views.append(((0,0,*image.size),True))
        for tile,mirrored in views:
            crop=image.crop(tile);d=self.raw(ImageOps.mirror(crop) if mirrored else crop)
            for i in range(len(d['scores'])):
                if int(d['labels'][i])==1 and float(d['scores'][i])>=.85:
                    local=d['masks'][i,0].numpy()
                    if mirrored:local=local[:,::-1]
                    patch=people[tile[1]:tile[3],tile[0]:tile[2]];np.maximum(patch,local*float(d['scores'][i]),out=patch)
            for i in range(len(d['scores'])):
                score=float(d['scores'][i]);label=int(d['labels'][i]);box=d['boxes'][i].tolist()
                if label not in (3,8) or score<threshold:continue
                if label==8 and (box[2]-box[0])/max(1,box[3]-box[1])>3.5:continue
                local=d['masks'][i,0].numpy()
                if mirrored:local=local[:,::-1]
                m=np.zeros((image.height,image.width),dtype=np.float32);m[tile[1]:tile[3],tile[0]:tile[2]]=local
                binary=m>=.5
                if int(binary.sum())<60:continue
                records.append(dict(mask=binary,probability=m,score=score,label=label))
        for r in records:
            r['mask'] &= ~((people>.65)&(people>r['probability']*r['score']))
        records=[r for r in records if int(r['mask'].sum())>=60]
        kept=resolve_instances(records)
        if kept:
            # Each visible pixel belongs to only one instance. This prevents colour
            # from being counted twice where uncertain boundary masks intersect.
            # A soft prediction outside a neighbour's accepted shape must not
            # steal a pixel from a valid, lower-confidence vehicle mask.
            evidence=np.stack([np.where(r['mask'],r['probability']*r['score'],-1.) for r in kept]);owner=evidence.argmax(0)
            for i,r in enumerate(kept):r['mask'] &= owner==i
        result=[dict(mask=r['mask'],score=r['score'],label=r['label'],merged_proposals=r['proposals']) for r in kept if int(r['mask'].sum())>=60]
        self._cache_key=key;self._cache_rows=[dict(r,mask=r['mask'].copy()) for r in result];self._cache_views=len(views)
        return result,len(views)
