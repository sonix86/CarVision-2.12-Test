"""Instance-aware scene analysis. Paint never includes pixels from a neighbour."""
import time
import copy
import numpy as np
import torch
from PIL import Image,ImageOps
from .segmentation import tight_box,encode_mask,polygons,isolate

def manual_foreground(image):
    import cv2
    a=np.asarray(image);labels=np.zeros(a.shape[:2],dtype=np.uint8)
    try:
        cv2.grabCut(a,labels,(2,2,max(1,image.width-4),max(1,image.height-4)),None,None,3,cv2.GC_INIT_WITH_RECT)
        mask=(labels==1)|(labels==3)
        if int(mask.sum())>=20:return mask
    except cv2.error:pass
    return np.ones(a.shape[:2],dtype=bool)

@torch.inference_mode()
def classify(p,crop,mask):
    from .core import COLOURS,heuristic_colour
    focused=isolate(crop,mask);brand='unknown';bp=0.;brand_alternatives=[];brand_source='own make classifier'
    if p.classifier is not None:
        def brand_prob(im):
            views=[im,ImageOps.mirror(im)];logits=p.classifier(torch.stack([p.preprocess(v) for v in views]))[0]
            return (logits/p.brand_temperature).softmax(1).mean(0)
        weight=p.mask_fusion['brand_mask_weight'];prob=brand_prob(crop)*(1-weight)+brand_prob(focused)*weight;bp,index=prob.max(0);bp=float(bp);brand=p.brands[int(index)] if bp>=p.brand_threshold else 'unknown'
        brand_alternatives=[dict(brand=p.brands[int(i)],confidence=float(prob[i])) for i in prob.topk(min(3,len(p.brands))).indices]
    model=dict(model='unknown',model_candidate='unknown',model_confidence=0.,model_alternatives=[])
    if p.vehicle_classifier is not None:
        weight=p.mask_fusion['model_mask_weight'];prob=p.vehicle_classifier.probabilities(crop,p.classifier,p.preprocess)*(1-weight)+p.vehicle_classifier.probabilities(focused,p.classifier,p.preprocess)*weight;model=p.vehicle_classifier.describe(prob,brand)
        # The older make head has a closed vocabulary. A newly trained model
        # head can identify an added manufacturer; never force it into one of
        # the old18 brands. Require the model head's calibrated acceptance gate.
        new_brand=model.get('model_candidate_brand')
        reference=p.vehicle_classifier.references
        gate=reference.threshold if reference is not None else p.vehicle_classifier.threshold
        if new_brand and new_brand not in p.brands and model.get('model_confidence',0)>=max(.85,gate):
            brand=new_brand;bp=sum(r['confidence'] for r in model.get('model_ranking',[]) if r['brand']==new_brand);model=p.vehicle_classifier.describe(prob,brand);brand_source='own model head taxonomy; manufacturer absent from older make head'
    colour,cp=heuristic_colour(focused);source='foreground HSV fallback';threshold=p.colour_reliability_threshold
    if p.colour_classifier is not None:
        prob=p.colour_classifier.probabilities(focused);source='own colour head on isolated visible instance'
        if p.masked_colour is not None:
            prob=p.masked_colour.blend(prob,p.masked_colour.probabilities(crop,mask));threshold=p.masked_colour.threshold;source='own source-labelled visible-paint head + isolated-instance colour evidence'
        cp,index=prob.max(0);cp=float(cp);colour=COLOURS[int(index)];alternatives=[dict(colour=COLOURS[int(i)],confidence=float(prob[i])) for i in prob.topk(2).indices]
    else:alternatives=[dict(colour=colour,confidence=cp)]
    return dict(brand=brand,brand_confidence=bp,brand_alternatives=brand_alternatives,brand_source=brand_source,**model,colour=colour,colour_confidence=cp,colour_reliable=cp>=threshold,colour_alternatives=alternatives,colour_source=source)

@torch.inference_mode()
def predict_instances(p,original,path,region,single,started):
    image=original if single else original.crop(region) if region else original;offset=[0,0] if single else region[:2] if region else [0,0];work=image.copy();work.thumbnail((1600,1600));records,views=p.segmenter.instances(work,p.detection,p.mode);warnings=[]
    if single:
        # Preserve the full-image occlusion context. Segmenting only the prompt
        # rectangle can otherwise merge a background car with its foreground neighbour.
        prompt=[region[0]*work.width/original.width,region[1]*work.height/original.height,region[2]*work.width/original.width,region[3]*work.height/original.height];matches=[]
        for record in records:
            box=tight_box(record['mask']);x=max(box[0],prompt[0]);y=max(box[1],prompt[1]);r=min(box[2],prompt[2]);b=min(box[3],prompt[3]);intersection=max(0,r-x)*max(0,b-y);union=(box[2]-box[0])*(box[3]-box[1])+(prompt[2]-prompt[0])*(prompt[3]-prompt[1])-intersection;iou=intersection/max(1,union)
            if iou<.03:continue
            mask=record['mask'].copy();x,y,r,b=[int(v) for v in [np.floor(prompt[0]),np.floor(prompt[1]),np.ceil(prompt[2]),np.ceil(prompt[3])]];mask[:y]=False;mask[b:]=False;mask[:,:x]=False;mask[:,r:]=False
            visible=int(mask.sum());prompt_area=max(1,(prompt[2]-prompt[0])*(prompt[3]-prompt[1]));visible_iou=visible/max(1,int(record['mask'].sum())+prompt_area-visible)
            if visible>=20:matches.append((visible_iou*record['score'],dict(record,mask=mask)))
        if matches:records=[max(matches,key=lambda r:r[0])[1]]
        else:
            image=original.crop(region);offset=region[:2];work=image.copy();work.thumbnail((1600,1600));records,extra=p.segmenter.instances(work,p.detection,p.mode);views+=extra
            if records:records=[max(records,key=lambda r:(lambda b:(b[2]-b[0])*(b[3]-b[1]))(tight_box(r['mask']))*r['score'])]
            else:
                mask=manual_foreground(work);records=[dict(mask=mask,score=None,label=3,approximate=True)];warnings.append('No COCO car mask in selected region; approximate GrabCut foreground used. The selected object may not be a car.')
    sx=image.width/work.width;sy=image.height/work.height;result=[]
    for record in records:
        mask=record['mask'];box=tight_box(mask)
        if box is None:continue
        x,y,r,b=box;global_box=[round(x*sx+offset[0]),round(y*sy+offset[1]),round(r*sx+offset[0]),round(b*sy+offset[1])]
        local=mask[y:b,x:r];crop=original.crop(global_box);crop.thumbnail((640,640));cm=np.asarray(Image.fromarray(local).resize(crop.size,Image.Resampling.NEAREST));row=classify(p,crop,cm)
        from .visual_parts import quality_of_fragment
        reasons=quality_of_fragment(local)
        if global_box[0]<=1 or global_box[1]<=1 or global_box[2]>=original.width-1 or global_box[3]>=original.height-1:
            reasons.append('Машина обрезана границей фотографии: показаны варианты без подтверждения модели.')
        row['visual_limitations']=reasons;row['model_limited']=bool(reasons)
        if reasons:
            row['brand_candidate']=row['brand'];row['brand']='unknown';row['model']='unknown'
        elif row.get('reference_comparison') and not row.get('model_reason') and row['brand']!='unknown' and row.get('model_candidate_brand')!=row['brand'] and row.get('model_confidence',0)>=.5:
            row['brand_candidate']=row['brand'];row['brand']='unknown';row['model']='unknown';row['visual_limitations']=['Классификатор марки и сравнение деталей расходятся: название не подтверждено.']
        encoded=encode_mask(local,global_box[:2]);encoded['scale']=[(global_box[2]-global_box[0])/local.shape[1],(global_box[3]-global_box[1])/local.shape[0]]
        contours=[[[round((px+x)*sx+offset[0]),round((py+y)*sy+offset[1])] for px,py in contour] for contour in polygons(local)]
        result.append(dict(row,box=global_box,segmentation=encoded,contours=contours,mask_pixels=int(local.sum()),mask_box_fraction=float(local.mean()),merged_proposals=record.get('merged_proposals',1),visible_parts=len(contours),detection_confidence=record['score'],box_source='approximate GrabCut selected foreground' if record.get('approximate') else 'user region + visible-instance mask' if single else 'COCO visible-instance mask',approximate_mask=bool(record.get('approximate'))))
    result.sort(key=lambda r:(r['box'][1],r['box'][0]))
    for i,row in enumerate(result,1):row['id']=i
    warnings.append('Contours describe visible pixels, not reconstructed hidden vehicle parts. Unrepresented models and strong occlusion can still produce wrong candidates.')
    supported=sorted(set(p.brands)|set(getattr(getattr(p,'vehicle_classifier',None),'brands',[])))
    return dict(image=path,cars=result,seconds=round(time.perf_counter()-started,3),classifier_trained=p.classifier is not None,supported_brands=supported,mode=p.mode,region=region,single_car_region=single,detector_views=views,segmented=True,warnings=warnings)

def merge_prediction(p,original,prediction,ids):
    """Explicit user correction: union existing visible pixels, then reclassify.

    Never fill the space between parts. No make/model answer is inherited from
    the old guesses. Unselected cars and their masks remain untouched.
    """
    from PIL import ImageChops
    from .segmentation import decode_mask
    ids=set(ids);selected=[r for r in prediction['cars'] if r['id'] in ids]
    if len(selected)!=len(ids) or len(ids)<2 or any(not r.get('segmentation') for r in selected):raise ValueError('Выберите минимум два контура одной машины.')
    started=time.perf_counter();work=original.copy();work.thumbnail((1600,1600));sx=work.width/original.width;sy=work.height/original.height;union=Image.new('L',work.size)
    for row in selected:
        encoded=row['segmentation'];mask=Image.fromarray(decode_mask(encoded).astype('uint8')*255);scale=encoded.get('scale',[1,1]);mask=mask.resize((max(1,round(mask.width*scale[0]*sx)),max(1,round(mask.height*scale[1]*sy))),Image.Resampling.NEAREST)
        layer=Image.new('L',work.size);layer.paste(mask,(round(encoded['origin'][0]*sx),round(encoded['origin'][1]*sy)));union=ImageChops.lighter(union,layer)
    class FixedMask:
        def instances(self,*args):return [dict(mask=np.asarray(union)>0,score=min(r['detection_confidence'] or 0 for r in selected),label=3,merged_proposals=sum(r.get('merged_proposals',1) for r in selected))],0
    proxy=copy.copy(p);proxy.segmenter=FixedMask()
    combined=predict_instances(proxy,original,prediction['image'],None,False,started)['cars'][0]
    combined['box_source']='user-confirmed union of existing visible parts';combined['manually_merged_from']=sorted(ids)
    result=copy.deepcopy(prediction);result['cars']=[r for r in result['cars'] if r['id'] not in ids]+[combined];result['cars'].sort(key=lambda r:(r['box'][1],r['box'][0]))
    for i,r in enumerate(result['cars'],1):r['id']=i
    result['seconds']=round(time.perf_counter()-started,3);result.setdefault('warnings',[]).append('User merged selected visible parts into one instance; hidden areas were not reconstructed. IDs renumbered.')
    return result
