import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large,fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms

ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
COLOURS = ['white', 'black', 'grey', 'red', 'blue', 'green', 'brown', 'yellow']
BRANDS = ['Chevrolet', 'Daewoo', 'Lada', 'Kia', 'Hyundai', 'Toyota', 'Mercedes-Benz', 'BMW', 'Volkswagen', 'Nissan', 'BYD', 'Chery', 'Haval', 'Changan', 'Geely', 'Jetour', 'Li Auto', 'Zeekr', 'Exeed', 'Omoda', 'Jaecoo', 'Hongqi', 'Leapmotor', 'JAC']
def make_preprocess(size=224,center_margin=0):
    steps=[transforms.Resize((size+center_margin,size+center_margin))]
    if center_margin:steps.append(transforms.CenterCrop(size))
    return transforms.Compose(steps+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
PREPROCESS = make_preprocess()

def read_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')

class Classifier(nn.Module):
    """MobileNetV3 Small + concatenated average/max pooling + two heads."""
    def __init__(self, classes):
        super().__init__()
        m = models.mobilenet_v3_small(weights=None)
        self.features = m.features
        self.neck = nn.Sequential(nn.Linear(1152, 256), nn.Hardswish(), nn.Dropout(.3))
        self.brand = nn.Linear(256, classes)
        self.colour = nn.Linear(256, len(COLOURS))

    def forward(self, x):
        f = self.features(x)
        f = self.neck(torch.cat([f.mean((2,3)), f.amax((2,3))], 1))
        return self.brand(f), self.colour(f)

class ResidualBlock(nn.Module):
    def __init__(self,inputs,outputs,stride=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(inputs,outputs,3,stride,1,bias=False),nn.BatchNorm2d(outputs),nn.ReLU(),nn.Conv2d(outputs,outputs,3,1,1,bias=False),nn.BatchNorm2d(outputs))
        self.skip=nn.Identity() if inputs==outputs and stride==1 else nn.Sequential(nn.Conv2d(inputs,outputs,1,stride,bias=False),nn.BatchNorm2d(outputs))
    def forward(self,x):return torch.relu(self.body(x)+self.skip(x))

class CompactResidualClassifier(nn.Module):
    """Small original residual brand network, initialized entirely at random."""
    def __init__(self,classes):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,7,2,3,bias=False),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(3,2,1),ResidualBlock(32,32),ResidualBlock(32,64,2),ResidualBlock(64,128,2),ResidualBlock(128,256,2))
        self.neck=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(.15));self.brand=nn.Linear(256,classes);self.colour=nn.Linear(256,len(COLOURS))
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
    def forward(self,x):
        f=self.features(x);f=self.neck(torch.cat([f.mean((2,3)),f.amax((2,3))],1));return self.brand(f),self.colour(f)

def heuristic_colour(crop):
    # Original non-learned method; thresholds chosen on a separate 23-crop dev review.
    # See models/colour_rules.json. No final-test reference colours were used to fit these.
    w,h = crop.size
    a = np.asarray(crop.crop((w*.15,h*.35,w*.85,h*.8)).resize((96,64)).convert('HSV')).astype(float)
    hue,s,v = a[:,:,0]*360/255, a[:,:,1]/255, a[:,:,2]/255
    chrom=(s>=.22)&(v>=.26)
    masks = [(s<.22)&(v>.72), v<.26, (s<.22)&(v>=.26)&(v<=.72), ((hue<18)|(hue>=345))&chrom, (hue>=185)&(hue<275)&chrom, (hue>=65)&(hue<185)&chrom, (hue>=18)&(hue<65)&chrom&(v<.68), (hue>=18)&(hue<65)&chrom&(v>=.68)]
    counts = np.array([m.sum() for m in masks])
    return COLOURS[int(counts.argmax())], float(counts.max()/max(1,counts.sum()))

def detection_tiles(width, height, size=768, overlap=.25):
    """Full view followed by overlapping crops; coordinates are in working-image pixels."""
    result=[(0,0,width,height)]
    if max(width,height)<=size:return result
    def starts(length):
        if length<=size:return [0]
        values=list(range(0,length-size+1,max(1,round(size*(1-overlap)))))
        if values[-1]!=length-size:values.append(length-size)
        return values
    for y in starts(height):
        for x in starts(width):
            rect=(x,y,min(x+size,width),min(y+size,height))
            if rect not in result:result.append(rect)
    return result

def restore_tile_boxes(detection, tile, width, height, threshold, car_label=1):
    """Map local boxes to full coordinates, excluding objects cut at internal seams."""
    x,y,right,bottom=tile
    label_mask=torch.isin(detection['labels'],torch.tensor(car_label,device=detection['labels'].device)) if isinstance(car_label,(list,tuple)) else detection['labels']==car_label
    mask=label_mask&(detection['scores']>=threshold)
    boxes=detection['boxes'][mask].clone();scores=detection['scores'][mask]
    if not len(boxes):return boxes,scores
    margin=3
    valid=torch.ones(len(boxes),dtype=torch.bool)
    if x>0:valid &= boxes[:,0]>margin
    if y>0:valid &= boxes[:,1]>margin
    if right<width:valid &= boxes[:,2]<right-x-margin
    if bottom<height:valid &= boxes[:,3]<bottom-y-margin
    boxes=boxes[valid];scores=scores[valid]
    boxes+=torch.tensor([x,y,x,y],dtype=boxes.dtype)
    return boxes,scores

def merge_vehicle_boxes(boxes,scores,full_count=0):
    """Prefer complete full-frame objects over nested tile fragments.

    Applies containment suppression only across full/tile views, not two cars
    found together. Ordinary overlapping cars remain eligible for standard NMS.
    """
    valid=torch.ones(len(boxes),dtype=torch.bool)
    if full_count:
        anchors=boxes[:full_count]
        anchor_area=(anchors[:,2]-anchors[:,0])*(anchors[:,3]-anchors[:,1])
        for i in range(full_count,len(boxes)):
            box=boxes[i];area=(box[2]-box[0])*(box[3]-box[1])
            intersection=(torch.minimum(anchors[:,2:],box[2:])-torch.maximum(anchors[:,:2],box[:2])).clamp_min(0).prod(1)
            nested=(intersection/area.clamp_min(1)>.92)&(area/anchor_area.clamp_min(1)<.75)&(scores[:full_count]>.85)
            if bool(nested.any()):valid[i]=False
    indices=torch.where(valid)[0];return indices[nms(boxes[indices],scores[indices],.45)]

class Pipeline:
    def __init__(self, directory=None, detection=.5, brand_threshold=.5, colour_mode='trained', mode='balanced'):
        torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
        self.directory = Path(directory or ROOT/'models')
        self.detection, self.brand_threshold, self.colour_mode = detection, brand_threshold, colour_mode
        if mode not in ('fast','balanced','detail'):raise ValueError('Unknown detection mode')
        self.mode=mode
        detector_file = self.directory/'detector.pt'
        if not detector_file.exists():
            raise FileNotFoundError('Детектор ещё не обучен. Добавьте свои фото, разметьте все машины и запустите обучение во вкладке «Обучение». Готовые веса не используются.')
        self.segmenter=None
        if (self.directory/'segmenter.pt').exists():
            from .segmentation import InstanceSegmenter
            self.segmenter=InstanceSegmenter(torch.load(self.directory/'segmenter.pt',map_location='cpu',weights_only=True));self.detector=None
        else:
            detector_checkpoint=torch.load(detector_file,map_location='cpu',weights_only=True)
            origin=detector_checkpoint.get('origin')
            if origin not in ('team-trained-from-scratch','generic-coco-detector','team-trained-detector-coco-backbone'):raise ValueError('Unsupported detector origin.')
            self.car_label=(3,8) if origin=='generic-coco-detector' else 1
            self.detector_batch_size=1 if detector_checkpoint.get('architecture')=='fasterrcnn_resnet50_fpn_v2_coco_v1' else 4
            if detector_checkpoint.get('architecture')=='hog_linear_multiscale_v1':
                from .hog_detector import HOGCarDetector
                self.detector=HOGCarDetector(detector_checkpoint)
            elif detector_checkpoint.get('architecture')=='fasterrcnn_resnet50_fpn_v2_coco_v1':
                self.detector=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=800,max_size=1333).eval()
                self.detector.load_state_dict(detector_checkpoint['state'])
            else:
                self.detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None,num_classes=91 if origin=='generic-coco-detector' else 2).eval()
                self.detector.load_state_dict(detector_checkpoint['state'])
        self.classifier, self.brands = None, []
        self.colour_classifier=None
        self.vehicle_classifier=None
        calibration=self.directory/'colour_calibration.json'
        self.colour_reliability_threshold=json.loads(calibration.read_text(encoding='utf-8'))['threshold'] if calibration.exists() else .75
        self.masked_colour=None
        if (self.directory/'masked_colour.pt').exists():
            from .masked_colour import MaskedColourClassifier
            self.masked_colour=MaskedColourClassifier(torch.load(self.directory/'masked_colour.pt',map_location='cpu',weights_only=True))
        self.mask_fusion=json.loads((self.directory/'mask_fusion.json').read_text(encoding='utf-8')) if (self.directory/'mask_fusion.json').exists() else dict(brand_mask_weight=.5,model_mask_weight=.5)
        self.public_prototype=False
        checkpoint = self.directory/'classifier.pt'
        if checkpoint.exists():
            ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
            if ck.get('origin') not in ('team-trained-from-scratch','team-trained-brand-head-imagenet-backbone'):raise ValueError('Expected a team-trained brand classifier.')
            self.brands = ck['brands']
            self.brand_temperature=float(ck.get('temperature',1.))
            self.multiview_brand=bool(ck.get('regional_training',False))
            self.public_prototype='public Stanford' in ck.get('source','')
            constructor=CompactResidualClassifier if ck.get('architecture')=='compact_residual_avgmax_v1' else Classifier
            self.classifier = constructor(len(self.brands)).eval()
            self.classifier.load_state_dict(ck['state'])
            self.preprocess=make_preprocess(int(ck.get('image_size',224)),16 if self.public_prototype else 0)
            if ck.get('joint_colour_head'):
                from .colour_model import JointColourClassifier
                self.colour_classifier=JointColourClassifier(self.classifier,self.preprocess);self.masked_colour=None;self.colour_reliability_threshold=.75
            if ck.get('colour_mode')=='body_histogram':self.colour_mode='heuristic'
        separate_colour=self.directory/'colour_model.pt'
        if separate_colour.exists() and self.colour_mode!='trained' and self.colour_classifier is None:
            from .colour_model import ColourClassifier
            self.colour_classifier=ColourClassifier(torch.load(separate_colour,map_location='cpu',weights_only=True))

        model_file=self.directory/'vehicle_model.pt'
        if model_file.exists() and self.classifier is not None:
            from .vehicle_model import VehicleModelClassifier
            self.vehicle_classifier=VehicleModelClassifier(torch.load(model_file,map_location='cpu',weights_only=True))
            reference_file=self.directory/'visual_references.pt'
            if reference_file.exists():
                import hashlib
                from .visual_parts import VisualReferences
                reference=torch.load(reference_file,map_location='cpu',weights_only=True)
                if reference['source_model_sha256']==hashlib.sha256(model_file.read_bytes()).hexdigest() and reference['labels']==self.vehicle_classifier.labels and reference['brands']==self.vehicle_classifier.brands:
                    self.vehicle_classifier.references=VisualReferences(reference)
            if self.vehicle_classifier.requires_references and self.vehicle_classifier.references is None:raise ValueError('Банк сравнения моделей отсутствует или не соответствует весам. Восстановите всю папку models из переносимого архива.')

    @torch.inference_mode()
    def predict(self, image, region=None, single=False):
        started = time.perf_counter()
        original = read_image(image)
        if region is not None:
            if len(region)!=4 or not all(np.isfinite(v) for v in region):raise ValueError('Invalid analysis region')
            x,y,x2,y2=region
            region=[round(max(0,min(original.width,x))),round(max(0,min(original.height,y))),round(max(0,min(original.width,x2))),round(max(0,min(original.height,y2)))]
            if min(region[2]-region[0],region[3]-region[1])<16:raise ValueError('Analysis region must be at least 16 × 16 pixels')
        if single and region is None:raise ValueError('Select a single-car region first')
        if getattr(self,'segmenter',None) is not None:
            from .scene import predict_instances
            return predict_instances(self,original,str(image),region,single,started)
        im = original.crop(region) if region is not None else original
        offset=region[:2] if region is not None else (0,0)
        # Bound memory for modern high-resolution phone photographs.
        work = im.copy(); work.thumbnail((1920,1920))
        tiles=[] if single else detection_tiles(*work.size) if self.mode!='fast' else [(0,0,*work.size)]
        all_boxes=[];all_scores=[];full_count=0
        batch_size=getattr(self,'detector_batch_size',4)
        for start in range(0,len(tiles),batch_size):
            batch=tiles[start:start+batch_size]
            detections=self.detector([transforms.functional.to_tensor(work.crop(tile)) for tile in batch])
            for d,tile in zip(detections,batch):
                b,s=restore_tile_boxes(d,tile,work.width,work.height,self.detection,getattr(self,'car_label',1))
                if tile==(0,0,*work.size):full_count=len(b)
                all_boxes.append(b);all_scores.append(s)
        if self.mode=='detail' and not single:
            d=self.detector([transforms.functional.to_tensor(ImageOps.mirror(work))])[0]
            b,s=restore_tile_boxes(d,(0,0,*work.size),*work.size,self.detection,getattr(self,'car_label',1))
            left=b[:,0].clone();b[:,0]=work.width-b[:,2];b[:,2]=work.width-left
            all_boxes.append(b);all_scores.append(s)
        boxes,scores=(torch.tensor([[0.,0.,*work.size]]),torch.tensor([1.])) if single else (torch.cat(all_boxes),torch.cat(all_scores))
        keep = merge_vehicle_boxes(boxes,scores,full_count)
        result = []
        for k in keep:
            box = boxes[k].tolist()
            box = [max(0,min(im.width,box[0]*im.width/work.width)),max(0,min(im.height,box[1]*im.height/work.height)),max(0,min(im.width,box[2]*im.width/work.width)),max(0,min(im.height,box[3]*im.height/work.height))]
            if box[2]-box[0]<16 or box[3]-box[1]<16: continue
            crop = im.crop(box)
            brand, bp = 'unknown', 0.
            colour, cp = heuristic_colour(crop)
            colour_source = 'original HSV body histogram'
            if self.classifier is not None:
                preprocess=getattr(self,'preprocess',PREPROCESS)
                views=[crop,ImageOps.mirror(crop)] if getattr(self,'multiview_brand',False) else [crop]
                b,c = self.classifier(torch.stack([preprocess(v) for v in views]))
                probabilities=(b/getattr(self,'brand_temperature',1.)).softmax(1).mean(0);bp,bi=probabilities.max(0); bp=float(bp)
                brand = self.brands[int(bi)] if bp>=self.brand_threshold else 'unknown'
                if self.colour_mode=='trained':
                    cp,ci=c.softmax(1).mean(0).max(0); cp=float(cp); colour=COLOURS[int(ci)]; colour_source='trained'
            if getattr(self,'colour_classifier',None) is not None:
                colour,cp=self.colour_classifier.predict(crop);colour_source='project-trained VCoR colour head'
            model_result=dict(model='unknown',model_candidate='unknown',model_confidence=0.,model_alternatives=[])
            if getattr(self,'vehicle_classifier',None) is not None:
                model_result=self.vehicle_classifier.predict_details(crop,brand,self.classifier,self.preprocess)
            global_box=[round(box[0]+offset[0]),round(box[1]+offset[1]),round(box[2]+offset[0]),round(box[3]+offset[1])]
            result.append(dict(brand=brand,**model_result,colour=colour,colour_reliable=cp>=getattr(self,'colour_reliability_threshold',.75),brand_confidence=bp,colour_confidence=cp,colour_source=colour_source,detection_confidence=None if single else float(scores[k]),box_source='user region' if single else 'detector',box=global_box))
        result.sort(key=lambda r:(r['box'][1],r['box'][0]))
        for i,r in enumerate(result,1): r['id']=i
        warnings=[] if self.classifier is not None else ['Brand classifier not trained; unknown is expected. Colour is an unvalidated HSV baseline.']
        if getattr(self,'public_prototype',False):warnings.append('Public-data prototype. Coverage is limited to supported_brands; full Tashkent-street accuracy has not been established. Colour uses a separately project-trained head.' if self.colour_classifier is not None else 'Public-data prototype. Coverage is limited to supported_brands; full Tashkent-street accuracy has not been established. Colour uses an original non-learned method.')
        return dict(image=str(image),cars=result,seconds=round(time.perf_counter()-started,3),classifier_trained=self.classifier is not None,supported_brands=getattr(self,'brands',[]),mode=self.mode,region=region,single_car_region=single,detector_views=len(tiles)+(self.mode=='detail' and not single),warnings=warnings)

def save_prediction(source, result, output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    from .vehicle_model import vehicle_name,detailed_answer
    stem=Path(source).stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (out/(stem+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/(stem+'.txt')).write_text('; '.join(r['brand']+' '+r['colour'] for r in result['cars']) or 'No cars detected',encoding='utf-8')
    (out/(stem+'.details.txt')).write_text('\n'.join(detailed_answer(r) for r in result['cars']) or 'No cars detected',encoding='utf-8')
    from .overlays import marker_positions,instance_colour
    im=read_image(source); draw=ImageDraw.Draw(im);font_size=max(14,im.width//90);size=max(24,im.width//60)
    # Pillow's default font lacks Cyrillic. Standard Windows fonts stay on the
    # destination machine; they are never bundled or redistributed with the app.
    try:font=ImageFont.truetype('segoeui.ttf',font_size)
    except OSError:
        try:font=ImageFont.truetype('DejaVuSans.ttf',font_size)
        except OSError:font=ImageFont.load_default(size=font_size)
    markers=marker_positions([r['box'] for r in result['cars']],im.width,im.height,size)
    for r in result['cars']:
        colour=instance_colour(r['id'])
        if r.get('contours'):
            for contour in r['contours']:draw.line([tuple(v) for v in contour]+[tuple(contour[0])],fill=colour,width=max(2,im.width//400))
        else:draw.rectangle(r['box'],outline=colour,width=max(2,im.width//400))
    for r,rect in zip(result['cars'],markers):
        colour=instance_colour(r['id']);draw.line((r['box'][0],r['box'][1],rect[0]+size/2,rect[1]+size/2),fill=colour,width=2)
        draw.rectangle(rect,fill='#14243a',outline=colour,width=2);draw.text((rect[0]+4,rect[1]+2),str(r['id']),font=font,fill='white')
    # A separate legend cannot cover vehicles or other captions.
    if result['cars']:
        line=max(25,im.width//60);annotated=Image.new('RGB',(im.width,im.height+line*(len(result['cars'])+1)), '#14243a');annotated.paste(im,(0,0));draw=ImageDraw.Draw(annotated)
        for i,r in enumerate(result['cars']):draw.text((12,im.height+line*(i+.5)),f"{r['id']}: {vehicle_name(r)} | {r['colour']}{' (uncertain)' if not r.get('colour_reliable',True) else ''}",font=font,fill='white')
        im=annotated
    im.save(out/(stem+'_annotated.jpg'))
    return out/(stem+'_annotated.jpg')
