import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import nms

ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
COLOURS = ['white', 'black', 'grey', 'red', 'blue', 'green', 'brown', 'yellow']
BRANDS = ['Chevrolet', 'Daewoo', 'Lada', 'Kia', 'Hyundai', 'Toyota', 'Mercedes-Benz', 'BMW', 'Volkswagen', 'Nissan', 'BYD', 'Chery', 'Haval', 'Changan', 'Geely', 'Jetour', 'Li Auto', 'Zeekr', 'Exeed', 'Omoda', 'Jaecoo', 'Hongqi', 'Leapmotor', 'JAC']
PREPROCESS = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

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

def heuristic_colour(crop):
    # Baseline only: central body region, saturation/value histogram.
    w,h = crop.size
    a = np.asarray(crop.crop((w*.15,h*.4,w*.85,h*.85)).resize((96,64)).convert('HSV')).astype(float)
    hue,s,v = a[:,:,0]*360/255, a[:,:,1]/255, a[:,:,2]/255
    masks = [(s<.19)&(v>.76), v<.22, (s<.22)&(v>=.22)&(v<=.76), ((hue<18)|(hue>=345))&(s>=.22)&(v>=.22), (hue>=185)&(hue<275)&(s>=.22)&(v>=.22), (hue>=65)&(hue<185)&(s>=.22)&(v>=.22), (hue>=18)&(hue<65)&(s>=.22)&(v<.68), (hue>=18)&(hue<65)&(s>=.22)&(v>=.68)]
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
    mask=(detection['labels']==car_label)&(detection['scores']>=threshold)
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

class Pipeline:
    def __init__(self, directory=None, detection=.5, brand_threshold=.65, colour_mode='trained', mode='balanced'):
        torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
        self.directory = Path(directory or ROOT/'models')
        self.detection, self.brand_threshold, self.colour_mode = detection, brand_threshold, colour_mode
        if mode not in ('fast','balanced','detail'):raise ValueError('Unknown detection mode')
        self.mode=mode
        detector_file = self.directory/'detector.pt'
        if not detector_file.exists():
            raise FileNotFoundError('Детектор ещё не обучен. Добавьте свои фото, разметьте все машины и запустите обучение во вкладке «Обучение». Готовые веса не используются.')
        detector_checkpoint=torch.load(detector_file,map_location='cpu',weights_only=True)
        if detector_checkpoint.get('origin')!='team-trained-from-scratch':raise ValueError('Expected an original detector trained from scratch.')
        self.detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None,num_classes=2).eval()
        self.detector.load_state_dict(detector_checkpoint['state'])
        self.classifier, self.brands = None, []
        checkpoint = self.directory/'classifier.pt'
        if checkpoint.exists():
            ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
            if ck.get('origin')!='team-trained-from-scratch':raise ValueError('Expected an original classifier trained from scratch.')
            self.brands = ck['brands']
            self.classifier = Classifier(len(self.brands)).eval()
            self.classifier.load_state_dict(ck['state'])

    @torch.inference_mode()
    def predict(self, image):
        started = time.perf_counter()
        im = read_image(image)
        # Bound memory for modern high-resolution phone photographs.
        work = im.copy(); work.thumbnail((1920,1920))
        tiles=detection_tiles(*work.size) if self.mode!='fast' else [(0,0,*work.size)]
        all_boxes=[];all_scores=[]
        for start in range(0,len(tiles),4):
            batch=tiles[start:start+4]
            detections=self.detector([transforms.functional.to_tensor(work.crop(tile)) for tile in batch])
            for d,tile in zip(detections,batch):
                b,s=restore_tile_boxes(d,tile,work.width,work.height,self.detection)
                all_boxes.append(b);all_scores.append(s)
        if self.mode=='detail':
            d=self.detector([transforms.functional.to_tensor(ImageOps.mirror(work))])[0]
            b,s=restore_tile_boxes(d,(0,0,*work.size),*work.size,self.detection)
            left=b[:,0].clone();b[:,0]=work.width-b[:,2];b[:,2]=work.width-left
            all_boxes.append(b);all_scores.append(s)
        boxes,scores=torch.cat(all_boxes),torch.cat(all_scores)
        keep = nms(boxes,scores,.45)
        result = []
        for k in keep:
            box = boxes[k].tolist()
            box = [max(0,min(im.width,box[0]*im.width/work.width)),max(0,min(im.height,box[1]*im.height/work.height)),max(0,min(im.width,box[2]*im.width/work.width)),max(0,min(im.height,box[3]*im.height/work.height))]
            if box[2]-box[0]<16 or box[3]-box[1]<16: continue
            crop = im.crop(box)
            brand, bp = 'unknown', 0.
            colour, cp = heuristic_colour(crop)
            colour_source = 'HSV baseline'
            if self.classifier is not None:
                b,c = self.classifier(PREPROCESS(crop).unsqueeze(0))
                bp,bi = b.softmax(1).max(1); bp=float(bp)
                brand = self.brands[int(bi)] if bp>=self.brand_threshold else 'unknown'
                if self.colour_mode=='trained':
                    cp,ci=c.softmax(1).max(1); cp=float(cp); colour=COLOURS[int(ci)]; colour_source='trained'
            result.append(dict(brand=brand,colour=colour,brand_confidence=bp,colour_confidence=cp,colour_source=colour_source,detection_confidence=float(scores[k]),box=[round(v) for v in box]))
        result.sort(key=lambda r:(r['box'][1],r['box'][0]))
        for i,r in enumerate(result,1): r['id']=i
        return dict(image=str(image),cars=result,seconds=round(time.perf_counter()-started,3),classifier_trained=self.classifier is not None,mode=self.mode,detector_views=len(tiles)+(self.mode=='detail'),warnings=[] if self.classifier is not None else ['Brand classifier not trained; unknown is expected. Colour is an unvalidated HSV baseline.'])

def save_prediction(source, result, output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    stem=Path(source).stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (out/(stem+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/(stem+'.txt')).write_text('; '.join(r['brand']+' '+r['colour'] for r in result['cars']) or 'No cars detected',encoding='utf-8')
    im=read_image(source); draw=ImageDraw.Draw(im)
    for r in result['cars']:
        draw.rectangle(r['box'],outline='#14b8a6',width=max(2,im.width//400))
        draw.text((r['box'][0]+3,r['box'][1]+3),f"{r['id']}: {r['brand']} {r['colour']}",fill='white',stroke_width=2,stroke_fill='black')
    im.save(out/(stem+'_annotated.jpg'))
    return out/(stem+'_annotated.jpg')
