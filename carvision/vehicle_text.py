"""General local OCR as auxiliary features; no car-brand rules or remote calls."""
import hashlib,re,unicodedata
from pathlib import Path
import numpy as np
from PIL import Image

MODEL_FILES=('ch_PP-OCRv4_det_infer.onnx','ch_ppocr_mobile_v2.0_cls_infer.onnx','ch_PP-OCRv4_rec_infer.onnx')

def text_features(records,dimension=512):
    result=np.zeros(dimension,dtype=np.float32)
    for record in records:
        word=''.join(c for c in unicodedata.normalize('NFKC',record['text']).lower() if c.isalnum())
        # Plates, telephone numbers, websites and watermarks are not make evidence.
        if len(word)<2 or len(word)>40 or sum(c.isdigit() for c in word)>=2:continue
        if any(s in word for s in ('http','www','copyright','telegram')):continue
        for size in (2,3,4):
            for i in range(max(0,len(word)-size+1)):
                token=word[i:i+size];digest=hashlib.sha256(token.encode('utf-8')).digest()
                result[int.from_bytes(digest[:4],'little')%dimension]+=float(record['confidence'])
    norm=np.linalg.norm(result)
    return result/max(float(norm),1.)

class VehicleTextReader:
    def __init__(self,directory):
        from rapidocr_onnxruntime import RapidOCR
        directory=Path(directory)
        for name in MODEL_FILES:
            if not (directory/name).is_file():raise FileNotFoundError('Missing offline OCR weight: '+name)
        self.engine=RapidOCR(det_model_path=str(directory/MODEL_FILES[0]),cls_model_path=str(directory/MODEL_FILES[1]),rec_model_path=str(directory/MODEL_FILES[2]),intra_op_num_threads=2,inter_op_num_threads=1,det_limit_side_len=640,det_limit_type='max',text_score=.7)

    def read(self,image,mask):
        image=image.convert('RGB');scale=min(3.,800/max(image.size));scaled=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.BICUBIC)
        # The reader expects OpenCV BGR arrays. Returned boxes are mapped to the car crop.
        rows,_=self.engine(np.asarray(scaled)[:,:,::-1].copy());result=[];mask=np.asarray(mask,dtype=bool)
        for box,text,confidence in rows or []:
            pts=np.asarray(box,dtype=float)/scale;x1,y1=pts.min(0);x2,y2=pts.max(0)
            x1,y1=max(0,int(x1)),max(0,int(y1));x2,y2=min(image.width,int(np.ceil(x2))),min(image.height,int(np.ceil(y2)))
            if x2<=x1 or y2<=y1 or float(mask[y1:y2,x1:x2].mean())<.8:continue
            if float(confidence)<.7:continue
            record=dict(text=text,confidence=float(confidence),box=pts.tolist())
            # Keep only text that survives the same plate/watermark filtering as training.
            if text_features([record]).any():result.append(record)
        return result
