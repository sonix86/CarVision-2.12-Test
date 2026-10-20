"""Model-name head trained locally; no downloaded vehicle-model classifier."""
import torch
from torch import nn
from PIL import Image,ImageOps

def pooled_features(network, images):
    f=network.features(images)
    return torch.cat([f.mean((2,3)),f.amax((2,3))],1)

class VehicleModelClassifier:
    def __init__(self, checkpoint):
        if checkpoint.get('origin')!='project-trained-regional-model-head':
            raise ValueError('Expected our independently trained model-name head')
        self.labels=checkpoint['labels'];self.brands=checkpoint['brands']
        self.references=None
        self.network=None
        if checkpoint.get('architecture')=='finetuned-model-network-v1':
            from .core import Classifier
            self.network=Classifier(len(self.labels)).eval();self.network.load_state_dict(checkpoint['state'])
            self.temperature=checkpoint['temperature'];self.threshold=checkpoint['threshold'];return
        self.mean=checkpoint['mean'];self.std=checkpoint['std']
        self.head=nn.Linear(len(self.mean),len(self.labels)).eval()
        self.head.load_state_dict(checkpoint['state'])
        self.temperature=checkpoint['temperature'];self.threshold=checkpoint['threshold']
    @torch.inference_mode()
    def probabilities(self,image,network,preprocess):
        if self.network is not None:
            from .core import make_preprocess
            base=make_preprocess(160,16)
            preprocess=lambda im:base(im.resize((176,176),Image.Resampling.BICUBIC))
        views=torch.stack([preprocess(image),preprocess(ImageOps.mirror(image))])
        if self.network is not None:
            x=pooled_features(self.network,views);f=self.network.neck(x);logits=self.network.brand(f)/self.temperature
            if self.references is not None:
                feature=torch.nn.functional.normalize(torch.nn.functional.normalize(f,dim=1).mean(0),dim=0)
                reference=self.references.probabilities(feature,image.width/image.height)
                return logits.softmax(1).mean(0)*(1-self.references.weight)+reference*self.references.weight
        else:
            x=pooled_features(network,views)
            logits=self.head((x-self.mean)/self.std)/self.temperature
        return logits.softmax(1).mean(0)

    def describe(self,probabilities,brand):
        if self.references is not None:probabilities=self.references.calibrated(probabilities)
        if brand!='unknown' and brand not in self.brands:
            return dict(model='unknown',model_candidate='unknown',model_confidence=0.,model_alternatives=[],model_reason='No model-name training examples for this manufacturer')
        confidence,index=probabilities.max(0);index=int(index);confidence=float(confidence)
        candidate=self.labels[index];candidate_brand=self.brands[index]
        alternatives=[dict(brand=self.brands[int(i)],model=self.labels[int(i)],confidence=float(probabilities[i])) for i in probabilities.topk(min(3,len(self.labels))).indices]
        threshold=self.references.threshold if self.references is not None else self.threshold
        ranking=[dict(brand=self.brands[int(i)],model=self.labels[int(i)],confidence=float(probabilities[i])) for i in probabilities.argsort(descending=True)]
        return dict(model=candidate if confidence>=threshold and candidate_brand==brand else 'unknown',model_candidate=candidate,model_candidate_brand=candidate_brand,model_confidence=confidence,model_alternatives=alternatives,model_ranking=ranking,reference_comparison=self.references is not None)

    @torch.inference_mode()
    def predict_details(self,image,brand,network,preprocess):
        return self.describe(self.probabilities(image,network,preprocess),brand)

    def predict(self,image,brand,network,preprocess):
        result=self.predict_details(image,brand,network,preprocess)
        return result['model'],result['model_confidence']

def vehicle_name(row):
    if row.get('model_limited'):return 'Недостаточно деталей'
    model=row.get('model','unknown')
    if model!='unknown':return row['brand']+' '+model
    candidate=row.get('model_candidate','unknown');candidate_brand=row.get('model_candidate_brand',row['brand'])
    if row['brand']=='unknown' and candidate!='unknown':return candidate_brand+' '+candidate+'?'
    return row['brand']+(' '+candidate+'?' if candidate!='unknown' and candidate_brand==row['brand'] else '')

def detailed_answer(row):
    text=vehicle_name(row)+' '+row['colour']+(' (uncertain colour)' if not row.get('colour_reliable',True) else '')
    alternatives=row.get('model_alternatives',[])
    if alternatives:text+=' | models: '+', '.join(f"{r['brand']} {r['model']} {r['confidence']:.0%}" for r in alternatives)
    return text
