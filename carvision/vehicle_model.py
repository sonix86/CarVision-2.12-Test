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
    def predict_details(self, image, brand, network, preprocess):
        if brand=='unknown' or brand not in self.brands:return dict(model='unknown',model_candidate='unknown',model_confidence=0.,model_alternatives=[])
        if self.network is not None:
            from .core import make_preprocess
            base=make_preprocess(160,16)
            preprocess=lambda im:base(im.resize((176,176),Image.Resampling.BICUBIC))
        views=torch.stack([preprocess(image),preprocess(ImageOps.mirror(image))])
        if self.network is not None:logits=self.network(views)[0]/self.temperature
        else:
            x=pooled_features(network,views)
            logits=self.head((x-self.mean)/self.std)/self.temperature
        # Global probabilities preserve uncertainty for unrepresented models;
        # conditioning on a brand with a single class would falsely give 100%.
        probabilities=logits.softmax(1).mean(0);confidence,index=probabilities.max(0)
        index=int(index);confidence=float(confidence)
        candidate=self.labels[index] if self.brands[index]==brand else 'unknown'
        alternatives=[dict(brand=self.brands[int(i)],model=self.labels[int(i)],confidence=float(probabilities[i])) for i in probabilities.topk(min(3,len(self.labels))).indices]
        return dict(model=candidate if confidence>=self.threshold else 'unknown',model_candidate=candidate,model_confidence=confidence,model_alternatives=alternatives)

    def predict(self,image,brand,network,preprocess):
        result=self.predict_details(image,brand,network,preprocess)
        return result['model'],result['model_confidence']

def vehicle_name(row):
    model=row.get('model','unknown')
    if model!='unknown':return row['brand']+' '+model
    candidate=row.get('model_candidate','unknown')
    return row['brand']+(' '+candidate+'?' if candidate!='unknown' else '')
