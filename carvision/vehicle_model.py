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
        self.body_expert=None
        self.badge_evidence=None
        if checkpoint.get('badge_evidence'):
            from .badge_evidence import BadgeEvidence
            self.badge_evidence=BadgeEvidence(checkpoint['badge_evidence'])
        if checkpoint.get('body_expert'):
            from .body_expert import BodyExpert
            self.body_expert=BodyExpert(checkpoint['body_expert'],list(zip(self.brands,self.labels)))
        self.references=None
        self.requires_references=checkpoint.get('architecture')=='reference-expanded-model-network-v1'
        self.legacy_model=None;self.legacy_weight=float(checkpoint.get('legacy_weight',0.))
        if not 0<=self.legacy_weight<=1:raise ValueError('Invalid legacy mixture weight')
        if checkpoint.get('legacy_model'):
            legacy=checkpoint['legacy_model']
            if legacy.get('legacy_model'):raise ValueError('Nested legacy mixtures are not supported')
            self.legacy_model=VehicleModelClassifier(legacy)
            from .visual_parts import VisualReferences
            reference=checkpoint['legacy_references']
            if reference['source_model_sha256']!=checkpoint['legacy_model_sha256'] or reference['labels']!=legacy['labels'] or reference['brands']!=legacy['brands']:raise ValueError('Legacy reference provenance mismatch')
            self.legacy_model.references=VisualReferences(reference)
            if list(zip(legacy['brands'],legacy['labels']))!=list(zip(self.brands,self.labels))[:len(legacy['labels'])]:raise ValueError('Legacy labels must retain the exact prefix order')
        self.network=None
        if checkpoint.get('architecture') in ('finetuned-model-network-v1','reference-expanded-model-network-v1'):
            from .core import Classifier
            neural_labels=checkpoint.get('neural_labels',self.labels);neural_brands=checkpoint.get('neural_brands',self.brands)
            if list(zip(neural_brands,neural_labels))!=list(zip(self.brands,self.labels))[:len(neural_labels)]:raise ValueError('Neural taxonomy must be an exact prefix of the reference taxonomy')
            self.network=Classifier(len(neural_labels)).eval();self.network.load_state_dict(checkpoint['state'])
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
            probability=logits.softmax(1).mean(0)
            if len(probability)<len(self.labels):probability=torch.nn.functional.pad(probability,(0,len(self.labels)-len(probability)))
            if self.references is not None:
                feature=torch.nn.functional.normalize(torch.nn.functional.normalize(f,dim=1).mean(0),dim=0)
                reference=self.references.probabilities(feature,image.width/image.height)
                probability=probability*(1-self.references.weight)+reference*self.references.weight
            if self.legacy_model is not None and self.legacy_weight:
                old=self.legacy_model.probabilities(image,network,preprocess);old=torch.nn.functional.pad(old,(0,len(probability)-len(old)))
                probability=probability*(1-self.legacy_weight)+old*self.legacy_weight
            return self._body_blend(probability,image)
        else:
            x=pooled_features(network,views)
            logits=self.head((x-self.mean)/self.std)/self.temperature
        return self._body_blend(logits.softmax(1).mean(0),image)

    def _body_blend(self,probability,image):
        if self.body_expert is None:return probability
        weight=self.body_expert.weight
        return probability*(1-weight)+self.body_expert.probabilities(image)*weight

    def describe(self,probabilities,brand):
        if self.references is not None:probabilities=self.references.calibrated(probabilities)
        unsupported_brand=brand!='unknown' and brand not in self.brands
        confidence,index=probabilities.max(0);index=int(index);confidence=float(confidence)
        candidate=self.labels[index];candidate_brand=self.brands[index]
        alternatives=[dict(brand=self.brands[int(i)],model=self.labels[int(i)],confidence=float(probabilities[i])) for i in probabilities.topk(min(3,len(self.labels))).indices]
        threshold=self.references.threshold if self.references is not None else self.threshold
        ranking=[dict(brand=self.brands[int(i)],model=self.labels[int(i)],confidence=float(probabilities[i])) for i in probabilities.argsort(descending=True)]
        result=dict(model=candidate if confidence>=threshold and candidate_brand==brand else 'unknown',model_candidate=candidate,model_candidate_brand=candidate_brand,model_confidence=confidence,model_alternatives=alternatives,model_ranking=ranking,reference_comparison=self.references is not None)
        if unsupported_brand:result['model_reason']='No model-name training examples for this manufacturer; alternatives compare supported classes only'
        return result

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
