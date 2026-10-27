"""Project-trained make recognition, independent of the closed model vocabulary."""
import torch
from PIL import Image,ImageOps
from torchvision import models
from .body_expert import body_tensor
from .segmentation import isolate


def make_head(classes):
    return torch.nn.Sequential(torch.nn.Linear(512,256),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(256,classes))


class MakeExpert:
    def __init__(self,checkpoint):
        if checkpoint.get('origin')!='project-trained-make-expert-v1':
            raise ValueError('Expected the project-trained make expert')
        self.brands=checkpoint['brands']
        self.display_names=checkpoint.get('display_names',{})
        if not self.brands or len(set(self.brands))!=len(self.brands):raise ValueError('Invalid make taxonomy')
        self.encoder=models.resnet18(weights=None);self.encoder.fc=torch.nn.Identity()
        self.encoder.load_state_dict(checkpoint['encoder']);self.encoder.eval()
        self.head=make_head(len(self.brands));self.head.load_state_dict(checkpoint['head']);self.head.eval()
        self.temperature=float(checkpoint['temperature']);self.thresholds=checkpoint['thresholds']
        self.margin=float(checkpoint.get('margin',.15))
        if self.temperature<=0 or not 0<=self.margin<=1:raise ValueError('Invalid make calibration')
        if any(not 0<=float(v)<=1 for v in self.thresholds.values()):raise ValueError('Invalid make threshold')

    @torch.inference_mode()
    def feature(self,image,mask=None):
        focus=isolate(image,mask) if mask is not None else image
        views=torch.stack([body_tensor(v) for im in [image,focus] for v in [im,ImageOps.mirror(im)]])
        z=self.encoder(views);z=z[:2].mean(0)*.75+z[2:].mean(0)*.25
        return torch.nn.functional.normalize(z,dim=0)

    @torch.inference_mode()
    def predict(self,image,mask=None,minimum=None):
        p=(self.head(self.feature(image,mask))/self.temperature).softmax(0)
        values,indices=p.topk(min(3,len(p)));first=int(indices[0]);confidence=float(values[0])
        margin=confidence-(float(values[1]) if len(values)>1 else 0.)
        calibrated_threshold=float(self.thresholds.get(self.brands[first],1.));threshold=calibrated_threshold
        if minimum is not None:
            if not 0<=float(minimum)<=1:raise ValueError('Минимальная уверенность марки должна быть от 0 до 1')
            threshold=max(threshold,float(minimum))
        accepted=threshold<1. and confidence>=threshold and confidence>0 and margin>=self.margin and min(image.size)>=40
        reason=('Марка прошла проверочный порог; отдельные ошибки всё ещё возможны.' if accepted else
                'Для этой марки пока недостаточно независимых проверочных примеров. Показано только предположение.' if calibrated_threshold>=1. else
                'Пользовательский порог запрещает подтверждение марки. Показано только предположение.' if threshold>=1. else
                'Машина слишком мала для подтверждения марки.' if min(image.size)<40 else
                'Сходства недостаточно или несколько марок получили близкие оценки. Показаны варианты.')
        display=lambda b:getattr(self,'display_names',{}).get(b,b)
        return dict(brand=self.brands[first] if accepted else 'unknown',candidate=self.brands[first],
                    display_name=display(self.brands[first]),
                    confidence=confidence,accepted=accepted,margin=margin,threshold=threshold,acceptance_reason=reason,
                    alternatives=[dict(brand=self.brands[int(i)],display_name=display(self.brands[int(i)]),confidence=float(p[i])) for i in indices],
                    ranking=[dict(brand=self.brands[int(i)],display_name=display(self.brands[int(i)]),confidence=float(p[i])) for i in p.argsort(descending=True)],
                    source='independent project-trained make expert; grouped-validation acceptance')


def model_details_for_make(model,brand):
    """Keep calibrated global probabilities; conditioning must not invent certainty."""
    if brand=='unknown':return model
    result=dict(model)
    all_options=model.get('model_ranking',model.get('model_alternatives',[]))
    same=[r for r in all_options if r['brand']==brand]
    result['model_alternatives']=same[:3]
    if same:
        best=same[0]
        changed=(model.get('model_candidate_brand')!=brand or model.get('model_candidate')!=best['model'])
        result.update(model_candidate_brand=brand,model_candidate=best['model'],model_confidence=best['confidence'])
        # Preserve an already independently accepted same-make model only.
        if changed:result['model']='unknown'
    else:
        result.update(model='unknown',model_candidate='unknown',model_candidate_brand=brand,model_confidence=0.,
                      model_reason='Manufacturer recognized; model families for this make are not trained')
    return result
