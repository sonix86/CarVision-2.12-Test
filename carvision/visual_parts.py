"""Our labelled reference descriptors; no downloaded model-name classifier."""
import torch

class VisualReferences:
    def __init__(self,checkpoint):
        if checkpoint.get('origin')!='project-trained-visible-part-references':raise ValueError('Expected our train-only part references')
        self.features=checkpoint['features'];self.targets=checkpoint['targets'];self.ratios=checkpoint['ratios']
        self.labels=checkpoint['labels'];self.brands=checkpoint['brands'];self.temperature=checkpoint['temperature'];self.penalty=checkpoint['aspect_penalty'];self.neighbours=checkpoint['neighbours'];self.weight=checkpoint['weight']
        self.probability_temperature=checkpoint['probability_temperature'];self.threshold=checkpoint['confidence_threshold']

    @torch.inference_mode()
    def probabilities(self,feature,ratio):
        similarity=feature@self.features.T-self.penalty*(torch.tensor(float(ratio)).log()-self.ratios.log()).abs()
        scores=torch.stack([similarity[self.targets==i].topk(min(self.neighbours,int((self.targets==i).sum()))).values.mean() for i in range(len(self.labels))])
        return (scores/self.temperature).softmax(0)

    def calibrated(self,p):
        p=p.clamp_min(1e-9).pow(1/self.probability_temperature);return p/p.sum()

def quality_of_fragment(mask):
    """Geometry warnings, not a claimed estimate of the hidden vehicle area."""
    h,w=mask.shape;reasons=[]
    if w/max(1,h)>3.2:reasons.append('Виден узкий фрагмент: крыша или часть кузова. По нему модель не подтверждается.')
    if min(h,w)<64 or int(mask.sum())<1200:reasons.append('Машина мала в исходном кадре: не хватает различимых деталей.')
    return reasons
