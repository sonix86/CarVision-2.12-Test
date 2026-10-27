"""Supervised visual reference bank. Each source group casts at most one vote."""
import torch
import math

def validate_bank(bank,classes,dimension):
    if bank.get('mode')!='abstention-only':raise ValueError('Unsupported reference routing')
    keys,labels,groups=bank['keys'],bank['labels'],bank['groups']
    if keys.ndim!=3 or keys.shape[0]!=2 or keys.shape[2]!=dimension or keys.shape[1]<1:raise ValueError('Invalid reference geometry')
    if labels.shape!=(keys.shape[1],) or groups.shape!=labels.shape or labels.dtype!=torch.long or groups.dtype!=torch.long:raise ValueError('Invalid reference labels/groups')
    if not torch.isfinite(keys).all() or int(labels.min())<0 or int(labels.max())>=classes or int(groups.min())<0:raise ValueError('Invalid reference values')
    if not isinstance(bank['k'],int) or bank['k']<1 or not 0<bank['alpha']<1 or not math.isfinite(bank['beta']) or bank['beta']<=0:raise ValueError('Invalid reference mixture')

def routed_probabilities(query,neural,bank,brands,thresholds,margin,geometry,minimum=.5):
    """Keep a confirmed primary prediction byte-for-byte; fill abstentions only."""
    v,ids=neural.topk(2);b=brands[int(ids[0])];gate=max(float(thresholds[b]),minimum)
    primary=gate<1 and float(v[0])>=gate and float(v[0]-v[1])>=margin and geometry
    if primary:return neural,gate,'primary'
    ref=retrieval_probabilities(query,bank['keys'],bank['labels'],bank['groups'],len(brands),bank['k'],bank['beta'])
    mixed=neural*(1-bank['alpha'])+ref*bank['alpha'];b=brands[int(mixed.argmax())]
    return mixed,max(float(bank['fallback_thresholds'][b]),minimum),'reference-fallback'

@torch.inference_mode()
def retrieval_probabilities(query,keys,labels,groups,classes,k=3,beta=50.):
    single=query.ndim==1
    if single:query=query[None]
    query=torch.nn.functional.normalize(query,dim=1)
    # Two illumination views of the same training image never cast two votes.
    keys=torch.nn.functional.normalize(keys,dim=2)
    similarity=torch.einsum('bd,vnd->bvn',query,keys).amax(1)
    values,ids=similarity.topk(min(64,similarity.shape[1]),dim=1);result=torch.zeros((len(query),classes),dtype=query.dtype)
    for i in range(len(query)):
        seen=set();chosen=[]
        for j,index in enumerate(ids[i].tolist()):
            group=int(groups[index])
            if group in seen:continue
            seen.add(group);chosen.append(j)
            if len(chosen)>=k:break
        weights=(values[i,chosen]*beta).softmax(0)
        for weight,j in zip(weights,chosen):result[i,int(labels[int(ids[i,j])])]+=weight
    return result[0] if single else result
