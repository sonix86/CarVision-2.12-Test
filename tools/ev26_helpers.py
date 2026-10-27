"""Shared validation and runtime-equivalent routing; never selects using test."""
import torch
from carvision.visual_retrieval import retrieval_probabilities

def geometry(rows,kinds=('whole','phone')):
    result=[]
    for kind in kinds:
        for r in rows:
            h,w=r['body_rgb'].shape[:2];s=min(1,96/max(h,w)) if kind=='phone' else 1;result.append(min(round(w*s),round(h*s))>=40)
    return torch.tensor(result)

def acceptance(p,thresholds,brands,valid,margin=.06):
    v,ids=p.topk(2,dim=1);g=torch.tensor([thresholds[brands[int(j)]] for j in ids[:,0]]);return (g<1)&(v[:,0]>=g)&((v[:,0]-v[:,1])>=margin)&valid

def calibrate(p,truth,groups,brands,valid,precision,exclude=None):
    v,ids=p.topk(2,dim=1);thresholds={}
    for j,b in enumerate(brands):
        options=[]
        for t in (.5,.6,.7,.8,.9,.95,.98):
            a=(ids[:,0]==j)&(v[:,0]>=t)&((v[:,0]-v[:,1])>=.06)&valid
            if exclude is not None:a&=~exclude
            if len(groups[a].unique())>=3 and float((truth[a]==j).float().mean())>=precision:options.append((int(a.sum()),t))
        thresholds[b]=max(options,default=(0,1.))[1]
    return thresholds

def route(p,reference,thresholds,fallback_thresholds,brands,valid):
    primary=acceptance(p,thresholds,brands,valid);mixed=p*.5+reference*.5;chosen=torch.where(primary[:,None],p,mixed);accepted=primary|((~primary)&acceptance(mixed,fallback_thresholds,brands,valid));return chosen,accepted

def reference_bank(train,features,brands):
    from carvision.semantic_make import embedding_features
    from tools.cache_semantic_v25 import token
    names=sorted({r['group'] for r in train});return dict(keys=torch.stack([torch.stack([embedding_features(features[token(r,k)],False) for r in train]) for k in ('whole','phone')]),labels=torch.tensor([r['y'] for r in train]),groups=torch.tensor([names.index(r['group']) for r in train]),k=8,alpha=.5,beta=50.,mode='abstention-only')

def references(q,bank,classes):return retrieval_probabilities(q,bank['keys'],bank['labels'],bank['groups'],classes,8,50.)

def metrics(p,accepted,rows,brands,priority):
    truth=torch.tensor([r['y'] for r in rows]*2);historical=torch.tensor([r['historical'] for r in rows]);n=len(rows);result={}
    for i,kind in enumerate(('whole','phone')):
        good=(p.argmax(1)==truth)[i*n:(i+1)*n];a=accepted[i*n:(i+1)*n];y=truth[i*n:(i+1)*n]
        result[kind]=dict(top1=float(good.float().mean()),historical=float(good[historical].float().mean()),macro=float(torch.tensor([float(good[y==j].float().mean()) for j in y.unique()]).mean()),priority_macro=float(torch.tensor([float(good[y==brands.index(b)].float().mean()) for b in priority if (y==brands.index(b)).any()]).mean()),confirmed=int(a.sum()),correct_confirmed=int((a&good).sum()),precision=float(good[a].float().mean()) if a.any() else 0.)
    return result
