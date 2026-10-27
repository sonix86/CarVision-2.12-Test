"""Two bounded paint fits; preserved validation/test, owner outing TRAIN only."""
import copy,json,sys
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import COLOURS
from carvision.masked_colour import features,head,MaskedColourClassifier
from tools.cache_owner_domain_v29 import view

def main():
    torch.set_num_threads(3);out=ROOT/'models/owner_domain_v29'
    with (out/'colour_training_once.json').open('x') as f:json.dump(dict(rounds=2,epochs=100),f)
    data=torch.load(ROOT/'cache/masked_colour_v15.pt',weights_only=True)['rows']
    train=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='validation']
    own=[r for r in torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True)['rows'] if r.get('domain_v29')]
    extra=[(r,k) for r in own for k in ('whole','flip','cast')]
    x=torch.stack([r['x'] for r in train]+[features(*view(r,k)[:2]) for r,k in extra]);y=torch.tensor([COLOURS.index(r['colour']) for r in train]+[COLOURS.index(r['colour']) for r,k in extra])
    q=torch.stack([r['x'] for r in val]);vy=torch.tensor([COLOURS.index(r['colour']) for r in val]);base=torch.stack([r['baseline'] for r in val])
    old=MaskedColourClassifier(torch.load(ROOT/'outputs/CarVision-2.6-Full/CarVision/models/masked_colour.pt',weights_only=True))
    with torch.inference_mode():bp=base*.5+old.feature_probabilities(q)*.5;baseline=float((bp.argmax(1)==vy).float().mean())
    mean=x.mean(0);std=x.std(0).clamp_min(.03);x=(x-mean)/std;q=(q-mean)/std;rounds=[];best=None
    for rid in (1,2):
        torch.manual_seed(290100+rid);model=head(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.02);sw=torch.tensor([1.]*len(train)+[6. if rid==1 else 12.]*len(extra));chosen=None
        for epoch in range(100):
            model.train()
            for ids in torch.randperm(len(x)).split(64):
                opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids],label_smoothing=.02,reduction='none');(loss.mul(sw[ids]).sum()/sw[ids].sum()).backward();opt.step()
            model.eval()
            with torch.inference_mode():p=base*.5+model(q).softmax(1)*.5;acc=float((p.argmax(1)==vy).float().mean())
            if acc>=baseline and (chosen is None or acc>chosen['accuracy']):chosen=dict(accuracy=acc,epoch=epoch+1,state=copy.deepcopy(model.state_dict()),round=rid)
        rounds.append(dict(round=rid,passed=chosen is not None,selected={k:v for k,v in chosen.items() if k!='state'} if chosen else None))
        if chosen and (best is None or chosen['accuracy']>best['accuracy']):best=chosen
    report=dict(passed=best is not None,baseline=baseline,rounds=rounds,limitations='Cached base-colour probabilities shared identically by both paint heads. Not a full current-runtime colour benchmark. Own photos TRAIN only; historical VCoR source splits lack certified vehicle identity groups.')
    if best:
        ck=dict(origin='project-trained-masked-paint-head',state=best['state'],mean=mean,std=std,classes=COLOURS,temperature=1.,weight=.5,threshold=old.threshold)
        test=[r for r in data if r['split']=='test'];tx=torch.stack([r['x'] for r in test]);ty=torch.tensor([COLOURS.index(r['colour']) for r in test]);tb=torch.stack([r['baseline'] for r in test]);new=MaskedColourClassifier(ck)
        with torch.inference_mode():
            a=float(((tb*.5+old.feature_probabilities(tx)*.5).argmax(1)==ty).float().mean());b=float(((tb*.5+new.feature_probabilities(tx)*.5).argmax(1)==ty).float().mean())
        report['test']=dict(baseline=a,candidate=b,passed=b>=a,count=len(test));report['passed']=b>=a
        if report['passed']:torch.save(ck,out/'masked_colour.pt')
    (out/'colour_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
