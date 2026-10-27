"""One own colour head, source validation and lighting stress selection only."""
import copy,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import COLOURS
from carvision.illumination_colour import head
torch.set_num_threads(2);torch.manual_seed(240026);out=ROOT/'models/multimodal_v24'
assert not (out/'colour_training.json').exists(),'Do not repeat completed colour fit'
rows=list(torch.load(ROOT/'cache/illumination_v24.pt',weights_only=True)['items'].values())
def data(split):
    rs=[(r,k) for r in rows if r['split']==split for k in ('whole','warm','shadow')]
    return torch.stack([r['variants'][k]['x'] for r,k in rs]),torch.tensor([COLOURS.index(r['colour']) for r,k in rs]),torch.stack([r['variants'][k]['baseline'] for r,k in rs]),rs
x,y,_,train=data('train');vx,vy,baseline,val=data('validation');mean=x.mean(0);std=x.std(0).clamp_min(.01);x=(x-mean)/std;vx=(vx-mean)/std
normal=torch.tensor([k=='whole' for r,k in val]);stress=~normal;base_correct=baseline.argmax(1)==vy
network=head(len(mean));optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.02);counts=torch.bincount(y,minlength=8).float();cw=counts.rsqrt();cw/=cw.mean();sample_weight=torch.tensor([5. if r['owner'] else 1. for r,k in train]);best=-1;chosen=None;history=[]
for epoch in range(150):
    network.train()
    for ids in torch.randperm(len(x)).split(64):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(network(x[ids]),y[ids],weight=cw,reduction='none',label_smoothing=.02);loss=(loss*sample_weight[ids]).sum()/sample_weight[ids].sum();loss.backward();optimizer.step()
    network.eval()
    with torch.inference_mode():p=network(vx).softmax(1)
    candidates=[]
    for weight in (.25,.5,.75,1.):
        q=p*weight+baseline*(1-weight);good=q.argmax(1)==vy;whole=float(good[normal].float().mean());dark=float(good[stress].float().mean());score=.5*whole+.5*dark
        if whole>=float(base_correct[normal].float().mean())-.02 and dark>=float(base_correct[stress].float().mean())+.03:candidates.append((score,weight,whole,dark))
    if candidates:
        score,weight,whole,dark=max(candidates)
        if score>best:best=score;chosen=dict(head=copy.deepcopy(network.state_dict()),weight=weight,whole=whole,stress=dark,epoch=epoch+1)
    history.append(dict(epoch=epoch+1,selected_epoch=chosen['epoch'] if chosen else None))
    if epoch%25==0:print('COLOUR',epoch+1,'best',best,flush=True)
report=dict(selected=chosen is not None,epochs=150,baseline_whole=float(base_correct[normal].float().mean()),baseline_stress=float(base_correct[stress].float().mean()),history=history,test_used_for_selection=False)
if chosen:
    network.load_state_dict(chosen['head']);network.eval()
    with torch.inference_mode():logits=network(vx)
    temperature=min((.5,.75,1.,1.5,2.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,vy)))
    # Calibration changes the mixture, so validate the actual saved predictor.
    with torch.inference_mode():q=(logits/temperature).softmax(1)*chosen['weight']+baseline*(1-chosen['weight'])
    correct=q.argmax(1)==vy;whole=float(correct[normal].float().mean());dark=float(correct[stress].float().mean())
    calibrated_ok=whole>=report['baseline_whole']-.02 and dark>=report['baseline_stress']+.03
    report.update({k:v for k,v in chosen.items() if k!='head'});report.update(temperature=temperature,calibrated_whole=whole,calibrated_stress=dark,selected=calibrated_ok)
    if calibrated_ok:
        checkpoint=dict(origin='project-trained-illumination-paint-v24',mean=mean,std=std,head=chosen['head'],weight=chosen['weight'],temperature=temperature,threshold=.8,classes=COLOURS)
        torch.save(checkpoint,out/'illumination_colour.pt')
    else:report['rejection_reason']='Temperature-calibrated mixture failed the predeclared validation guard.'
(out/'colour_training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('COLOUR COMPLETE',json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
