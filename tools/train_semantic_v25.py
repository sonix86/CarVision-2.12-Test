"""Two supervised semantic fits; fixed validation policy and no test access."""
import copy,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead,embedding_features
from tools.cache_semantic_v25 import token

def main():
    torch.set_num_threads(3);torch.manual_seed(250026);out=ROOT/'models/semantic_v25';assert not (out/'training.json').exists(),'Do not repeat the two finite fits'
    with (out/'training_once.json').open('x') as f:json.dump(dict(started=time.time()),f)
    policy=json.loads((out/'policy.json').read_text(encoding='utf-8'));assert policy['rounds_max']==2
    dataset=ROOT/'cache/multimodal_v24_data.pt';data=torch.load(dataset,weights_only=True);brands=data['brands'];rows=data['rows'];train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    cache=torch.load(ROOT/'cache/semantic_v25/features.pt',weights_only=True);initialization=torch.load(ROOT/'cache/semantic_v25/initialization.pt',weights_only=True);assert brands==initialization['brands']
    legacy=json.loads((ROOT/'models/make_scope_v23/training.json').read_text(encoding='utf-8'))['validation_baseline_body_head'];y=torch.tensor([r['y'] for r in train]*2);vy=torch.tensor([r['y'] for r in val]);hist=torch.tensor([r['historical'] for r in val]);counts=torch.bincount(y,minlength=len(brands)).float().clamp_min(1);cw=counts.pow(-.2);cw/=cw.mean();sw=torch.tensor([3. if r.get('owner') else 1. for r in train]*2)
    rounds=[];best=None;started=time.time()
    for round_id,two in enumerate((False,True),1):
        def matrix(rs,kind):return embedding_features(torch.stack([cache['items'][token(r,kind)] for r in rs]),two)
        x=torch.cat([matrix(train,k) for k in ('whole','phone')]);vx={k:matrix(val,k) for k in ('whole','phone')}
        prototype=initialization['prototypes'];prototype=torch.nn.functional.normalize(torch.cat([prototype*.7,prototype*.3],1),dim=1) if two else prototype
        network=CosineMakeHead(prototype);optimizer=torch.optim.AdamW(network.parameters(),lr=.0001,weight_decay=.01);history=[];chosen=None
        for epoch in range(80):
            network.train()
            for ids in torch.randperm(len(x)).split(128):
                optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(network(x[ids]),y[ids],weight=cw,label_smoothing=.01,reduction='none');loss=(loss*sw[ids]).sum()/sw[ids].sum()+.1*(network.weight-prototype).square().mean();loss.backward();optimizer.step()
            network.eval();metrics={}
            with torch.inference_mode():
                for k in vx:
                    p=network(vx[k]).softmax(1);good=p.argmax(1)==vy
                    metrics[k]=dict(top1=float(good.float().mean()),historical=float(good[hist].float().mean()),macro=float(torch.tensor([float(good[vy==j].float().mean()) for j in vy.unique()]).mean()))
            with torch.inference_mode():
                good=network(vx['whole']).argmax(1)==vy;priority_ids=[brands.index(b) for b in policy['priority_makes']];priority_macro=float(torch.tensor([float(good[vy==j].float().mean()) for j in priority_ids if (vy==j).any()]).mean())
            guard=all(metrics[k]['historical']>=legacy[k]['top1']-.02 for k in vx);score=.5*metrics['whole']['macro']+.2*metrics['phone']['macro']+.3*priority_macro;record=dict(epoch=epoch+1,guard=guard,score=score,priority_macro=priority_macro,metrics=metrics);history.append(record)
            if guard and (chosen is None or score>chosen['score']):chosen=dict(head=copy.deepcopy(network.state_dict()),score=score,epoch=epoch+1,metrics=metrics)
            if epoch%20==0:print('SEMANTIC ROUND',round_id,json.dumps(record),flush=True)
        rounds.append(dict(round=round_id,two_views=two,selected_epoch=chosen['epoch'] if chosen else None,history=history))
        if chosen and (best is None or chosen['score']>best['score']):best=dict(chosen,two_views=two,prototype=prototype,vx=vx,round=round_id)
        (out/'training_progress.json').write_text(json.dumps(dict(rounds=rounds)),encoding='utf-8')
    assert best is not None,'No historical-validation-safe semantic candidate. Stop without replacement.'
    network=CosineMakeHead(best['prototype']).eval();network.load_state_dict(best['head'])
    with torch.inference_mode():logits=torch.cat([network(best['vx'][k]) for k in ('whole','phone')])
    labels=vy.repeat(2);temperature=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,labels)));p=(logits/temperature).softmax(1);values,ids=p.topk(2,dim=1)
    group_names=sorted({r['group'] for r in val});groups=torch.tensor([group_names.index(r['group']) for r in val]*2);geometry=[]
    for kind in ('whole','phone'):
        for r in val:
            h,w=r['body_rgb'].shape[:2];scale=min(1,96/max(w,h)) if kind=='phone' else 1;geometry.append(min(round(w*scale),round(h*scale))>=40)
    geometry=torch.tensor(geometry);thresholds={};calibration={};accepted=torch.zeros(len(p),dtype=torch.bool)
    for j,b in enumerate(brands):
        options=[]
        for gate in (.5,.6,.7,.8,.9,.95,.98):
            eligible=(ids[:,0]==j)&(values[:,0]>=gate)&((values[:,0]-values[:,1])>=.06)&geometry
            if len(groups[eligible].unique())>=3 and float((labels[eligible]==j).float().mean())>=.9:options.append((int(eligible.sum()),gate))
        _,gate=max(options,default=(0,1.));thresholds[b]=gate;eligible=(ids[:,0]==j)&(values[:,0]>=gate)&((values[:,0]-values[:,1])>=.06)&geometry if gate<1 else torch.zeros(len(p),dtype=torch.bool);accepted|=eligible;calibration[b]=dict(threshold=gate,views=int(eligible.sum()),groups=len(groups[eligible].unique()))
    encoder=torch.load(ROOT/'cache/semantic_v25/image_encoder.pt',weights_only=True)
    assert not torch.equal(best['head']['weight'],best['prototype']),'Do not deploy an untrained text-prototype classifier'
    checkpoint=dict(origin='project-trained-semantic-make-v25',brands=brands,two_views=best['two_views'],encoder_state=encoder['state'],head=best['head'],temperature=temperature,thresholds=thresholds,margin=.06,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),encoder_source_sha256=encoder['source_sha256'])
    torch.save(checkpoint,out/'semantic_make.pt')
    priority={b:thresholds[b]<1 for b in policy['priority_makes']};report=dict(selected_round=best['round'],selected_epoch=best['epoch'],selected_metrics=best['metrics'],temperature=temperature,rounds=rounds,calibration=calibration,priority_enabled=priority,accepted_validation_views=int(accepted.sum()),accepted_validation_precision=float((ids[accepted,0]==labels[accepted]).float().mean()) if accepted.any() else None,confirmed_make_count=sum(v<1 for v in thresholds.values()),dataset_sha256=checkpoint['dataset_sha256'],encoder_source_sha256=encoder['source_sha256'],candidate_sha256=hashlib.sha256((out/'semantic_make.pt').read_bytes()).hexdigest(),seconds=time.time()-started,test_used_for_selection=False,policy=policy)
    (out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('SEMANTIC TRAINING COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('rounds','calibration','policy')}),flush=True)

if __name__=='__main__':main()
