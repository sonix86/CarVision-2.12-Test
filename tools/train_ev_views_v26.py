"""Two finite own-head fits: a cosine continuation and four learned visual modes."""
import copy,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead,MultiPrototypeMakeHead,embedding_features,load_make_head
from tools.cache_semantic_v25 import token
from tools.ev26_helpers import geometry,acceptance,calibrate,route,reference_bank,references,metrics

def main():
    torch.set_num_threads(3);torch.manual_seed(260027);out=ROOT/'models/ev_views_v26';policy=json.loads((out/'policy.json').read_text());assert policy['rounds_max']==2
    with (out/'training_once.json').open('x') as f:json.dump(dict(started=time.time()),f)
    dataset=ROOT/'cache/ev_views_v26_data.pt';data=torch.load(dataset,weights_only=True);brands=data['brands'];train=[r for r in data['rows'] if r['split']=='train'];val=[r for r in data['rows'] if r['split']=='validation'];features=torch.load(ROOT/'cache/ev_views_v26_features.pt',weights_only=True)['items']
    def matrix(rs):return torch.cat([torch.stack([embedding_features(features[token(r,k)],False) for r in rs]) for k in ('whole','phone')])
    x=matrix(train);q=matrix(val);y=torch.tensor([r['y'] for r in train]*2);vy=torch.tensor([r['y'] for r in val]*2);names=sorted({r['group'] for r in val});groups=torch.tensor([names.index(r['group']) for r in val]*2);valid=geometry(val)
    old=torch.load(ROOT/'models/fallback_v253/semantic_make.pt',weights_only=True);teacher=load_make_head(old)
    with torch.inference_mode():
        teacher_train=teacher(x).detach().clone();old_p=(teacher(q)/old['temperature']).softmax(1);old_ref=references(q,old['retrieval'],len(brands));old_route,old_accept=route(old_p,old_ref,old['thresholds'],old['retrieval']['fallback_thresholds'],brands,valid);baseline=metrics(old_route,old_accept,val,brands,policy['priority_makes']);raw_baseline=metrics(old_p,acceptance(old_p,old['thresholds'],brands,valid),val,brands,policy['priority_makes'])
        bank=reference_bank(train,features,brands);reference=references(q,bank,len(brands))
    sw=torch.tensor([5. if r.get('owner') else 2. if r['image'].startswith('ev_views_v26/') else 1. for r in train]*2);cw=torch.bincount(y,minlength=len(brands)).float().clamp_min(1).pow(-.2);cw/=cw.mean();rounds=[];best=None;start=time.time()
    for round_id in (1,2):
        if round_id==1:
            network=CosineMakeHead(old['head']['weight']);network.load_state_dict(old['head'])
        else:
            protos=[]
            for j in range(len(brands)):
                points=x[y==j];centres=[torch.nn.functional.normalize(old['head']['weight'][j],dim=0)]
                for _ in range(3):centres.append(points[((1-points@torch.stack(centres).T).amin(1)).argmax()].clone() if len(points) else centres[0].clone())
                c=torch.stack(centres)
                if len(points):
                    for _ in range(8):
                        assigned=(points@c.T).argmax(1)
                        for k in range(1,4):
                            if (assigned==k).any():c[k]=torch.nn.functional.normalize(points[assigned==k].mean(0),dim=0)
                protos.append(c)
            network=MultiPrototypeMakeHead(torch.stack(protos));network.bias.data.copy_(old['head']['bias']);network.log_scale.data.copy_(old['head']['log_scale'])
        initial=network.weight.detach().clone();opt=torch.optim.AdamW(network.parameters(),lr=.00005 if round_id==1 else .0001,weight_decay=.01);chosen=None;history=[]
        for epoch in range(policy['epochs_each']):
            network.train()
            for ids in torch.randperm(len(x)).split(128):
                opt.zero_grad();logits=network(x[ids]);ce=torch.nn.functional.cross_entropy(logits,y[ids],weight=cw,label_smoothing=.01,reduction='none');loss=(ce*sw[ids]).sum()/sw[ids].sum()+.2*torch.nn.functional.kl_div((logits/2).log_softmax(1),(teacher_train[ids]/2).softmax(1),reduction='batchmean')*4+.01*(network.weight-initial).square().mean();loss.backward();opt.step()
            network.eval()
            with torch.inference_mode():
                p=network(q).softmax(1);m=metrics(p,torch.zeros(len(q),dtype=torch.bool),val,brands,policy['priority_makes']);score=.45*m['whole']['macro']+.2*m['phone']['macro']+.35*m['whole']['priority_macro'];guard=all(m[k]['historical']>=raw_baseline[k]['historical']-.01 for k in m)
            history.append(dict(epoch=epoch+1,score=score,guard=guard,metrics=m))
            if guard and (chosen is None or score>chosen['score']):chosen=dict(head=copy.deepcopy(network.state_dict()),epoch=epoch+1,score=score)
            if epoch%20==0:print('FIT',round_id,'EPOCH',epoch+1,'SCORE',score,'GUARD',guard,flush=True)
        record=dict(round=round_id,head_kind='cosine' if round_id==1 else 'multi-prototype',selected_epoch=chosen['epoch'] if chosen else None,history=history)
        if chosen:
            network.load_state_dict(chosen['head'])
            with torch.inference_mode():
                logits=network(q);temp=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,vy)));p=(logits/temp).softmax(1);thresholds=calibrate(p,vy,groups,brands,valid,.9);primary=acceptance(p,thresholds,brands,valid);mixed=p*.5+reference*.5;fallback=calibrate(mixed,vy,groups,brands,valid,.95,primary);chosen_p,accepted=route(p,reference,thresholds,fallback,brands,valid);m=metrics(chosen_p,accepted,val,brands,policy['priority_makes']);score=.45*m['whole']['macro']+.2*m['phone']['macro']+.35*m['whole']['priority_macro']
            enabled={b:thresholds[b]<1 or fallback[b]<1 for b in policy['mandatory_makes']};checks=dict(historical=all(m[k]['historical']>=baseline[k]['historical']-.01 for k in m),whole_accuracy=m['whole']['top1']>=baseline['whole']['top1'],phone_accuracy=m['phone']['top1']>=baseline['phone']['top1'],priority_gain=m['whole']['priority_macro']>=baseline['whole']['priority_macro']+.01,precision=m['whole']['precision']>=.95,correct_confirmed=m['whole']['correct_confirmed']>=baseline['whole']['correct_confirmed'],mandatory_makes=all(enabled.values()));record.update(metrics=m,checks=checks,passed=all(checks.values()),temperature=temp,priority_enabled=enabled)
            if record['passed'] and (best is None or score>best['score']):best=dict(score=score,head=chosen['head'],head_kind=record['head_kind'],prototype_count=4,temperature=temp,thresholds=thresholds,fallback_thresholds=fallback,round=round_id,epoch=chosen['epoch'])
        rounds.append(record);(out/'training_progress.json').write_text(json.dumps(dict(rounds=rounds)),encoding='utf-8')
    report=dict(passed=best is not None,baseline=baseline,rounds=rounds,selected_round=best['round'] if best else None,selected_epoch=best['epoch'] if best else None,seconds=time.time()-start,policy=policy,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),test_used_for_selection=False)
    if best:
        ck=dict(old,origin='project-trained-semantic-make-v26',head=best['head'],head_kind=best['head_kind'],prototype_count=4,temperature=best['temperature'],thresholds=best['thresholds'],dataset_sha256=report['dataset_sha256']);ck['retrieval']=dict(bank,fallback_thresholds=best['fallback_thresholds']);torch.save(ck,out/'semantic_make.pt');report['candidate_sha256']=hashlib.sha256((out/'semantic_make.pt').read_bytes()).hexdigest()
    (out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(out/'validation_gate.json').write_text(json.dumps(dict(passed=best is not None),indent=2),encoding='utf-8');print('TWO FITS COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('rounds','policy')}),flush=True)

if __name__=='__main__':main()
