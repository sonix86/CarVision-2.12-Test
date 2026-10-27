"""Two bounded supervised heads over general image features, validation only."""
import copy,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead,embedding_features,load_make_head
from tools.cache_semantic_v25 import token
from tools.ev26_helpers import geometry,calibrate,acceptance,references,route,metrics

def bank(z,rows):
    names=sorted({r['group'] for r in rows})
    return dict(keys=z.reshape(2,len(rows),-1),labels=torch.tensor([r['y'] for r in rows]),groups=torch.tensor([names.index(r['group']) for r in rows]),k=8,alpha=.5,beta=50.,mode='abstention-only')

def main():
    torch.set_num_threads(3);out=ROOT/'models/dino_v28';policy=json.loads((out/'policy.json').read_text());assert policy['rounds_max']==2
    assert json.loads((out/'features.json').read_text())['complete']
    with (out/'training_once.json').open('x') as f:json.dump(dict(started=time.time()),f)
    data=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True);brands=data['brands'];train=[r for r in data['rows'] if r['split']=='train'];val=[r for r in data['rows'] if r['split']=='validation'];assert {r['group'] for r in train}.isdisjoint({r['group'] for r in val})
    df=torch.load(ROOT/'cache/dino_v28/features.pt',weights_only=True)['items'];cf=torch.load(ROOT/'cache/ev_views_v26_features.pt',weights_only=True)['items']
    def matrix(rs,features,clip=False):
        return torch.cat([torch.stack([embedding_features(features[token(r,k)],False) if clip else features[token(r,k)] for r in rs]) for k in ('whole','phone')])
    dx=matrix(train,df);dq=matrix(val,df);cx=matrix(train,cf,True);cq=matrix(val,cf,True)
    old=torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True);valid=geometry(val)
    y=torch.tensor([r['y'] for r in train]*2);vy=torch.tensor([r['y'] for r in val]*2);names=sorted({r['group'] for r in val});groups=torch.tensor([names.index(r['group']) for r in val]*2)
    with torch.inference_mode():
        p=(load_make_head(old)(cq)/old['temperature']).softmax(1);ref=references(cq,old['retrieval'],len(brands));rp,ra=route(p,ref,old['thresholds'],old['retrieval']['fallback_thresholds'],brands,valid);baseline=metrics(rp,ra,val,brands,policy['priority_makes']);raw_base=metrics(p,torch.zeros(len(vy),dtype=torch.bool),val,brands,policy['priority_makes'])
    sw=torch.tensor([5. if r.get('owner') else 1. for r in train]*2);cw=torch.bincount(y,minlength=len(brands)).float().clamp_min(1).pow(-.2);cw/=cw.mean();rounds=[];best=None;start=time.time()
    for rid,kind in enumerate(('dino','joint'),1):
        torch.manual_seed(280000+rid);x=dx if kind=='dino' else torch.nn.functional.normalize(torch.cat([dx,cx],1),dim=1);q=dq if kind=='dino' else torch.nn.functional.normalize(torch.cat([dq,cq],1),dim=1)
        initial=torch.stack([torch.nn.functional.normalize(x[y==j].mean(0),dim=0) if (y==j).any() else torch.zeros(x.shape[1]) for j in range(len(brands))]);head=CosineMakeHead(initial);opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.02);chosen=None;history=[]
        for epoch in range(policy['epochs_each']):
            head.train()
            for ids in torch.randperm(len(x)).split(128):
                opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[ids]),y[ids],weight=cw,label_smoothing=.02,reduction='none');loss=(loss*sw[ids]).sum()/sw[ids].sum();loss.backward();opt.step()
            head.eval()
            with torch.inference_mode():
                p=head(q).softmax(1);m=metrics(p,torch.zeros(len(q),dtype=torch.bool),val,brands,policy['priority_makes']);score=.45*m['whole']['macro']+.2*m['phone']['macro']+.35*m['whole']['priority_macro'];guard=all(m[k]['historical']>=raw_base[k]['historical']-.01 for k in m)
            history.append(dict(epoch=epoch+1,score=score,guard=guard,metrics=m))
            if guard and (chosen is None or score>chosen['score']):chosen=dict(head=copy.deepcopy(head.state_dict()),score=score,epoch=epoch+1)
            if epoch%25==0:print('DINO FIT',rid,'EPOCH',epoch+1,'SCORE',score,'GUARD',guard,flush=True)
        record=dict(round=rid,kind=kind,history=history,passed=False)
        if chosen:
            head.load_state_dict(chosen['head'])
            with torch.inference_mode():
                logits=head(q);temp=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,vy)));p=(logits/temp).softmax(1);thresholds=calibrate(p,vy,groups,brands,valid,.95);b=bank(x,train);ref=references(q,b,len(brands));primary=acceptance(p,thresholds,brands,valid);fallback=calibrate(p*.5+ref*.5,vy,groups,brands,valid,.95,primary);rp,ra=route(p,ref,thresholds,fallback,brands,valid);m=metrics(rp,ra,val,brands,policy['priority_makes'])
            checks=dict(whole=m['whole']['top1']>=baseline['whole']['top1'],phone=m['phone']['top1']>=baseline['phone']['top1'],historical=all(m[k]['historical']>=baseline[k]['historical']-.01 for k in m),priority=m['whole']['priority_macro']>=baseline['whole']['priority_macro']+.01,precision_both=all(m[k]['precision']>=max(.95,baseline[k]['precision']-.01) for k in m),correct_confirmed_both=all(m[k]['correct_confirmed']>=baseline[k]['correct_confirmed'] for k in m),mandatory=all(thresholds[k]<1 or fallback[k]<1 for k in policy['mandatory_makes']))
            record.update(selected_epoch=chosen['epoch'],metrics=m,checks=checks,passed=all(checks.values()))
            if record['passed'] and (best is None or chosen['score']>best['score']):b['fallback_thresholds']=fallback;best=dict(chosen,kind=kind,round=rid,temperature=temp,thresholds=thresholds,retrieval=b)
        rounds.append(record);(out/'training_progress.json').write_text(json.dumps(dict(rounds=rounds)),encoding='utf-8')
    if best:
        ck=dict(origin='project-trained-dino-make-v28',brands=brands,kind=best['kind'],head=best['head'],temperature=best['temperature'],thresholds=best['thresholds'],margin=.06,retrieval=best['retrieval'],encoder_state=torch.load(ROOT/'cache/dino_v28/dinov2_vits14_pretrain.pth',weights_only=True),clip_checkpoint_sha256=hashlib.sha256((ROOT/'models/ev_views_v26/semantic_make.pt').read_bytes()).hexdigest())
        torch.save(ck,out/'dino_make.pt')
    report=dict(passed=best is not None,baseline=baseline,rounds=rounds,selected_round=best['round'] if best else None,selected_epoch=best['epoch'] if best else None,seconds=time.time()-start,test_used_for_selection=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if best is None:(out/'NO_CHANGE.txt').write_text('Both bounded DINO supervised candidates failed validation. No test extraction and no production replacement.\n',encoding='utf-8')
    print('DINO TWO FITS COMPLETE',bool(best),flush=True)

if __name__=='__main__':main()
