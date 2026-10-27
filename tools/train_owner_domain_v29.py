"""Two bounded head fits after verified owner-data/crop corrections."""
import copy,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead,embedding_features,load_make_head
from tools.cache_semantic_v25 import token
from tools.ev26_helpers import geometry,acceptance,calibrate,references,route,metrics
from tools.train_dino_v28 import bank

def main():
    torch.set_num_threads(3);out=ROOT/'models/owner_domain_v29';cache=ROOT/'cache/owner_domain_v29';policy=json.loads((out/'policy.json').read_text());assert policy['rounds_max']==2
    assert json.loads((out/'features.json').read_text())['complete']
    with (out/'training_once.json').open('x') as f:json.dump(dict(started=time.time()),f)
    data=torch.load(cache/'data.pt',weights_only=True);brands=data['brands'];train=[r for r in data['rows'] if r['split']=='train'];val=[r for r in data['rows'] if r['split']=='validation']
    cf=torch.load(ROOT/'cache/ev_views_v26_features.pt',weights_only=True)['items'];df=torch.load(ROOT/'cache/dino_v28/features.pt',weights_only=True)['items'];own=torch.load(cache/'features.pt',weights_only=True);cf.update(own['clip']);df.update(own['dino'])
    entries=[(r,k) for k in ('whole','phone') for r in train]+[(r,k) for r in train if r.get('domain_v29') for k in ('flip','cast')];ve=[(r,k) for k in ('whole','phone') for r in val]
    def matrix(entries,kind):
        c=torch.stack([embedding_features(cf[token(r,k)],False) for r,k in entries])
        if kind=='clip':return c
        d=torch.stack([df[token(r,k)] for r,k in entries]);return torch.nn.functional.normalize(torch.cat([d,c],1),dim=1)
    y=torch.tensor([r['y'] for r,k in entries]);vy=torch.tensor([r['y'] for r,k in ve]);private=torch.tensor([bool(r.get('owner')) for r,k in entries]);domain=torch.tensor([bool(r.get('domain_v29')) for r,k in entries]);valid=geometry(val);names=sorted({r['group'] for r in val});groups=torch.tensor([names.index(r['group']) for r,k in ve]);sw=torch.where(private,torch.tensor(12.),torch.tensor(1.));cw=torch.bincount(y,minlength=len(brands)).float().clamp_min(1).pow(-.2);cw/=cw.mean()
    old=torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True);qclip=matrix(ve,'clip')
    with torch.inference_mode():
        bp=(load_make_head(old)(qclip)/old['temperature']).softmax(1);ref=references(qclip,old['retrieval'],len(brands));p,a=route(bp,ref,old['thresholds'],old['retrieval']['fallback_thresholds'],brands,valid);baseline=metrics(p,a,val,brands,policy['priority_makes']);raw_base=metrics(bp,torch.zeros(len(vy),dtype=torch.bool),val,brands,policy['priority_makes'])
    rounds=[];best=None;start=time.time()
    for rid,kind in enumerate(('clip','joint'),1):
        torch.manual_seed(290000+rid);initial=old if kind=='clip' else torch.load(ROOT/'models/dino_v28/dino_make.pt',weights_only=True);x=matrix(entries,kind);q=matrix(ve,kind);head=CosineMakeHead(torch.zeros(len(brands),x.shape[1]));head.load_state_dict(initial['head']);head.eval()
        with torch.no_grad():teacher=head(x).detach().clone()
        opt=torch.optim.AdamW(head.parameters(),lr=.0002 if kind=='clip' else .0001,weight_decay=.02);chosen=None;history=[]
        for epoch in range(policy['epochs_each']):
            head.train()
            for ids in torch.randperm(len(x)).split(128):
                opt.zero_grad();logits=head(x[ids]);loss=torch.nn.functional.cross_entropy(logits,y[ids],weight=cw,label_smoothing=.01,reduction='none');loss=(loss*sw[ids]).sum()/sw[ids].sum();pub=~private[ids]
                if pub.any():loss=loss+.2*torch.nn.functional.kl_div((logits[pub]/2).log_softmax(1),(teacher[ids][pub]/2).softmax(1),reduction='batchmean')*4
                loss.backward();opt.step()
            head.eval()
            with torch.inference_mode():
                p=head(q).softmax(1);m=metrics(p,torch.zeros(len(vy),dtype=torch.bool),val,brands,policy['priority_makes']);fit=float((head(x[domain]).argmax(1)==y[domain]).float().mean());guard=fit>=.9 and all(m[k]['historical']>=raw_base[k]['historical']-.01 for k in m);score=.45*m['whole']['macro']+.2*m['phone']['macro']+.35*m['whole']['priority_macro']
            history.append(dict(epoch=epoch+1,score=score,guard=guard,private_training_fit=fit,metrics=m))
            if guard and (chosen is None or score>chosen['score']):chosen=dict(head=copy.deepcopy(head.state_dict()),epoch=epoch+1,score=score,private_training_fit=fit)
            if epoch%25==0:print('OWNER FIT',rid,'EPOCH',epoch+1,'TRAIN FIT',fit,'VAL SCORE',score,'GUARD',guard,flush=True)
        record=dict(round=rid,kind=kind,history=history,passed=False)
        if chosen:
            head.load_state_dict(chosen['head'])
            with torch.inference_mode():
                logits=head(q);temp=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,vy)));p=(logits/temp).softmax(1);thresholds=calibrate(p,vy,groups,brands,valid,.95);b=bank(x[:2*len(train)],train);ref=references(q,b,len(brands));primary=acceptance(p,thresholds,brands,valid);fallback=calibrate(p*.5+ref*.5,vy,groups,brands,valid,.95,primary);rp,ra=route(p,ref,thresholds,fallback,brands,valid);m=metrics(rp,ra,val,brands,policy['priority_makes'])
            checks=dict(whole=m['whole']['top1']>=baseline['whole']['top1'],phone=m['phone']['top1']>=baseline['phone']['top1'],historical=all(m[k]['historical']>=baseline[k]['historical']-.01 for k in m),priority=m['whole']['priority_macro']>=baseline['whole']['priority_macro']+.01,precision_both=all(m[k]['precision']>=max(.95,baseline[k]['precision']-.01) for k in m),correct_confirmed_both=all(m[k]['correct_confirmed']>=baseline[k]['correct_confirmed'] for k in m),mandatory=all(thresholds[k]<1 or fallback[k]<1 for k in policy['mandatory_makes']))
            record.update(selected_epoch=chosen['epoch'],private_training_fit=chosen['private_training_fit'],metrics=m,checks=checks,passed=all(checks.values()))
            if record['passed'] and (best is None or chosen['score']>best['score']):b['fallback_thresholds']=fallback;best=dict(chosen,kind=kind,round=rid,temperature=temp,thresholds=thresholds,retrieval=b,initial=initial)
        rounds.append(record);(out/'training_progress.json').write_text(json.dumps(dict(rounds=rounds)),encoding='utf-8')
    if best:
        ck=dict(best['initial'],head=best['head'],temperature=best['temperature'],thresholds=best['thresholds'],retrieval=best['retrieval']);ck['owner_domain_v29']=dict(dataset_sha256=json.loads((out/'dataset.json').read_text())['dataset_sha256'],private_training_fit=best['private_training_fit'],role='Own training examples; not independent phone evaluation',round=best['round'],epoch=best['epoch'])
        name='semantic_make.pt' if best['kind']=='clip' else 'dino_make.pt';torch.save(ck,out/name);(out/'selection.json').write_text(json.dumps(dict(kind=best['kind'],checkpoint=name,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    report=dict(passed=best is not None,baseline=baseline,rounds=rounds,selected_round=best['round'] if best else None,selected_epoch=best['epoch'] if best else None,test_used=False,seconds=time.time()-start,known_phone_errors_now_train=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if best is None:(out/'NO_CHANGE.txt').write_text('Both bounded own-data correction candidates failed validation. Do not extract/evaluate a new test or change production weights.\n',encoding='utf-8')
    print('TWO OWNER FITS COMPLETE',best is not None,flush=True)

if __name__=='__main__':main()
