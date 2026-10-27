"""Two predeclared make-head fits; reject instead of weakening validation gates."""
import copy,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from tools.ev26_helpers import references,route,calibrate,acceptance,metrics
from tools.train_truth_v30 import valid

def main():
    torch.set_num_threads(3);out=ROOT/'models/phone_v32';policy=json.loads((out/'policy.json').read_text(encoding='utf-8'))
    with (out/'training_once.json').open('x') as f:json.dump(dict(started=time.time()),f)
    rows=json.loads((ROOT/'cache/truth_v31/rows.json').read_text(encoding='utf-8'));d=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True);own=torch.load(ROOT/'cache/phone_v32/features.pt',weights_only=True);ck=torch.load(ROOT/'models/truth_v31/scoped_candidate.pt',weights_only=True);brands=ck['brands'];n=len(rows);x=d['x'];y=torch.tensor([brands.index(r['brand']) for r in rows]*2);ti=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train']);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation']);train=torch.cat([ti,ti+n]);val=torch.cat([vi,vi+n]);vr=[dict(rows[int(i)],y=int(y[i]),historical=rows[int(i)].get('historical',False)) for i in vi];vg=torch.tensor([sorted({r['group'] for r in vr}).index(r['group']) for r in vr]*2);geom=torch.cat([valid(vr,k) for k in ('whole','phone')]);priority=['Zeekr','Leapmotor','Voyah','BYD','Chevrolet','Dongfeng']
    head=CosineMakeHead(torch.zeros(len(brands),1280)).eval();head.load_state_dict(ck['head'])
    def measured(state,temp,bank,thresholds,fallback):
        h=CosineMakeHead(torch.zeros(len(brands),1280)).eval();h.load_state_dict(state);p=(h(x[val])/temp).softmax(1);rp,ra=route(p,references(x[val],bank,len(brands)),thresholds,fallback,brands,geom);return metrics(rp,ra,vr,brands,priority),metrics(p,torch.zeros(len(val),dtype=torch.bool),vr,brands,priority)
    with torch.no_grad():baseline,rawbase=measured(ck['head'],ck['temperature'],ck['retrieval'],ck['thresholds'],ck['retrieval']['fallback_thresholds']);teacher=head(x[train]).detach()
    tx=torch.cat([x[train],own['x']]);ty=torch.cat([y[train],own['y']]);private=torch.arange(len(tx))>=len(train);records=[];best=None
    # One new source group per outing; augmentations never create independent votes.
    bank=copy.deepcopy(ck['retrieval']);truth=json.loads((out/'truth.json').read_text(encoding='utf-8'));groupbase=int(bank['groups'].max())+1;gnames=sorted({r['group'] for r in truth});whole=torch.stack([own['x'][j*6] for j in range(len(truth))]);phone=torch.stack([own['x'][j*6+1] for j in range(len(truth))]);bank['keys']=torch.cat([bank['keys'],torch.stack([whole,phone])],1);bank['labels']=torch.cat([bank['labels'],own['y'][::6]]);bank['groups']=torch.cat([bank['groups'],torch.tensor([groupbase+gnames.index(r['group']) for r in truth])]);ref=references(x[val],bank,len(brands))
    for rid,config in enumerate(policy['configs'],1):
        torch.manual_seed(320000+rid);h=CosineMakeHead(torch.zeros(len(brands),1280));h.load_state_dict(ck['head']);opt=torch.optim.AdamW(h.parameters(),lr=config['lr'],weight_decay=.01);sw=torch.where(private,torch.tensor(float(config['own_weight'])),torch.tensor(1.));history=[];chosen=None
        for epoch in range(policy['epochs']):
            h.train()
            for ids in torch.randperm(len(tx)).split(128):
                opt.zero_grad();z=h(tx[ids]);loss=torch.nn.functional.cross_entropy(z,ty[ids],label_smoothing=.01,reduction='none');loss=(loss*sw[ids]).sum()/sw[ids].sum();pub=~private[ids]
                if pub.any():loss+=4*torch.nn.functional.kl_div((z[pub]/2).log_softmax(1),(teacher[ids[pub]]/2).softmax(1),reduction='batchmean')
                loss.backward();opt.step()
            h.eval()
            with torch.no_grad():
                logits=h(x[val]);vp=logits.softmax(1);raw=metrics(vp,torch.zeros(len(val),dtype=torch.bool),vr,brands,priority);fit=float((h(own['x']).argmax(1)==own['y']).float().mean());guard=fit==1 and all(raw[k]['top1']>=rawbase[k]['top1']-.005 for k in raw);score=sum(raw[k]['macro'] for k in raw)/2
            history.append(dict(epoch=epoch+1,fit=fit,guard=guard,raw=raw))
            if guard and (chosen is None or score>chosen['score']):chosen=dict(head=copy.deepcopy(h.state_dict()),epoch=epoch+1,score=score,train_fit=fit)
            if epoch%20==0:print('PHONE32 FIT',rid,epoch+1,fit,score,guard,flush=True)
        record=dict(round=rid,history=history,passed=False)
        if chosen:
            h.load_state_dict(chosen['head'])
            with torch.no_grad():
                logits=h(x[val]);temp=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,y[val])));p=(logits/temp).softmax(1);thresholds=calibrate(p,y[val],vg,brands,geom,.95);primary=acceptance(p,thresholds,brands,geom);fallback=calibrate(p*.5+ref*.5,y[val],vg,brands,geom,.95,primary);rp,ra=route(p,ref,thresholds,fallback,brands,geom);m=metrics(rp,ra,vr,brands,priority);op=(h(own['x'])/temp).softmax(1);og=torch.tensor([min(v['size'])>=40 for v in own['meta']]);oq,oa=route(op,references(own['x'],bank,len(brands)),thresholds,fallback,brands,og);wholeids=torch.tensor([i for i,v in enumerate(own['meta']) if v['kind']=='whole']);known_ok=bool((oa[wholeids]&(oq[wholeids].argmax(1)==own['y'][wholeids])).all())
            checks=dict(known_train_originals_correct=known_ok,precision=all(m[k]['precision']>=baseline[k]['precision']-.002 for k in m),confirmed=all(m[k]['correct_confirmed']>=baseline[k]['correct_confirmed'] for k in m),top1=all(m[k]['top1']>=baseline[k]['top1']-.005 for k in m));record.update(passed=all(checks.values()),checks=checks,metrics=m,epoch=chosen['epoch'],known_train_accepted=[dict(brand=own['meta'][int(i)]['brand'],accepted=bool(oa[i]),candidate=brands[int(oq[i].argmax())],confidence=float(oq[i].max())) for i in wholeids]);candidate=dict(ck,head=chosen['head'],temperature=temp,thresholds=thresholds,retrieval=dict(bank,fallback_thresholds=fallback));candidate['phone_v32']=dict(role='Known supplied errors now TRAIN, not independent accuracy',round=rid,epoch=chosen['epoch']);torch.save(candidate,out/f'round{rid}.pt')
            if record['passed'] and (best is None or chosen['score']>best['score']):best=dict(score=chosen['score'],round=rid)
        records.append(record);(out/'validation.json').write_text(json.dumps(dict(passed=best is not None,baseline=baseline,raw_baseline=rawbase,rounds=records,selected=best,test_used=False),indent=2),encoding='utf-8')
    if best:
        import shutil
        shutil.copy2(out/f"round{best['round']}.pt",out/'candidate.pt')
    else:(out/'NO_CHANGE.txt').write_text('Both predeclared supervised candidates failed validation. Do not deploy or lower gates.\n',encoding='utf-8')
    print('PHONE32 COMPLETE',best,flush=True)
if __name__=='__main__':main()
