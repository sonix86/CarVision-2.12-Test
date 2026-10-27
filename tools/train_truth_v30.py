"""Bounded independent make/family fits; no classifier-generated truth."""
import copy,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from tools.ev26_helpers import calibrate,acceptance,references,route,metrics
from tools.train_dino_v28 import bank

def evidence_correct(probabilities,truth):
    """An all-zero/invalid abstention never matches class0 by argmax default."""
    return torch.isfinite(probabilities).all(1)&(probabilities.sum(1)>0)&(probabilities.argmax(1)==truth)

def valid(rows,kind='whole',family=False):
    out=[]
    for r in rows:
        h,w=r['shape'][:2];scale=min(1,96/max(h,w)) if kind=='phone' else 1
        ok=min(round(w*scale),round(h*scale))>=(64 if family else 40)
        if family:ok=ok and r['mask_pixels']*scale**2>=1200 and r['mask_fraction']>=.35 and w/h<=3.2
        out.append(ok)
    return torch.tensor(out)

def main():
    torch.set_num_threads(3);out=ROOT/'models/truth_v30';cache=ROOT/'cache/truth_v30';policy=json.loads((out/'policy.json').read_text());assert (cache/'legacy_model_validation.pt').exists() and (cache/'edge_validation.pt').exists()
    with (out/'training_once.json').open('x') as f:json.dump(dict(started=time.time(),policy=policy),f)
    rows=json.loads((cache/'all_rows.json').read_text(encoding='utf-8'));d=torch.load(cache/'joint.pt',weights_only=True);extra=torch.load(cache/'e70_features.pt',weights_only=True)['x'];e=len(extra)//2;n=len(rows);oldn=n-e;x=torch.cat([d['x'][:oldn],extra[:e],d['x'][oldn:],extra[e:]]);brands=d['brands'];pairs=[tuple(p) for p in d['pairs']]+[('Dongfeng','Aeolus E70')];keys=[b+'|'+m for b,m in pairs];modelbrands=torch.tensor([brands.index(b) for b,m in pairs]);priority=['BYD','Changan','Deepal','Dongfeng','GAC Aion','Geely','Geometry','Haval','Honda','Leapmotor','Li Auto','Neta','Nio','Toyota','Voyah','Weltmeister','XPeng','Zeekr']
    for r in rows:r['y']=brands.index(r['brand']);r.setdefault('historical',False)
    ti=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train']);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation']);train=torch.cat([ti,ti+n]);val=torch.cat([vi,vi+n]);tr=[rows[i] for i in ti];vr=[rows[i] for i in vi];y=torch.tensor([r['y'] for r in rows]*2);vy=y[val];vg=torch.tensor([sorted({r['group'] for r in vr}).index(r['group']) for r in vr]*2);vgood=torch.cat([valid(vr,k) for k in ('whole','phone')]);old=torch.load(ROOT/'models/owner_domain_v29/dino_make.pt',weights_only=True);oldhead=CosineMakeHead(torch.zeros(len(brands),1280)).eval();oldhead.load_state_dict(old['head'])
    def makepred(ck,z,good):
        h=CosineMakeHead(torch.zeros(len(brands),1280)).eval();h.load_state_dict(ck['head']);p=(h(z)/ck['temperature']).softmax(1);return route(p,references(z,ck['retrieval'],len(brands)),ck['thresholds'],ck['retrieval']['fallback_thresholds'],brands,good)
    with torch.inference_mode():bp,ba=makepred(old,x[val],vgood);bm=metrics(bp,ba,vr,brands,priority);teacher=oldhead(x[train]).detach()
    rounds=[];best=None
    for rid,config in enumerate(policy['make_rounds'],1):
        torch.manual_seed(300000+rid);head=CosineMakeHead(torch.zeros(len(brands),1280));head.load_state_dict(old['head']);opt=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=.02);private=torch.tensor([bool(rows[int(i)%n].get('owner')) for i in train]);isnew=torch.tensor([int(i)%n>=oldn for i in train]);sw=torch.where(private,torch.tensor(8.),torch.tensor(1.));sw[isnew]=config['e70_weight'];cw=torch.bincount(y[train],minlength=len(brands)).float().clamp_min(1).pow(-.2);cw/=cw.mean();chosen=None
        for epoch in range(policy['epochs_each']):
            head.train()
            for local in torch.randperm(len(train)).split(128):
                ids=train[local];opt.zero_grad();z=head(x[ids]);loss=torch.nn.functional.cross_entropy(z,y[ids],weight=cw,label_smoothing=.01,reduction='none');loss=(loss*sw[local]).sum()/sw[local].sum();distill=~(private[local]|isnew[local])
                if distill.any():loss=loss+.15*4*torch.nn.functional.kl_div((z[distill]/2).log_softmax(1),(teacher[local][distill]/2).softmax(1),reduction='batchmean')
                loss.backward();opt.step()
            head.eval()
            with torch.inference_mode():p=head(x[val]).softmax(1);m=metrics(p,torch.zeros(len(val),dtype=torch.bool),vr,brands,priority);guard=all(m[k]['top1']>=bm[k]['top1'] and m[k]['historical']>=bm[k]['historical']-.01 for k in m);score=.45*m['whole']['macro']+.2*m['phone']['macro']+.35*m['whole']['priority_macro']
            if guard and (chosen is None or score>chosen['score']):chosen=dict(state=copy.deepcopy(head.state_dict()),epoch=epoch+1,score=score)
        record=dict(round=rid,selected_epoch=chosen['epoch'] if chosen else None,passed=False)
        if chosen:
            head.load_state_dict(chosen['state']);head.eval()
            with torch.inference_mode():
                z=head(x[val]);temperature=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(z/t,vy)));p=(z/temperature).softmax(1);thresholds=calibrate(p,vy,vg,brands,vgood,.95);b=bank(x[train],tr);ref=references(x[val],b,len(brands));fallback=calibrate(p*.5+ref*.5,vy,vg,brands,vgood,.95,acceptance(p,thresholds,brands,vgood));b['fallback_thresholds']=fallback;ck=dict(old,head=chosen['state'],temperature=temperature,thresholds=thresholds,retrieval=b);rp,ra=makepred(ck,x[val],vgood);m=metrics(rp,ra,vr,brands,priority)
            checks=dict(top1=all(m[k]['top1']>=bm[k]['top1'] for k in m),historical=all(m[k]['historical']>=bm[k]['historical']-.01 for k in m),precision=all(m[k]['precision']>=max(.95,bm[k]['precision']-.01) for k in m),correct_confirmed=all(m[k]['correct_confirmed']>=bm[k]['correct_confirmed'] for k in m),dongfeng_enabled=thresholds['Dongfeng']<1 or fallback['Dongfeng']<1)
            record.update(metrics=m,checks=checks,passed=all(checks.values()))
            if record['passed'] and (best is None or chosen['score']>best['score']):best=dict(score=chosen['score'],checkpoint=ck,round=rid,epoch=chosen['epoch'])
        rounds.append(record);print('MAKE ROUND',json.dumps(record),flush=True)
    selected=best['checkpoint'] if best else old;make_status=dict(promoted=best is not None,baseline=bm,rounds=rounds)
    # Family truth is independent source labels, with provisional make-only rows excluded.
    known=lambda r:(r['brand'],r.get('family')) in pairs and not r['image'].startswith('make_scope_v23/')
    fti=torch.tensor([i for i in ti if known(rows[int(i)])]);ftrain=torch.cat([fti,fti+n]);fvi=torch.tensor([i for i in vi if known(rows[int(i)])]);fr=[rows[int(i)] for i in fvi];fv=torch.cat([fvi,fvi+n]);fy=torch.tensor([pairs.index((r['brand'],r.get('family'))) if known(r) else -1 for r in rows]*2);truth=fy[fv];legacy=torch.load(cache/'legacy_model_validation.pt',weights_only=True);assert legacy['selected']==fvi.tolist();edge=torch.load(cache/'edge_validation.pt',weights_only=True);assert edge['indices']==fvi.tolist();q=torch.cat([x[fv],edge['x']]);fvalid=torch.cat([valid(fr,'whole',True),valid(fr,'phone',True),edge['geometry']]);groups=torch.tensor([sorted({r['group'] for r in fr}).index(r['group']) for r in fr]*3);ft=torch.cat([truth,truth[:len(fr)]]);ftruth=truth[:len(fr)]
    with torch.inference_mode():mp,ma=makepred(selected,q,torch.cat([valid(fr),valid(fr,'phone'),torch.ones(len(fr),dtype=torch.bool)]));op,oa=makepred(old,x[fv],torch.cat([valid(fr),valid(fr,'phone')]))
    def bind(p,pm,am):return p*(modelbrands[None,:]==pm.argmax(1)[:,None])*am[:,None]
    def scores(p,a,y0=ftruth,kinds=('whole','phone','edge')):
        result={};length=len(y0)
        for j,k in enumerate(kinds):
            pp=p[j*length:(j+1)*length];aa=a[j*length:(j+1)*length];good=evidence_correct(pp,y0);result[k]=dict(top1=float(good.float().mean()),confirmed=int(aa.sum()),correct_confirmed=int((aa&good).sum()),precision=float(good[aa].float().mean()) if aa.any() else 0.)
        return result
    lp=torch.cat([legacy['probabilities'][k] for k in ('whole','phone')]);lv,li=lp.max(1);la=(lv>=legacy['threshold'])&oa&(modelbrands[li]==op.argmax(1))&fvalid[:len(fv)];lm=scores(lp,la,kinds=('whole','phone'));frounds=[];fb=None
    for rid,config in enumerate(policy['family_rounds'],1):
        torch.manual_seed(300100+rid);proto=torch.stack([torch.nn.functional.normalize(x[ftrain][fy[ftrain]==j].mean(0),dim=0) for j in range(len(pairs))]);head=CosineMakeHead(proto);opt=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=.02);cw=torch.bincount(fy[ftrain],minlength=len(pairs)).float().clamp_min(1).pow(-.3);cw/=cw.mean();chosen=None
        for epoch in range(policy['epochs_each']):
            head.train()
            for ids in ftrain[torch.randperm(len(ftrain))].split(128):
                opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[ids]),fy[ids],weight=cw,label_smoothing=config['label_smoothing']);loss.backward();opt.step()
            head.eval()
            with torch.inference_mode():p=head(x[fv]).softmax(1);raw=scores(p,torch.zeros(len(p),dtype=torch.bool),kinds=('whole','phone'));score=sum(v['top1'] for v in raw.values())/2;guard=all(raw[k]['top1']>=lm[k]['top1'] for k in raw)
            if guard and (chosen is None or score>chosen['score']):chosen=dict(state=copy.deepcopy(head.state_dict()),epoch=epoch+1,score=score,raw=raw)
        record=dict(round=rid,selected_epoch=chosen['epoch'] if chosen else None,passed=False)
        if chosen:
            head.load_state_dict(chosen['state']);head.eval()
            with torch.inference_mode():z=head(q);temperature=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(z/t,ft)));p=bind((z/temperature).softmax(1),mp,ma);gates=calibrate(p,ft,groups,keys,fvalid,.95);accepted=acceptance(p,gates,keys,fvalid);m=scores(p,accepted)
            checks=dict(precision=all(v['precision']>=.95 for v in m.values()),correct_confirmed=all(m[k]['correct_confirmed']>=lm[k]['correct_confirmed'] for k in lm),edge_count=m['edge']['confirmed']>0,song_plus_enabled=gates['BYD|Song Plus']<1,e70_enabled=gates['Dongfeng|Aeolus E70']<1)
            record.update(metrics=m,raw_metrics=chosen['raw'],checks=checks,passed=all(checks.values()))
            if record['passed'] and (fb is None or chosen['score']>fb['score']):fb=dict(score=chosen['score'],head=chosen['state'],temperature=temperature,thresholds=gates,epoch=chosen['epoch'],round=rid)
        frounds.append(record);print('FAMILY ROUND',json.dumps(record),flush=True)
    report=dict(make_passed=best is not None,family_passed=fb is not None,make=make_status,family=dict(baseline=lm,rounds=frounds),truth_sources='Original annotations/manual source review, not predictions. Own first2andSongPlusphoto5not familyTRAIN. Known diagnostics notblind. Legacy public splits reused.',test_used=False)
    if best or fb:
        ck=copy.deepcopy(selected)
        if fb:ck['model_head']=dict(origin='project-trained-family-head-v30',pairs=pairs,head=fb['head'],temperature=fb['temperature'],thresholds=fb['thresholds'],margin=.06,clipped_validation_passed=True)
        ck['truth_v30']=dict(own_first2_training=False,own_photo5_family_training=False,make_round=best['round'] if best else None,family_round=fb['round'] if fb else None,source_review='models/truth_v30',test_used=False);torch.save(ck,out/'candidate.pt')
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('BOUNDED FITS DONE',best is not None,fb is not None,flush=True)
if __name__=='__main__':main()
