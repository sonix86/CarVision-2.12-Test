"""One predeclared fusion: preserve accepted makes, calibrate family fallback."""
import copy,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from tools.train_truth_v30 import valid,evidence_correct
from tools.ev26_helpers import route,references,calibrate,acceptance,metrics

def main():
    torch.set_num_threads(3);out=ROOT/'models/truth_v30';cache=ROOT/'cache/truth_v30';config=dict(rule='Preserve every accepted v29 make. A directly accepted exact-family class may establish manufacturer on make abstentions. Calibrate exact-family probability AND absolute feature-similarity gates on source-labelled families plus all other-make negatives across the entire make validation. Unknown same-make families are excluded from exact-family scoring, not counted as correct. No known-owner fit/selection.',fits_added=0,fixed_family_round=1,fixed_family_epoch=49,make_precision_min=.95,family_precision_min=.95,min_source_groups=3,known_error_selection=False)
    with (out/'joint_once.json').open('x') as f:json.dump(dict(started=time.time(),config=config),f,indent=2)
    rows=json.loads((cache/'all_rows.json').read_text(encoding='utf-8'));d=torch.load(cache/'joint.pt',weights_only=True);ex=torch.load(cache/'e70_features.pt',weights_only=True)['x'];ne=len(ex)//2;no=len(rows)-ne;n=len(rows);x=torch.cat([d['x'][:no],ex[:ne],d['x'][no:],ex[ne:]]);brands=d['brands'];base=torch.load(ROOT/'models/owner_domain_v29/dino_make.pt',weights_only=True);mh=CosineMakeHead(torch.zeros(len(brands),1280)).eval();mh.load_state_dict(base['head']);family=torch.load(out/'family_raw.pt',weights_only=True);pairs=[tuple(p) for p in family['pairs']];keys=[b+'|'+m for b,m in pairs];fh=CosineMakeHead(torch.zeros(len(pairs),1280)).eval();fh.load_state_dict(family['head']);family_to_make=torch.tensor([brands.index(b) for b,m in pairs]);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation']);vr=[rows[int(i)] for i in vi];v=torch.cat([vi,vi+n]);known=lambda r:(r['brand'],r.get('family')) in pairs and not r['image'].startswith('make_scope_v23/');fi=torch.tensor([i for i in vi if known(rows[int(i)])]);fr=[rows[int(i)] for i in fi];fv=torch.cat([fi,fi+n]);edge=torch.load(cache/'edge_validation.pt',weights_only=True);assert edge['indices']==fi.tolist();ft=torch.tensor([pairs.index((r['brand'],r['family'])) for r in fr]*3);frows=len(fr);vq=torch.cat([x[v],edge['x']]);vg=torch.cat([valid(vr),valid(vr,'phone'),torch.ones(frows,dtype=torch.bool)]);truth=torch.tensor([brands.index(r['brand']) for r in vr]*2+[brands.index(r['brand']) for r in fr]);groupnames=sorted({r['group'] for r in vr});groups=torch.tensor([groupnames.index(r['group']) for r in vr]*2+[groupnames.index(r['group']) for r in fr]);fvalid=torch.cat([valid(fr,'whole',True),valid(fr,'phone',True),edge['geometry']]);fgroups=torch.tensor([groupnames.index(r['group']) for r in fr]*3)
    def oldmake(z,good):
        p=(mh(z)/base['temperature']).softmax(1);return route(p,references(z,base['retrieval'],len(brands)),base['thresholds'],base['retrieval']['fallback_thresholds'],brands,good)
    with torch.inference_mode():
        z=fh(torch.cat([x[fv],edge['x']]));temperature=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(z/t,ft)));fp=(fh(vq)/temperature).softmax(1);op,oa=oldmake(vq,vg)
        # Direct family can establish its manufacturer. Unknown same-make
        # families are unscored, never silently declared correct or incorrect.
        known_family=torch.tensor([pairs.index((r['brand'],r.get('family'))) if known(r) else -1 for r in vr]*2+[pairs.index((r['brand'],r['family'])) for r in fr])
        all_geometry=torch.cat([valid(vr,'whole',True),valid(vr,'phone',True),edge['geometry']]);pv,pi=fp.topk(2,dim=1);similarities=vq@torch.nn.functional.normalize(fh.weight,dim=1).T;thresholds={};similarity_gates={}
        for j,key in enumerate(keys):
            allowed=((known_family>=0)|(truth!=family_to_make[j]))&((~oa)|(op.argmax(1)==family_to_make[j]));success=known_family==j;options=[]
            for threshold in (.5,.6,.7,.8,.9,.95,.98):
                for similarity in (.0,.2,.3,.4,.5,.6,.7):
                    eligible=allowed&all_geometry&(pi[:,0]==j)&(pv[:,0]>=threshold)&((pv[:,0]-pv[:,1])>=.06)&(similarities[:,j]>=similarity)
                    if eligible.any() and len(groups[eligible&success].unique())>=3 and float(success[eligible].float().mean())>=.95:options.append((int(eligible.sum()),threshold,similarity))
            _,thresholds[key],similarity_gates[key]=max(options,default=(0,1.,1.))
        gatevec=torch.tensor([thresholds[keys[int(j)]] for j in pi[:,0]]);simvec=torch.tensor([similarity_gates[keys[int(j)]] for j in pi[:,0]]);direct=all_geometry&(gatevec<1)&(pv[:,0]>=gatevec)&((pv[:,0]-pv[:,1])>=.06)&(similarities.gather(1,pi[:,:1])[:,0]>=simvec)
        mp=torch.zeros(len(vq),len(brands));mp.scatter_add_(1,family_to_make.expand(len(vq),-1),fp);fa=(~oa)&direct;combined=torch.where(fa[:,None],mp,op);accepted=oa|fa
        fpositions=torch.tensor([vi.tolist().index(int(i)) for i in fi]);positions=torch.cat([fpositions,fpositions+len(vi),torch.arange(2*len(vi),len(vq))]);fm=combined[positions];fam_a=accepted[positions];fprob=fp[positions];bound=fprob*(family_to_make[None,:]==fm.argmax(1)[:,None])*fam_a[:,None];af=direct[positions]&(family_to_make[pi[positions,0]]==fm.argmax(1))&fam_a;gates={b:1. for b in brands};gates['Dongfeng']=thresholds['Dongfeng|Aeolus E70']
    for r in vr:r['y']=brands.index(r['brand']);r.setdefault('historical',False)
    priority=json.loads((ROOT/'models/owner_domain_v29/policy.json').read_text())['priority_makes'];bm=metrics(op[:2*len(vi)],oa[:2*len(vi)],vr,brands,priority);cm=metrics(combined[:2*len(vi)],accepted[:2*len(vi)],vr,brands,priority);legacy=torch.load(cache/'legacy_model_validation.pt',weights_only=True);lp=torch.cat([legacy['probabilities'][k] for k in ('whole','phone')]);om,om_a=oldmake(x[fv],torch.cat([valid(fr),valid(fr,'phone')]));lv,li=lp.max(1);la=(lv>=legacy['threshold'])&om_a&(family_to_make[li]==om.argmax(1))&fvalid[:2*frows];lg=evidence_correct(lp,ft[:2*frows]);fg=evidence_correct(bound,ft);baseline={};candidate={}
    for j,k in enumerate(('whole','phone','edge')):
        a=af[j*frows:(j+1)*frows];good=fg[j*frows:(j+1)*frows];candidate[k]=dict(correct_confirmed=int((a&good).sum()),confirmed=int(a.sum()),precision=float(good[a].float().mean()) if a.any() else 0.)
        if j<2:
            a=la[j*frows:(j+1)*frows];good=lg[j*frows:(j+1)*frows];baseline[k]=dict(correct_confirmed=int((a&good).sum()),confirmed=int(a.sum()),precision=float(good[a].float().mean()) if a.any() else 0.)
    checks=dict(make_top1=all(cm[k]['top1']>=bm[k]['top1'] for k in bm),make_precision=all(cm[k]['precision']>=max(.95,bm[k]['precision']-.01) for k in bm),make_correct=all(cm[k]['correct_confirmed']>=bm[k]['correct_confirmed'] for k in bm),family_precision=all(v['precision']>=.95 for v in candidate.values()),family_correct=all(candidate[k]['correct_confirmed']>=baseline[k]['correct_confirmed'] for k in baseline),song_plus_enabled=thresholds['BYD|Song Plus']<1,e70_enabled=thresholds['Dongfeng|Aeolus E70']<1,dongfeng_fallback_enabled=gates['Dongfeng']<1)
    report=dict(passed=all(checks.values()),similarity_thresholds=similarity_gates,checks=checks,make_baseline=bm,make_candidate=cm,family_baseline=baseline,family_candidate=candidate,family_thresholds=thresholds,make_fallback_thresholds=gates,config=config,test_used=False)
    if report['passed']:
        ck=copy.deepcopy(base);ck['model_head']=dict(origin='project-trained-family-head-v30',pairs=pairs,head=family['head'],temperature=temperature,thresholds=thresholds,margin=.06,clipped_validation_passed=True,similarity_thresholds=similarity_gates,direct_manufacturer_fallback=True,global_margin_required=True);ck['truth_v30']=dict(own_first2_training=False,own_photo5_family_training=False,fixed_family_round=1,fixed_family_epoch=49,make_head_unchanged=True,independently_calibrated_family_manufacturer_fallback=True);torch.save(ck,out/'joint_candidate.pt')
    (out/'joint_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
