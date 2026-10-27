"""Measure the existing open-vocabulary route on target makes, without EV-only filtering."""
import csv,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from tools.ev26_helpers import route,references,acceptance
from tools.train_truth_v30 import valid

TARGET_MAKES=['BYD','Zeekr','Leapmotor','GAC Aion','Li Auto','Voyah','XPeng','Neta','Nio','Geometry','Avatr','Aito','Denza','Hongqi','Weltmeister','Xiaomi','Stelato','Deepal','Arcfox','Dongfeng']

@torch.inference_mode()
def predict_features(ck,z,geometry,family_geometry):
    brands=ck['brands']
    def head(state,classes):
        h=CosineMakeHead(torch.zeros(classes,z.shape[1])).eval();h.load_state_dict(state);return h
    h=head(ck['head'],len(brands));p,a=route((h(z)/ck['temperature']).softmax(1),references(z,ck['retrieval'],len(brands)),ck['thresholds'],ck['retrieval']['fallback_thresholds'],brands,geometry)
    if ck.get('model_head'):
        f=ck['model_head'];pairs=list(map(tuple,f['pairs']));keys=[b+'|'+m for b,m in pairs];mapping=torch.tensor([brands.index(b) for b,m in pairs]);fh=head(f['head'],len(pairs));fp=(fh(z)/f['temperature']).softmax(1);v,ids=fp.topk(2,dim=1);index=ids[:,0];g=torch.tensor([f['thresholds'][keys[int(i)]] for i in index]);sg=torch.tensor([f['similarity_thresholds'][keys[int(i)]] for i in index]);sim=z@torch.nn.functional.normalize(fh.weight,dim=1).T;fa=(~a)&family_geometry&(g<1)&(v[:,0]>=g)&((v[:,0]-v[:,1])>=f['margin'])&(sim.gather(1,index[:,None])[:,0]>=sg)&torch.isin(mapping[index],torch.tensor([brands.index(b) for b in f['manufacturer_fallback_brands']]));mp=torch.zeros_like(p);mp.scatter_add_(1,mapping.expand(len(z),-1),fp);p=torch.where(fa[:,None],mp,p);a=a|fa
    if ck.get('make_correction_head'):
        c=ck['make_correction_head'];ch=head(c['head'],len(brands));cp=(ch(z)/c['temperature']).softmax(1);ca=acceptance(cp,c['thresholds'],brands,family_geometry,c['margin']);p=torch.where(ca[:,None],cp,p);a=a|ca
    return p,a

def main():
    torch.set_num_threads(3);out=ROOT/'outputs/ev-specialist-v212';out.mkdir(exist_ok=True);rows=json.loads((ROOT/'cache/truth_v31/rows.json').read_text(encoding='utf-8'));d=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True);ck=torch.load(ROOT/'models/dino_make.pt',weights_only=True);brands=ck['brands'];targets=[b for b in TARGET_MAKES if b in brands];n=len(rows);report=dict(version='2.12 TEST',target_makes=targets,scope='Target-make cohort, NOT proof of electric drivetrain. Mixed makes such as BYD/Dongfeng can include ICE/PHEV. All82 labels remain competitors, never renormalize over EV-only classes. Cached validation/test are reused development evidence, not blind Uzbekistan accuracy.',splits={},class_counts={b:{s:sum(r['brand']==b and r['split']==s for r in rows) for s in ('train','validation','test')} for b in targets},unsupported_known_make=['AIQAR']);table=[]
    for split in ('validation','test'):
        ii=torch.tensor([i for i,r in enumerate(rows) if r['split']==split]);rs=[rows[int(i)] for i in ii];z=d['x'][torch.cat([ii,ii+n])];geom=torch.cat([valid(rs,k) for k in ('whole','phone')]);fg=torch.cat([valid(rs,k,True) for k in ('whole','phone')]);p,a=predict_features(ck,z,geom,fg);truth=torch.tensor([brands.index(r['brand']) for r in rs]*2);correct=p.argmax(1)==truth;target=torch.tensor([r['brand'] in targets for r in rs]*2);count=len(rs);stats={}
        for j,view in enumerate(('whole','phone')):
            ix=torch.arange(j*count,(j+1)*count);selected=ix[target[ix]];groups=len({r['group'] for r in rs if r['brand'] in targets});den=len(selected);good=int((a[selected]&correct[selected]).sum());wrong=int((a[selected]&~correct[selected]).sum());unknown=int((~a[selected]).sum());stats[view]=dict(count=den,source_groups=groups,correct_confirmed=good,wrong_confirmed=wrong,unknown=unknown,successful_fraction=good/max(1,den),raw_top1=float(correct[selected].float().mean()),confirmed_precision=good/max(1,good+wrong))
            for b in targets:
                idx=ix[truth[ix]==brands.index(b)];table.append(dict(split=split,view=view,brand=b,count=len(idx),correct_confirmed=int((a[idx]&correct[idx]).sum()),wrong_confirmed=int((a[idx]&~correct[idx]).sum()),unknown=int((~a[idx]).sum())))
        report['splits'][split]=stats;print(split,json.dumps(stats),flush=True)
    (out/'baseline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'per_brand.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print('EV target-cohort audit complete; no weights changed.',flush=True)
if __name__=='__main__':main()
