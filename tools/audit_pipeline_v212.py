"""Audit current cached training pixels and shipped weights, never relabel from predictions."""
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.audit_ev_baseline_v212 import predict_features,TARGET_MAKES
from tools.train_truth_v30 import valid
from carvision.segmentation import isolate

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(3);out=ROOT/'outputs/pipeline-audit-v212';out.mkdir(exist_ok=True);app=ROOT/'outputs/CarVision-2.11-Test/CarVision';meta=json.loads((ROOT/'cache/truth_v31/rows.json').read_text(encoding='utf-8'));data=torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True)['rows'];data+=torch.load(ROOT/'cache/truth_v30/e70_crops.pt',weights_only=True)['rows'];data.append(torch.load(ROOT/'models/owner_domain_v29/proposals.pt',weights_only=True)['rows'][15]);lookup={r['image']:r for r in data};rows=[];groups=collections.defaultdict(list);pixels=collections.defaultdict(list);sources=collections.defaultdict(list);bad=[]
    for i,m in enumerate(meta):
        r=dict(lookup[m['image']],**m);rgb=r['body_rgb'];mask=r['body_mask'];rows.append(r)
        if tuple(rgb.shape[:2])!=tuple(mask.shape) or rgb.ndim!=3 or rgb.shape[-1]!=3 or int(mask.sum())==0:bad.append(dict(index=i,image=m['image'],reason='invalid RGB/mask geometry'))
        digest=hashlib.sha256(str((tuple(rgb.shape),str(rgb.dtype),tuple(mask.shape),str(mask.dtype))).encode()+rgb.numpy().tobytes()+mask.numpy().tobytes()).hexdigest();pixels[digest].append(i);groups[m['group']].append(i)
        def origin(v):
            if isinstance(v,dict):
                for k in ('original_sha256','sha256'):
                    if isinstance(v.get(k),str) and len(v[k])==64:return v[k]
                for k in ('source','original'):
                    if isinstance(v.get(k),dict):
                        q=origin(v[k])
                        if q:return q
        s=origin(m.get('source',{}))
        if s:sources[s].append(i)
        if i%1000==0:print('AUDIT PIXELS',i,len(meta),flush=True)
    def clashes(bank,field):return [dict(key=k,indices=ids,images=[meta[i]['image'] for i in ids],values=sorted({str(meta[i].get(field)) for i in ids})) for k,ids in bank.items() if len({str(meta[i].get(field)) for i in ids})>1]
    report=dict(rows=len(rows),invalid_geometry=bad,group_split_leaks=clashes(groups,'split'),pixel_split_leaks=clashes(pixels,'split'),pixel_label_conflicts=clashes(pixels,'brand'),source_split_leaks=clashes(sources,'split'),source_label_conflicts=clashes(sources,'brand'),source_hash_rows=sum(map(len,sources.values())),scope='Current cached3851 crops only; contradictions are flags for source review, predictions are never ground truth.')
    ck=torch.load(app/'models/dino_make.pt',weights_only=True);clip=torch.load(app/'models/semantic_make.pt',weights_only=True);reference=torch.load(ROOT/'models/ev_views_v26/semantic_make.pt',weights_only=True);report['joint_encoder_pairing']=dict(declared_clip_checkpoint_sha256=ck.get('clip_checkpoint_sha256'),actual_clip_checkpoint_sha256=sha(app/'models/semantic_make.pt'),encoder_matches_training_source=all(torch.equal(v,reference['encoder_state'][k]) for k,v in clip['encoder_state'].items()))
    features=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True)['x'];norm=features.norm(dim=1);report['cached_features']=dict(rows=len(features),expected_rows=2*len(meta),finite=bool(torch.isfinite(features).all()),norm_min=float(norm.min()),norm_max=float(norm.max()));brands=ck['brands'];report['taxonomy']=dict(brands=len(brands),families=len(ck['model_head']['pairs']),family_brands_missing=sorted({b for b,m in ck['model_head']['pairs']}-set(brands)),disabled_make_classes=[b for b in brands if ck['thresholds'][b]>=1 and ck['retrieval']['fallback_thresholds'][b]>=1]);(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('STRUCTURAL AUDIT',json.dumps({k:len(report[k]) for k in ('invalid_geometry','group_split_leaks','pixel_split_leaks','pixel_label_conflicts','source_split_leaks','source_label_conflicts')}),flush=True)
    trainids=[i for i,r in enumerate(meta) if r['split']=='train'];ix=torch.tensor(trainids);rs=[meta[i] for i in trainids];p,a=predict_features(ck,features[ix],valid(rs),valid(rs,family=True));truth=torch.tensor([brands.index(r['brand']) for r in rs]);score=p.max(1).values-p.gather(1,truth[:,None])[:,0];wrong=p.argmax(1)!=truth;order=score.argsort(descending=True).tolist();chosen=[]
    for ev in (True,False):
        chosen.extend([j for j in order if bool(wrong[j]) and (rs[j]['brand'] in TARGET_MAKES)==ev][:12])
    flags=[]
    for page in range((len(chosen)+11)//12):
        sheet=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(sheet)
        for slot,j in enumerate(chosen[page*12:(page+1)*12]):
            idx=trainids[j];r=rows[idx];x=slot%4*300;y=slot//4*300;im=ImageOps.contain(isolate(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy()),(294,245));sheet.paste(im,(x+(300-im.width)//2,y+42));prediction=brands[int(p[j].argmax())];draw.text((x+3,y+3),f"#{idx} LABEL: {r['brand']}",fill='black');draw.text((x+3,y+19),f"GUESS: {prediction} {float(p[j].max()):.0%}",fill='red');flags.append(dict(index=idx,image=r['image'],brand=r['brand'],family=r.get('family'),source=r.get('source'),candidate=prediction,confidence=float(p[j].max()),role='Unreviewed training-data flag; NEVER auto-relabel from a classifier'))
        sheet.save(out/f'flagged_train_{page}.jpg',quality=95)
    (out/'flagged_training.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2),encoding='utf-8');print('FLAGGED TRAIN CONTACTS',len(flags),flush=True)
    from carvision.dino_make import DinoMakeExpert
    expert=DinoMakeExpert(ck,clip);checks=[]
    with torch.inference_mode():
        for i in (0,100,450,1000,1800,2500,3300,3850):
            r=rows[i];actual=expert.feature(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy());err=float((actual-features[i]).abs().max());checks.append(dict(index=i,image=r['image'],max_error=err,passed=err<1e-4));print('CACHE PARITY',i,err,flush=True)
    report['runtime_cache_parity']=checks;(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
