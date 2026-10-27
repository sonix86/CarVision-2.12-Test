"""Finite colour comparison with our supervised heads and preserved source splits."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageEnhance,ImageFilter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import COLOURS,read_image
from carvision.dino_features import DinoEncoder,dino_input,descriptor
from carvision.colour_model import ColourClassifier,paint_statistics
from carvision.masked_colour import MaskedColourClassifier
from carvision.segmentation import isolate

OUT=ROOT/'models/dino_v28';CACHE=ROOT/'cache/dino_v28'

def view(im,mask,kind):
    im=im.copy();mask=np.asarray(mask)
    if kind=='stress':
        im.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST))
        im=ImageEnhance.Brightness(im).enhance(.55).filter(ImageFilter.GaussianBlur(.35))
        im=Image.fromarray(np.clip(np.asarray(im,dtype=np.float32)*np.array([.95,1.07,.93]),0,255).astype('uint8'))
    return im,mask

def prepare(test=False):
    if test:assert json.loads((OUT/'colour_validation.json').read_text())['passed']
    target=CACHE/('colour_test.pt' if test else 'colour_development.pt');assert not target.exists()
    manifest=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text())['rows'];chosen=[]
    for split in (['test'] if test else ['train','validation']):
        for colour in COLOURS:
            cohort=sorted([r for r in manifest if r['split']==split and r['colour']==colour],key=lambda r:r['sha256'])[:50 if split=='train' else 20]
            chosen.extend(dict(r,source='VCoR',owner=False,group='source-image:'+r['sha256']) for r in cohort)
    # Remove near-identical cross-split views without moving an original split.
    locked=json.loads((OUT/'colour_dataset.json').read_text())['rows'] if test else []
    locked=[r for r in locked if r.get('source')=='VCoR']
    kept=[]
    for r in chosen:
        if any(s['split']!=r['split'] and (int(s['dhash'],16)^int(r['dhash'],16)).bit_count()<=5 for s in locked+kept):continue
        kept.append(r)
    chosen=kept;images=[]
    for r in chosen:
        path=ROOT/'datasets/vcor/images'/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        im=read_image(path);images.append((im,np.ones((im.height,im.width),bool)))
    if not test:
        data=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True)
        for r in data['rows']:
            if r.get('colour') in COLOURS:
                assert r['split']=='train'
                chosen.append(dict(image=r['image'],colour=r['colour'],split='train',owner=True,group=r['group'],source='existing manually labelled own training crop'))
                images.append((Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy()))
        del data
    encoder=DinoEncoder().eval();encoder.load_state_dict(torch.load(CACHE/'dinov2_vits14_pretrain.pth',weights_only=True))
    modeldir=ROOT/'outputs/CarVision-2.6-Full/CarVision/models';old=ColourClassifier(torch.load(modeldir/'colour_model.pt',weights_only=True));paint=MaskedColourClassifier(torch.load(modeldir/'masked_colour.pt',weights_only=True))
    jobs=[(r,im,mask,k) for k in ('whole','stress') for r,(im,mask) in zip(chosen,images)];features=[];baseline=[];started=time.time()
    with torch.inference_mode():
        for first in range(0,len(jobs),8):
            batch=jobs[first:first+8];processed=[view(im,mask,k) for _,im,mask,k in batch];z=descriptor(encoder(torch.stack([dino_input(im,mask) for im,mask in processed])))
            for i,(im,mask) in enumerate(processed):
                focused=isolate(im,mask);base=old.probabilities(focused);base=paint.blend(base,paint.probabilities(im,mask));baseline.append(base)
                features.append(torch.cat([z[i],paint_statistics(focused),base]))
            if first%80==0:print('COLOUR FEATURES',first+len(batch),len(jobs),round(time.time()-started,1),flush=True)
    x=torch.stack(features);base=torch.stack(baseline)
    torch.save(dict(x=x,baseline=base,rows=chosen,classes=COLOURS,kinds=['whole','stress'],limitations='Preserved VCoR source splits with cross-split dHash exclusion; vehicle identity/author groups unavailable for historical VCoR. Own outing TRAIN only. Synthetic stress is not a blind real phone exam.'),target)
    (OUT/('colour_test_dataset.json' if test else 'colour_dataset.json')).write_text(json.dumps(dict(rows=chosen,records=len(chosen),features=len(jobs),classes=COLOURS,source_manifest_sha256=hashlib.sha256((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_bytes()).hexdigest(),test=test),indent=2),encoding='utf-8')

def measure(p,y,ids,n):
    return {k:float((p[ids+offset].argmax(1)==y[ids+offset]).float().mean()) for k,offset in [('whole',0),('stress',n)]}

def train():
    with (OUT/'colour_training_once.json').open('x') as f:json.dump(dict(rounds_max=2,epochs_each=100),f)
    d=torch.load(CACHE/'colour_development.pt',weights_only=True);rows=d['rows'];n=len(rows);x=d['x'];y=torch.tensor([COLOURS.index(r['colour']) for r in rows]*2);ti=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train']);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation']);train=torch.cat([ti,ti+n]);val=torch.cat([vi,vi+n]);assert {rows[i]['group'] for i in ti}.isdisjoint({rows[i]['group'] for i in vi})
    mean=x[train].mean(0);std=x[train].std(0).clamp_min(.03);x=(x-mean)/std;base=measure(d['baseline'],y,vi,n);sw=torch.tensor([4. if r['owner'] else 1. for r in rows]*2);rounds=[];best=None
    for rid in (1,2):
        torch.manual_seed(280010+rid);dimension=x.shape[1]-(8 if rid==1 else 0)
        head=torch.nn.Sequential(torch.nn.Linear(dimension,128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,8));opt=torch.optim.AdamW(head.parameters(),lr=.0007,weight_decay=.03);chosen=None;history=[]
        for epoch in range(100):
            head.train()
            for ids in train[torch.randperm(len(train))].split(64):
                opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[ids,:dimension]),y[ids],label_smoothing=.02,reduction='none');loss=(loss*sw[ids]).sum()/sw[ids].sum();loss.backward();opt.step()
            head.eval()
            with torch.inference_mode():m=measure(head(x[:,:dimension]),y,vi,n)
            checks=dict(whole=m['whole']>=base['whole'],stress=m['stress']>=base['stress']+.02);score=(m['whole']+m['stress'])/2;history.append(dict(epoch=epoch+1,metrics=m,checks=checks))
            if all(checks.values()) and (chosen is None or score>chosen['score']):chosen=dict(score=score,head=copy.deepcopy(head.state_dict()),epoch=epoch+1,metrics=m,dimension=dimension,round=rid)
        rounds.append(dict(round=rid,history=history,passed=chosen is not None))
        if chosen and (best is None or chosen['score']>best['score']):best=chosen
    if best:
        head=torch.nn.Sequential(torch.nn.Linear(best['dimension'],128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,8));head.load_state_dict(best['head']);head.eval()
        with torch.inference_mode():
            logits=head(x[val,:best['dimension']]);temperature=min((.5,.75,1.,1.5,2.,3.),key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,y[val])))
        torch.save(dict(origin='project-trained-dino-colour-v28',head=best['head'],mean=mean,std=std,classes=COLOURS,dimension=best['dimension'],round=best['round'],epoch=best['epoch'],temperature=temperature),OUT/'dino_colour.pt')
    (OUT/'colour_validation.json').write_text(json.dumps(dict(passed=best is not None,baseline=base,selected={k:v for k,v in best.items() if k!='head'} if best else None,rounds=rounds,test_used=False),indent=2),encoding='utf-8')
    print('TWO COLOUR FITS COMPLETE',best is not None,'BASELINE',base,'SELECTED',best['metrics'] if best else None,flush=True)

def evaluate():
    assert json.loads((OUT/'colour_validation.json').read_text())['passed']
    with (OUT/'colour_evaluation_once.json').open('x') as f:json.dump(dict(post_selection=True),f)
    ck=torch.load(OUT/'dino_colour.pt',weights_only=True);d=torch.load(CACHE/'colour_test.pt',weights_only=True);n=len(d['rows']);y=torch.tensor([COLOURS.index(r['colour']) for r in d['rows']]*2)
    head=torch.nn.Sequential(torch.nn.Linear(ck['dimension'],128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,8));head.load_state_dict(ck['head']);head.eval()
    with torch.inference_mode():p=(head(((d['x']-ck['mean'])/ck['std'])[:,:ck['dimension']])/ck['temperature']).softmax(1)
    ids=torch.arange(n);base=measure(d['baseline'],y,ids,n);candidate=measure(p,y,ids,n);checks={k:candidate[k]>=base[k] for k in base}
    report=dict(passed=all(checks.values()),checks=checks,baseline=base,candidate=candidate,images=n,limitations=d['limitations'])
    (OUT/'colour_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','prepare-test','train','evaluate']);args=parser.parse_args();torch.set_num_threads(3)
    if args.phase=='train':train()
    elif args.phase=='evaluate':evaluate()
    else:prepare(args.phase=='prepare-test')
