"""Two bounded corrections of3verified phone families; public guards retained."""
import copy,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
def main():
    torch.set_num_threads(3);out=ROOT/'models/truth_v31';cache=ROOT/'cache/truth_v31';policy=json.loads((out/'policy.json').read_text());assert policy['rounds_max']==2
    with (out/'training_once.json').open('x') as f:json.dump(dict(started=time.time()),f)
    d=torch.load(cache/'features.pt',weights_only=True);rows=json.loads((cache/'rows.json').read_text(encoding='utf-8'));n=len(rows);pairs=[tuple(p) for p in d['pairs']];x=d['x'];known=lambda r:(r['brand'],r.get('family')) in pairs and not r['image'].startswith('make_scope_v23/');own_images={r['image'] for r in json.loads((out/'truth_corrections.json').read_text(encoding='utf-8'))};ti=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train' and known(r) and r['image'] not in own_images]);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation' and known(r)]);train=torch.cat([ti,ti+n]);val=torch.cat([vi,vi+n]);y=torch.tensor([pairs.index((r['brand'],r.get('family'))) if known(r) else -1 for r in rows]*2);tx=torch.cat([x[train],d['owner_x']]);ty=torch.cat([y[train],d['owner_y']]);private=torch.arange(len(tx))>=len(train);initial=torch.load(ROOT/'models/truth_v30/family_raw.pt',weights_only=True);old=CosineMakeHead(torch.zeros(len(pairs),1280)).eval();old.load_state_dict(initial['head']);change=torch.zeros(len(pairs),dtype=torch.bool)
    for pair in [('BYD','Song Plus'),('BYD','Dolphin'),('Dongfeng','Aeolus E70')]:change[pairs.index(pair)]=True
    with torch.inference_mode():teacher=old(tx[~private]);basep=old(x[val]);baseline=[float((z.argmax(1)==y[vi]).float().mean()) for z in basep.split(len(vi))]
    records=[];best=None
    for rid,config in enumerate(policy['rounds'],1):
        torch.manual_seed(310000+rid);head=CosineMakeHead(torch.zeros(len(pairs),1280));head.load_state_dict(initial['head']);head.log_scale.requires_grad_(False);opt=torch.optim.AdamW([head.weight,head.bias],lr=config['lr'],weight_decay=0.);sw=torch.where(private,torch.tensor(float(config['phone_weight'])),torch.tensor(1.));chosen=None
        for epoch in range(policy['epochs_each']):
            head.train()
            for ids in torch.randperm(len(tx)).split(128):
                opt.zero_grad();z=head(tx[ids]);loss=torch.nn.functional.cross_entropy(z,ty[ids],reduction='none');loss=(loss*sw[ids]).sum()/sw[ids].sum();public=~private[ids]
                if public.any():loss+=4*torch.nn.functional.kl_div((z[public]/2).log_softmax(1),(teacher[ids[public]]/2).softmax(1),reduction='batchmean')
                if (~public).any():
                    ownids=ids[~public];similarity=(tx[ownids]*torch.nn.functional.normalize(head.weight[ty[ownids]],dim=1)).sum(1);loss+=.5*torch.relu(.5-similarity).mean()
                loss.backward();head.weight.grad[~change]=0;head.bias.grad[~change]=0;opt.step()
            head.eval()
            with torch.inference_mode():
                pp=head(d['owner_x']).softmax(1);fit=float((pp.argmax(1)==d['owner_y']).float().mean());confidence=float(pp.gather(1,d['owner_y'][:,None]).min());vp=head(x[val]);accuracy=[float((z.argmax(1)==y[vi]).float().mean()) for z in vp.split(len(vi))];guard=fit==1 and confidence>=.99 and all(a>=b for a,b in zip(accuracy,baseline));score=sum(accuracy)/2
            if guard and (chosen is None or score>chosen['score']):chosen=dict(head=copy.deepcopy(head.state_dict()),epoch=epoch+1,score=score,private_training_fit=fit,private_training_min_confidence=confidence,public_validation=accuracy)
        record=dict(round=rid,passed=chosen is not None,selected={k:v for k,v in chosen.items() if k!='head'} if chosen else None)
        if chosen:
            torch.save(dict(head=chosen['head'],pairs=pairs,round=rid,epoch=chosen['epoch'],private_training_only=True),out/f'round{rid}.pt')
            if best is None or chosen['score']>best['score']:best=dict(chosen,round=rid)
        records.append(record);print('PHONE FAMILY ROUND',json.dumps(record),flush=True)
    if best:torch.save(dict(head=best['head'],pairs=pairs,round=best['round'],epoch=best['epoch'],private_training_only=True),out/'family_raw.pt')
    report=dict(passed=best is not None,baseline_public_validation=baseline,rounds=records,owner_examples_are_train=True,test_used=False);(out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('BOUNDED PHONE CORRECTION DONE',best is not None,flush=True)
if __name__=='__main__':main()
