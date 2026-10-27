"""Recover the already selected round1epoch49, not another candidate search."""
import json,sys,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
def main():
    torch.set_num_threads(3);out=ROOT/'models/truth_v30';cache=ROOT/'cache/truth_v30';report=json.loads((out/'validation.json').read_text());selected=report['family']['rounds'][0];assert selected['selected_epoch']==49
    with (out/'reconstruction_once.json').open('x') as f:json.dump(dict(reason='Failed candidates were not serialized by original training script. Deterministically reconstruct the fixed round1epoch49 for architecture analysis; no new epoch/candidate selection.',seed=300101,epoch=49),f)
    rows=json.loads((cache/'all_rows.json').read_text(encoding='utf-8'));d=torch.load(cache/'joint.pt',weights_only=True);e=torch.load(cache/'e70_features.pt',weights_only=True)['x'];ne=len(e)//2;no=len(rows)-ne;n=len(rows);x=torch.cat([d['x'][:no],e[:ne],d['x'][no:],e[ne:]]);pairs=[tuple(p) for p in d['pairs']]+[('Dongfeng','Aeolus E70')];known=lambda r:(r['brand'],r.get('family')) in pairs and not r['image'].startswith('make_scope_v23/');ti=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train' and known(r)]);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation' and known(r)]);train=torch.cat([ti,ti+n]);val=torch.cat([vi,vi+n]);y=torch.tensor([pairs.index((r['brand'],r.get('family'))) if known(r) else -1 for r in rows]*2)
    torch.manual_seed(300101);proto=torch.stack([torch.nn.functional.normalize(x[train][y[train]==j].mean(0),dim=0) for j in range(len(pairs))]);head=CosineMakeHead(proto);opt=torch.optim.AdamW(head.parameters(),lr=.0005,weight_decay=.02);cw=torch.bincount(y[train],minlength=len(pairs)).float().clamp_min(1).pow(-.3);cw/=cw.mean()
    for epoch in range(49):
        head.train()
        for ids in train[torch.randperm(len(train))].split(128):opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[ids]),y[ids],weight=cw,label_smoothing=.02);loss.backward();opt.step()
    head.eval()
    with torch.inference_mode():p=head(x[val]).softmax(1);metrics={k:float((p[j*len(vi):(j+1)*len(vi)].argmax(1)==y[vi]).float().mean()) for j,k in enumerate(('whole','phone'))}
    assert all(abs(metrics[k]-selected['raw_metrics'][k]['top1'])<1e-7 for k in metrics),'Deterministic recovery differs from recorded validation result'
    torch.save(dict(head=head.state_dict(),pairs=pairs,seed=300101,epoch=49,original_metrics_matched=metrics),out/'family_raw.pt');print('FIXED HEAD RECOVERED',json.dumps(metrics),flush=True)
if __name__=='__main__':main()
