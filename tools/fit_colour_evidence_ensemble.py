"""Mix independently trained colour evidence using validation only.

Full vehicle and lower-body views address glass/background contamination. Mixture
weights are selected on VCoR validation and the declared 23 development reviews;
neither official test labels nor city example colours select the weights.
"""
import argparse,itertools,json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS,read_image
from carvision.colour_model import ColourClassifier

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--panels',action='store_true');args=parser.parse_args()
 torch.set_num_threads(1 if args.panels else 2);out=ROOT/'models'/('colour_panels_candidate' if args.panels else 'colour_ensemble_candidate');out.mkdir(exist_ok=True)
 paths=[ROOT/'models/backups/v12/colour_model.pt',ROOT/'models/robust_colour_candidate/colour_model.pt',ROOT/'models/paint_colour_candidate/colour_model.pt']
 if args.panels:paths=paths[:2]
 if not paths[0].exists():paths[0]=ROOT/'models/colour_model.pt'
 checkpoints=[torch.load(p,weights_only=True) for p in paths];models=[ColourClassifier(c) for c in checkpoints]
 rows=json.loads((ROOT/'data/public_prototype/colour_dataset_manifest.json').read_text())['rows'];validation=[r for r in rows if r['split']=='validation'];dev=[r for r in json.loads((ROOT/'data/public_prototype/colour_development_review.json').read_text())['references'] if r['reference_colour'] is not None]
 images=[read_image(ROOT/'datasets/vcor/images'/r['image']) for r in validation]+[read_image(ROOT/'data/public_prototype'/r['image']).crop(r['box']) for r in dev];y=torch.tensor([COLOURS.index(r['colour']) for r in validation]+[COLOURS.index(r['reference_colour']) for r in dev]);n=len(validation);regions=[None,[.08,.4,.92,.9]]
 if args.panels:regions += [[.05,.35,.35,.85],[.65,.35,.95,.85]]
 views=[];descriptions=[];started=time.perf_counter()
 for component,model in enumerate(models):
  for region in regions:
   predictions=[]
   for i,image in enumerate(images):
    view=image.crop(tuple(round(v*(image.width if k%2==0 else image.height)) for k,v in enumerate(region))) if region else image
    predictions.append(model.probabilities(view))
   views.append(torch.stack(predictions));descriptions.append(dict(component=component,region=region));print('EXTRACTED',component,region,flush=True)
 x=torch.stack(views);candidates=[];best=None
 # Coarse simplex, declared before inspecting any test results. Full-view single
 # models are included, so a mixture must actually improve development evidence.
 for counts in itertools.product(range(5),repeat=len(views)):
  if sum(counts)!=4:continue
  weights=torch.tensor(counts,dtype=torch.float32)/4;prob=(x*weights[:,None,None]).sum(0);pred=prob.argmax(1);a=float((pred[:n]==y[:n]).float().mean());b=float((pred[n:]==y[n:]).float().mean());score=(a+b)/2;row=dict(weights=weights.tolist(),validation_accuracy=a,cross_domain_development_accuracy=b,selection_score=score)
  if a>=.88 and (best is None or score>best['selection_score']):best=row
  candidates.append(row)
 mixture=[dict(desc,weight=w) for desc,w in zip(descriptions,best['weights']) if w>0]
 ck=dict(origin='team-trained-colour-ensemble',architecture='colour_evidence_ensemble_v3',classes=COLOURS,components=checkpoints,mixture=mixture,source='Own trained VCoR heads; full/body view mixture chosen on development only',ready_made_colour_model_used=False)
 torch.save(ck,out/'colour_model.pt');info=dict(best=best,views=descriptions,candidates=candidates,seconds=time.perf_counter()-started,train=6210,validation=n,test=sum(r['split']=='test' for r in rows),selection='Coarse simplex weights in 0.25 steps over three own trained heads and full/lower-body views; maximize mean VCoR validation and 23 separate development accuracy, VCoR>=88%; city examples and both tests excluded',test_used_for_selection=False)
 (out/'training.json').write_text(json.dumps(info,indent=2));print('SELECTED',json.dumps(best),'mixture',json.dumps(mixture),flush=True)
if __name__=='__main__':main()
