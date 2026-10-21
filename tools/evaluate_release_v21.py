"""Measure current make decisions on existing validation; no fitting or tuning."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from collections import Counter
from carvision.core import ROOT,Pipeline
from carvision.scene import classify
torch.set_num_threads(3);p=Pipeline(mode='fast');data=torch.load(ROOT/'cache/body_expert_v20.pt',weights_only=True)
rows=[r for r in data['rows'] if r['split']=='validation'];pairs=data['pairs'];brands=sorted({b for b,m in pairs});columns=sorted(set(brands)|set(p.brands))+['unknown'];matrix=np.zeros((len(brands),len(columns)),dtype=int);records=[];started=time.time()
for i,r in enumerate(rows):
 result=classify(p,Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy());truth=pairs[r['y']][0];matrix[brands.index(truth),columns.index(result['brand'])]+=1
 records.append(dict(image=r['image'],truth=truth,prediction=result['brand'],conflict=result['brand_model_conflict']))
 if i%40==0:print(i,'/',len(rows),flush=True)
report=dict(scope='Existing validation, manually/source-labelled body crops, no detector metric and not a fresh blind test',count=len(rows),brand_accuracy=sum(r['truth']==r['prediction'] for r in records)/len(rows),unknown=sum(r['prediction']=='unknown' for r in records),conflicts=sum(r['conflict'] for r in records),brands=brands,columns=columns,confusion=matrix.tolist(),records=records,seconds=time.time()-started,training_brand_counts=dict(Counter(pairs[r['y']][0] for r in data['rows'] if r['split']=='train')))
(ROOT/'models/release_v21_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('COMPLETE',report['brand_accuracy'],flush=True)
