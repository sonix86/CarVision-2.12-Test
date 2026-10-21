"""Find original sample files behind recent user screenshots; retain no filename labels."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Pipeline,save_prediction
p=Pipeline(mode='fast');out=ROOT/'outputs/user-regressions-v21';out.mkdir(exist_ok=True)
paths=set()
for f in sorted((ROOT/'dist/CarVision/outputs').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:12]:
 try:
  path=Path(json.loads(f.read_text(encoding='utf-8'))['image'])
  if path.exists():paths.add(path)
 except (KeyError,ValueError):pass
# Exercise the entire27image EV holdout folder, including the Chery that
# previously raised KeyError('brand') before it could write a result file.
paths.update((ROOT/'samples/Chinese_EV_Test').glob('*.jpg'))
summary=[]
for path in sorted(paths):
 result=p.predict(path);save_prediction(path,result,out)
 for row in result['cars']:
  if row.get('brand_model_conflict'):assert row['brand']=='unknown' and row['model']=='unknown'
 summary.append(dict(image=str(path),cars=len(result['cars']),seconds=result['seconds'],conflicts=sum(r.get('brand_model_conflict',False) for r in result['cars'])))
 print(path.name,'OK',len(result['cars']),flush=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
