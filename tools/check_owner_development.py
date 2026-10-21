"""Development diagnostics on user training photos, NOT independent accuracy."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Pipeline,save_prediction
p=Pipeline(mode='fast');out=ROOT/'outputs/owner-development-v20';out.mkdir(exist_ok=True)
results=[]
for photo in sorted((ROOT/'data/own_oct21/images').glob('*.jpg')):
 result=p.predict(photo);save_prediction(photo,result,out);results.append(result)
 print(photo.name,result['seconds'],[(r['id'],r['brand'],r.get('model_candidate'),round(r.get('model_confidence',0),2)) for r in result['cars']],flush=True)
(out/'summary.json').write_text(json.dumps(dict(role='Training/development photos. Never independent validation/test.',photos=results),ensure_ascii=False,indent=2),encoding='utf-8')
