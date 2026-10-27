"""Run full candidate pipeline on all99 original owner photographs."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import Pipeline,save_prediction

def main():
    out=ROOT/'outputs/mashinki-v29/candidate';out.mkdir(exist_ok=True)
    paths=json.loads((out.parent/'manifest.json').read_text(encoding='utf-8'))
    p=Pipeline(ROOT/'outputs/models-v29-runtime',mode='fast');records=[]
    for item in paths:
        target=out/f"{item['index']:03d}.json"
        if target.exists():r=json.loads(target.read_text(encoding='utf-8'))
        else:
            r=p.predict(item['source']);save_prediction(item['source'],r,out);target.write_text(json.dumps(r,indent=2),encoding='utf-8')
        records.append(dict(index=item['index'],source=item['source'],seconds=r['seconds'],cars=[{k:v for k,v in row.items() if k not in ('segmentation','contours')} for row in r['cars']]))
        (out.parent/'candidate_summary.json').write_text(json.dumps(dict(complete=len(records)==len(paths),processed=len(records),records=records,scope='Mostly unlabelled owner TRAIN/development outing; not independent accuracy.'),indent=2),encoding='utf-8')
        print('CANDIDATE',len(records),len(paths),'CARS',len(r['cars']),'SECONDS',r['seconds'],flush=True)

if __name__=='__main__':main()
