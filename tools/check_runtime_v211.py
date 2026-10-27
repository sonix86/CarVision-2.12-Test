import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import Pipeline
def main():
    assert json.loads((ROOT/'models/phone_v32/regression.json').read_text())['passed'];out=ROOT/'outputs/user-error-audit-v32/runtime';out.mkdir(parents=True,exist_ok=True);md=ROOT/'outputs/models-v32-runtime';md.mkdir(exist_ok=True);base=ROOT/'outputs/CarVision-2.10-Test/CarVision/models'
    for f in base.iterdir():
        if f.is_file() and f.name not in ('dino_make.pt','model_version.json') and not (md/f.name).exists():os.link(f,md/f.name)
    import shutil
    shutil.copy2(ROOT/'models/phone_v32/scoped_candidate.pt',md/'dino_make.pt');h=hashlib.sha256((md/'dino_make.pt').read_bytes()).hexdigest();(md/'model_version.json').write_text(json.dumps(dict(version='2.11 TEST',verify_active_head=True,selected_model_sha256=h)),encoding='utf-8');p=Pipeline(md,mode='fast');records=[]
    jobs=[('zeekr','C:/Users/Miku/Downloads/photo_2026-10-27_19-49-23.jpg'),('leapmotor','C:/Users/Miku/Downloads/photo_2026-10-26_15-59-53.jpg'),('aiqar','C:/Users/Miku/Downloads/photo_2026-10-26_11-51-34.jpg'),('garage','C:/Users/Miku/Downloads/photo_2026-10-26_15-53-32.jpg')]
    for i in (0,1,3,4):
        d=json.loads((ROOT/f'outputs/mashinki-v210/candidate/{i:03d}.json').read_text(encoding='utf-8'));jobs.append(('prior-'+str(i+1),d['image']))
    for name,path in jobs:
        r=p.predict(path);(out/(name+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8');summary=dict(name=name,path=path,seconds=r['seconds'],cars=[dict(brand=c['brand'],model=c.get('model'),candidate=(c.get('make_evidence') or {}).get('candidate'),box=c['box'],colour=c['colour']) for c in r['cars']]);records.append(summary);print(json.dumps(summary),flush=True)
        maincar=max(r['cars'],key=lambda c:(c['box'][2]-c['box'][0])*(c['box'][3]-c['box'][1]));expected={'zeekr':('Zeekr',None),'leapmotor':('Leapmotor',None),'prior-1':('Dongfeng','Aeolus E70'),'prior-2':('Dongfeng','Aeolus E70'),'prior-4':('BYD','Dolphin'),'prior-5':('BYD','Song Plus')}.get(name)
        if expected:assert maincar['brand']==expected[0] and (expected[1] is None or maincar['model']==expected[1]),(name,maincar['brand'],maincar['model'])
    (out/'summary.json').write_text(json.dumps(dict(passed_required_controls=True,records=records,scope='Known TRAIN/development cases; AIQAR and Voyah remain unresolved; not blind accuracy.'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
