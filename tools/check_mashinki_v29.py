"""Check all user photos with 2.6 weights, bind to actual EXE, keep originals."""
import argparse,hashlib,json,re,subprocess,sys,time
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import Pipeline,read_image,save_prediction

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(3);out=ROOT/'outputs/mashinki-v29';out.mkdir(exist_ok=True)
    if args.resume:assert (out/'baseline_once.json').exists()
    else:
        with (out/'baseline_once.json').open('x') as f:json.dump(dict(started=time.time()),f)
    paths=sorted(Path('D:/машинки').glob('*.jpg'),key=lambda p:0 if p.name=='11223344.jpg' else int(re.search(r'photo_(\d+)',p.name).group(1)))
    manifest=[dict(index=i,source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for i,p in enumerate(paths)]
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for first in range(0,len(paths),16):
        sheet=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(paths[first:first+16]):
            im=ImageOps.contain(read_image(p),(295,265));x=j%4*300;y=j//4*300;sheet.paste(im,(x+(300-im.width)//2,y+25));draw.text((x+5,y+5),f'{first+j}: '+p.name.split('_2026')[0],fill='black')
        sheet.save(out/f'originals_{first//16:02d}.jpg',quality=95)
    target=ROOT/'outputs/CarVision-2.6-Full/CarVision';pipeline=Pipeline(target/'models',mode='fast');results=[];checks=[]
    for i,p in enumerate(paths):
        saved=out/'baseline'/f'{i:03d}.json'
        if args.resume and saved.exists():result=json.loads(saved.read_text(encoding='utf-8'))
        else:
            result=pipeline.predict(p);save_prediction(p,result,out/'baseline');saved.write_text(json.dumps(result,indent=2),encoding='utf-8')
        if i in (0,3,10):
            existing=list((out/'exe-check'/str(i)).glob('*.json'))
            if args.resume and existing:actual=json.loads(existing[0].read_text(encoding='utf-8'))
            else:
                run=subprocess.run([str(target/'CarVision.exe'),'predict',str(p),'--mode','fast','--output',str(out/'exe-check'/str(i))],cwd=target,capture_output=True,timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));assert run.returncode==0,run.stderr[-1000:]
                actual=json.loads(run.stdout.decode('utf-8-sig'))
            def sig(d):return [(r['box'],r['brand'],r['colour'],r.get('make_evidence'),r.get('model'),r.get('model_alternatives')) for r in d['cars']]
            def close(a,b):
                if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)<=1e-5
                if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
                if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
                return a==b
            assert close(sig(actual),sig(result)),'Actual EXE decisions/geometry differ or probabilities exceed numerical tolerance'
            checks.append(dict(index=i,exe_parity=True,numeric_absolute_tolerance=1e-5))
        for r in result['cars']:r.pop('segmentation',None)
        results.append(dict(index=i,source=str(p),cars=result['cars'],seconds=result['seconds']))
        (out/'summary.json').write_text(json.dumps(dict(complete=i+1==len(paths),processed=i+1,total=len(paths),exe_checks=checks,records=results,role='Owner training/development pool; mostly unlabelled, not independent accuracy.'),indent=2),encoding='utf-8')
        print('PHOTO',i+1,len(paths),'CARS',len(result['cars']),'SECONDS',result['seconds'],flush=True)

if __name__=='__main__':main()
