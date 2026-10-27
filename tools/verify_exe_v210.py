"""Check the actual frozen app with Python removed from PATH."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def close(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)<=1e-5
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    return a==b
def sig(d):return [(r['box'],r['brand'],r['colour'],r.get('make_evidence'),r.get('model'),r.get('model_alternatives')) for r in d['cars']]
def main():
    target=ROOT/'outputs/CarVision-2.10-Test/CarVision';out=ROOT/'outputs/exe-v210-verification';out.mkdir(exist_ok=True);steps=[]
    env=os.environ.copy();win=Path(next(v for k,v in env.items() if k.lower()=='systemroot'));env.update(PATH=str(win/'System32')+';'+str(win),PYTHONHOME='',PYTHONPATH='',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
    def run(name,args,timeout=240):
        started=time.perf_counter();p=subprocess.run(args,cwd=target,env=env,capture_output=True,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));(out/(name+'.stdout.txt')).write_bytes(p.stdout);(out/(name+'.stderr.txt')).write_bytes(p.stderr);assert p.returncode==0,(name,p.stderr[-1200:]);steps.append(dict(name=name,seconds=time.perf_counter()-started,exit_code=0));print('EXE PASS',name,flush=True);return p.stdout
    exe=str(target/'CarVision.exe');run('help',[exe,'--help']);diagnostics=[]
    for index in (0,1,3,4,2,15):
        source=json.loads((ROOT/f'outputs/mashinki-v210/candidate/{index:03d}.json').read_text(encoding='utf-8'));raw=run('photo-'+str(index+1),[exe,'predict',source['image'],'--mode','fast','--output',str(out/str(index))]);actual=json.loads(raw.decode('utf-8-sig'));assert close(sig(actual),sig(source)),'Frozen/source decision mismatch';maincar=max(actual['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]));expected={0:('Dongfeng','Aeolus E70'),1:('Dongfeng','Aeolus E70'),3:('BYD','Dolphin'),4:('BYD','Song Plus')}.get(index);assert expected is None or (maincar['brand'],maincar['model'])==expected,('Known user error persists',index,maincar['brand'],maincar['model']);assert actual['runtime_identity']['family_labels']==43 and actual['runtime_identity']['build_version']=='2.10 TEST';diagnostics.append(dict(photo=index+1,seconds=actual['seconds'],decisions=[dict(brand=r['brand'],colour=r['colour'],candidate=(r.get('make_evidence') or {}).get('candidate')) for r in actual['cars']]))
        if index in (0,4):
            renamed=out/('independent-name-'+str(index)+'.jpg');shutil.copy2(source['image'],renamed);check=json.loads(run('rename-'+str(index),[exe,'predict',str(renamed),'--mode','fast','--output',str(out/'rename')]).decode('utf-8-sig'));assert close(sig(actual),sig(check))
    ps=win/'System32/WindowsPowerShell/v1.0/powershell.exe';run('selfcheck',[str(ps),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(target/'CHECK_ON_NEW_PC.ps1'),'-Quick'],timeout=360)
    report=dict(passed=True,steps=steps,diagnostics=diagnostics,source_frozen_parity=True,filename_independent=True,python_in_path=False,exe_sha256=hashlib.sha256((target/'CarVision.exe').read_bytes()).hexdigest(),scope='Technical test on this PC. Not a second-PC run, blind phone accuracy or100% guarantee.')
    for p in (out/'verification.json',target/'Development/FIX_2_10/RECIPIENT_VERIFICATION.json'):p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('VERIFIED',json.dumps(report),flush=True)
if __name__=='__main__':main()
