"""Verify the actual 2.11 EXE, including unresolved cases, without Python in PATH."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    app=ROOT/'outputs/CarVision-2.11-Test/CarVision';out=ROOT/'outputs/exe-v211-verification';out.mkdir(exist_ok=True);exe=str(app/'CarVision.exe');env=os.environ.copy();win=Path(next(v for k,v in env.items() if k.lower()=='systemroot'));env.update(PATH=str(win/'System32')+';'+str(win),PYTHONHOME='',PYTHONPATH='',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3');steps=[]
    def run(name,args,timeout=240):
        start=time.perf_counter();p=subprocess.run(args,cwd=app,env=env,capture_output=True,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));(out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,(name,p.stderr[-1000:]);steps.append(dict(name=name,seconds=time.perf_counter()-start));print('EXE211 PASS',name,flush=True);return p.stdout
    def sig(d):return [(c['box'],c['brand'],c.get('model'),c['colour']) for c in d['cars']]
    run('help',[exe,'--help'])
    summary=json.loads((ROOT/'outputs/user-error-audit-v32/runtime/summary.json').read_text(encoding='utf-8'))
    for row in summary['records']:
        name=row['name'];source=json.loads((ROOT/f'outputs/user-error-audit-v32/runtime/{name}.json').read_text(encoding='utf-8'));r=json.loads(run(name,[exe,'predict',row['path'],'--mode','fast','--output',str(out/'results')]).decode('utf-8-sig'));assert sig(r)==sig(source),(name,'source/EXE mismatch');assert r['runtime_identity']['build_version']=='2.11 TEST' and r['runtime_identity']['weights_version']=='2.11 TEST'
        if name in ('zeekr','leapmotor'):
            renamed=out/(name+'-renamed.jpg');shutil.copy2(row['path'],renamed);z=json.loads(run(name+'-renamed',[exe,'predict',str(renamed),'--mode','fast','--output',str(out/'renamed')]).decode('utf-8-sig'));assert sig(z)==sig(r)
    ps=win/'System32/WindowsPowerShell/v1.0/powershell.exe';run('selfcheck',[str(ps),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(app/'CHECK_ON_NEW_PC.ps1'),'-Quick'],360)
    report=dict(passed=True,steps=steps,source_exe_parity=True,filename_independent=True,python_in_path=False,exe_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),scope='Technical test on current PC and known TRAIN/development scenes only. Zeekr/Leapmotor/BYD corrected; AIQAR, Voyah and whiteTracker remain unresolved. No all-correct guarantee.')
    for p in (out/'verification.json',app/'Development/FIX_2_11/RECIPIENT_VERIFICATION.json'):p.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('VERIFIED',json.dumps(report),flush=True)
if __name__=='__main__':main()
