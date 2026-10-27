"""Verify audit fixes in the built EXE, including the formerly stale detector-only route."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    app=ROOT/'outputs/CarVision-2.12-Test/CarVision';out=ROOT/'outputs/exe-v212-verification';out.mkdir(exist_ok=True);env=os.environ.copy();win=Path(next(v for k,v in env.items() if k.lower()=='systemroot'));env.update(PATH=str(win/'System32')+';'+str(win),PYTHONHOME='',PYTHONPATH='',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3');exe=str(app/'CarVision.exe');steps=[]
    def run(name,args,timeout=240):
        start=time.perf_counter();p=subprocess.run(args,cwd=app,env=env,capture_output=True,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));(out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,(name,p.stderr[-1200:]);steps.append(dict(name=name,seconds=time.perf_counter()-start));print('AUDIT EXE PASS',name,flush=True);return p.stdout
    def sig(d):return [(c['box'],c['brand'],c.get('model'),c['colour']) for c in d['cars']]
    run('help',[exe,'--help']);jobs=[]
    guard=subprocess.run([exe,'train','--labels',str(out/'missing-labels.jsonl'),'--output',str(app/'models')],cwd=app,env=env,capture_output=True,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    (out/'training-output-guard.stderr').write_bytes(guard.stderr)
    # Frozen Windows stderr can use the console code page despite PYTHONIOENCODING.
    assert guard.returncode!=0 and all(token in guard.stderr for token in (b'ValueError:',b'dino_make.pt',b'--output')) and b'FileNotFoundError' not in guard.stderr,'EXE does not contain the final training-output guard'
    steps.append(dict(name='training-output-guard',rejected_before_training=True));print('AUDIT EXE PASS training-output-guard',flush=True)
    for name in ('zeekr','leapmotor','prior-1','prior-2','prior-4','prior-5'):
        source=json.loads((ROOT/f'outputs/user-error-audit-v32/runtime/{name}.json').read_text(encoding='utf-8'));path=source['image']
        if not Path(path).exists():path=str(ROOT/f'outputs/exe-v211-verification/{name}-renamed.jpg')
        assert Path(path).exists(),path;jobs.append((name,path,source));r=json.loads(run(name,[exe,'predict',path,'--mode','fast','--output',str(out/'results')]).decode('utf-8-sig'));assert sig(r)==sig(source),(name,'ordinary2.11answers changed unexpectedly');assert r['runtime_identity']['build_version']=='2.12 TEST' and r['runtime_identity']['weights_version']=='2.12 TEST';assert r['runtime_identity']['paired_clip_sha256']=='94a67eb9cc5f52e89cc309bdec66a027b2b7818219f121f4a0fe1023bf547f23'
    models=out/'detector-only-models';models.mkdir(exist_ok=True)
    for p in (app/'models').iterdir():
        if p.is_file() and p.name!='segmenter.pt' and not (models/p.name).exists():os.link(p,models/p.name)
    first=next(j[1] for j in jobs if j[0]=='prior-1');r=json.loads(run('detector-only',[exe,'predict',first,'--models',str(models),'--mode','fast','--output',str(out/'detector-only')]).decode('utf-8-sig'));assert r['cars'],'No box detected in legacy route';assert all(c.get('make_first') and c.get('make_evidence') for c in r['cars']);assert r['runtime_identity']['active_head_file']=='dino_make.pt';assert r['classifier_trained']
    ps=win/'System32/WindowsPowerShell/v1.0/powershell.exe';run('selfcheck',[str(ps),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(app/'CHECK_ON_NEW_PC.ps1'),'-Quick'],360)
    result=dict(passed=True,steps=steps,ordinary_211_answers_unchanged=True,detector_only_uses_active_make_head=True,paired_clip_verified=True,python_in_path=False,exe_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),scope='Engineering audit, not improved classification accuracy. Existing known cases remain TRAIN/development. Original garage and AIQAR photographs currently unavailable.')
    for p in (out/'verification.json',app/'Development/FIX_2_12/RECIPIENT_VERIFICATION.json'):p.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('AUDIT EXE VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':main()
