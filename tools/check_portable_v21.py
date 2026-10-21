"""Copy the application to a path with spaces; run actual models without Python in PATH."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Pipeline
source=ROOT/'dist/CarVision';destination=ROOT/'outputs/Portable v2.1 София'
p=argparse.ArgumentParser();p.add_argument('--refresh-models',action='store_true');args=p.parse_args()
destination.mkdir(parents=True,exist_ok=True)
if args.refresh_models:
    assert (destination/'CarVision.exe').exists()
    shutil.copytree(source/'models',destination/'models',dirs_exist_ok=True)
else:shutil.copytree(source,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('outputs','cache'))
env=os.environ.copy();system_root=Path(env.get('SystemRoot','C:/Windows'));env['PATH']=str(system_root/'System32')+';'+str(system_root);env['PYTHONHOME']='';env['PYTHONPATH']='';env['PYTHONIOENCODING']='utf-8'
photo=destination/'samples/images/03.jpg'
proc=subprocess.run([str(destination/'CarVision.exe'),'predict',str(photo),'--mode','fast','--output',str(destination/'outputs')],cwd=destination,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
if proc.returncode:raise RuntimeError('Portable EXE failed: '+proc.stderr[-2500:]+proc.stdout[-1000:])
actual=json.loads(proc.stdout);expected=Pipeline(source/'models',mode='fast').predict(photo)
def signature(result):return [(r['brand'],r.get('model'),r.get('model_candidate'),r.get('model_alternatives'),r['colour'],r['box'],r.get('segmentation'),r.get('model_ranking'),r.get('model_limited')) for r in result['cars']]
assert signature(actual)==signature(expected),'Frozen and source predictions differ'
assert list((destination/'outputs').glob('*.json')) and list((destination/'outputs').glob('*.txt')) and list((destination/'outputs').glob('*annotated.jpg'))
region=max(actual['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]))['box']
roi_proc=subprocess.run([str(destination/'CarVision.exe'),'predict',str(photo),'--mode','fast','--region',*[str(v) for v in region],'--single','--output',str(destination/'outputs')],cwd=destination,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
assert roi_proc.returncode==0,roi_proc.stderr[-2000:];roi_result=json.loads(roi_proc.stdout);assert len(roi_result['cars'])==1 and roi_result.get('segmented') and roi_result['cars'][0].get('contours')
result=dict(passed=True,path=str(destination),python_in_path=False,model_sha256={name:hashlib.sha256((destination/'models'/name).read_bytes()).hexdigest() for name in ['classifier.pt','detector.pt','colour_model.pt','vehicle_model.pt','segmenter.pt','masked_colour.pt','visual_references.pt']},predictions_match_sources=True,seconds=actual['seconds'],cars=actual['cars'],scope='Same physical machine, separate copied folder. Does not certify another physical computer.')
result['single_car_roi_passed']=True
(ROOT/'outputs/portable-v21-check.json').write_text(json.dumps(result,indent=2))
print('Portable EXE passed: no installed Python in PATH, actual trained models, matching results and three exported file types.',flush=True)
print(json.dumps(dict(seconds=actual['seconds'],cars=[(r['brand'],r['colour']) for r in actual['cars']]),indent=2))

