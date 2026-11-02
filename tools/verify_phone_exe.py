"""Local transport and direct/phone inference parity check; not a phone accuracy test."""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('app',type=Path);p.add_argument('photo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.app=a.app.resolve();a.photo=a.photo.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();windows=os.environ.get('SystemRoot','C:\\Windows');env['PATH']=windows+'\\System32;'+windows;env['PYTHONHOME']='';env['PYTHONPATH']='';env['OMP_NUM_THREADS']='3';env['MKL_NUM_THREADS']='3'
    exe=a.app/'CarVision.exe'
    common=dict(cwd=a.app,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    help_run=subprocess.run([str(exe),'--help'],capture_output=True,timeout=90,**common)
    assert help_run.returncode==0 and b'phone' in help_run.stdout,help_run.stderr
    ready_path=a.output/'ready.json'
    assert not ready_path.exists(),'Use a fresh verification output folder'
    with (a.output/'server.stdout.log').open('wb') as stdout,(a.output/'server.stderr.log').open('wb') as stderr:
        proc=subprocess.Popen([str(exe),'phone','--host','127.0.0.1','--port','0','--ready-file',str(ready_path),'--output',str(a.output/'http')],stdout=stdout,stderr=stderr,**common)
        try:
            started=time.monotonic()
            while not ready_path.exists():
                assert proc.poll() is None,'Server exited; inspect logs'
                assert time.monotonic()-started<90,'Server startup timeout'
                time.sleep(.25)
            ready=json.loads(ready_path.read_text('utf-8'));base=ready['usb_url']
            with urllib.request.urlopen(base,timeout=10) as r:
                assert r.status==200 and b'capture="environment"' in r.read()
            started=time.monotonic()
            with urllib.request.urlopen(urllib.request.Request(base+'upload',data=a.photo.read_bytes()),timeout=30) as r:
                assert r.status==202;ident=json.load(r)['id']
            while True:
                assert time.monotonic()-started<180,'Inference timeout'
                time.sleep(.5)
                with urllib.request.urlopen(base+'jobs/'+ident,timeout=10) as r:job=json.load(r)
                if job['status']!='processing':break
            assert job['status']=='done',job
            with urllib.request.urlopen(base+job['preview_url'],timeout=10) as r:preview=r.read()
            assert preview[:2]==b'\xff\xd8'
            (a.output/'preview.jpg').write_bytes(preview)
            normalised=next((a.output/'http/phone_inbox').glob('*/'+ident+'.jpg'))
            direct=subprocess.run([str(exe),'predict',str(normalised),'--mode','fast','--output',str(a.output/'direct')],capture_output=True,timeout=180,**common)
            assert direct.returncode==0,direct.stderr.decode('utf-8','replace')
            result=json.loads(direct.stdout)
            keys=('id','brand','colour','model')
            phone_answers=[tuple(r.get(k) for k in keys) for r in job['cars']]
            direct_answers=[tuple(r.get(k) for k in keys) for r in result['cars']]
            assert phone_answers==direct_answers,(phone_answers,direct_answers)
            evidence=dict(passed=True,scope='Windows EXE, PATH excludes Python, local HTTP upload, real offline model, preview, direct/phone answer parity on the identical normalised image. Known development photo; not independent accuracy or a physical phone/camera/USB test.',phone_seconds=job['seconds'],direct_seconds=result['seconds'],answers=phone_answers,preview_bytes=len(preview),weights_identity=result.get('runtime_identity',{}))
            (a.output/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),'utf-8')
            print(json.dumps(evidence,ensure_ascii=True,indent=2),flush=True)
        finally:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
