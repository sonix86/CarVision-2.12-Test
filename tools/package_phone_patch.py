"""Package a binary delta against the sealed 2.12 TEST; do not modify its files."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE_EXE='48d81ca9214d0528553db5ec409b8fe39d686e6bf0c2a09aa45e2d91e3eea830'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert sha(a.base/'CarVision.exe')==BASE_EXE,'Expected the unchanged sealed 2.12 TEST'
    a.output.mkdir(parents=True,exist_ok=True)
    payload=a.output/'payload';payload.mkdir(exist_ok=True)
    manifest=dict(version='2.12 PHONE',base='2.12 TEST',exe_sha256=sha(a.build/'CarVision.exe'),previous_exe_sha256=[BASE_EXE],files=[],required_files=[])
    def add(src,relative):
        target=payload/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
        old=a.base/relative
        manifest['files'].append(dict(path=relative,sha256=sha(target),previous_sha256=[sha(old)] if old.exists() else [],require_existing=old.exists()))
    for f in sorted(a.build.rglob('*')):
        if not f.is_file():continue
        rel=f.relative_to(a.build).as_posix();old=a.base/rel
        if not old.exists() or sha(f)!=sha(old):add(f,rel)
    for name in ('detector.pt','classifier.pt','dino_make.pt','semantic_make.pt'):
        f=a.base/'models'/name
        manifest['required_files'].append(dict(path='models/'+name,sha256=sha(f)))
    add(ROOT/'docs/PHONE_INPUT_RU.txt','PHONE_INPUT_RU.txt')
    check=a.output/'phone-selfcheck.ps1'
    check.write_text((a.base/'CHECK_ON_NEW_PC.ps1').read_text('utf-8-sig').replace("version='2.12 TEST'","version='2.12 PHONE'"),encoding='utf-8-sig')
    add(check,'CHECK_ON_NEW_PC.ps1');check.unlink()
    start=a.output/'START_PHONE.cmd'
    start.write_bytes(b'@echo off\r\ncd /d "%~dp0"\r\nstart "" "%~dp0CarVision.exe" gui --phone\r\n')
    add(start,'START_PHONE.cmd');start.unlink()
    metadata=json.loads((a.base/'models/model_version.json').read_text('utf-8'))
    metadata.update(version='2.12 PHONE',phone_patch='Local HTTP receiver, phone camera stills, optional localhost camera stream. Recognition weights unchanged from 2.12 TEST.')
    meta=a.output/'phone-model-version.json';meta.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),'utf-8');add(meta,'models/model_version.json');meta.unlink()
    (a.output/'UPDATE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
    installer=(ROOT/'handoff/APPLY_UPDATE_2_3.ps1').read_text('utf-8-sig')
    installer=installer.replace('supplied CarVision 2.1/2.2 EXE','original CarVision 2.12 TEST EXE').replace('Update 2.3','Phone patch').replace('update23-','phone212-')
    # Copy to a new file then rename: replacement is safe even for hard-linked test fixtures.
    installer=installer.replace('Copy-Item -LiteralPath $job.source -Destination $job.destination -Force',"$temporary=$job.destination+'.phone-patch-tmp'\n            Copy-Item -LiteralPath $job.source -Destination $temporary -Force\n            Move-Item -LiteralPath $temporary -Destination $job.destination -Force")
    (a.output/'APPLY_PHONE_PATCH.ps1').write_text(installer,encoding='utf-8-sig')
    (a.output/'APPLY_PHONE_PATCH.cmd').write_bytes(b'@echo off\r\n"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0APPLY_PHONE_PATCH.ps1"\r\nset "result=%errorlevel%"\r\npause\r\nexit /b %result%\r\n')
    shutil.copy2(ROOT/'docs/PHONE_INPUT_RU.txt',a.output/'READ_FIRST_RU.txt')
    source=a.output/'source';source.mkdir(exist_ok=True)
    for name in ('launch.py','carvision/phone_input.py','carvision/gui.py','carvision/version.py'):
        f=source/name;f.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,f)
    archive=a.output.parent/'CarVision-2.12-Phone-Patch.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(a.output).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n','ascii')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),changed_files=len(manifest['files']),base_exe_preserved=sha(a.base/'CarVision.exe')==BASE_EXE),indent=2))

if __name__=='__main__':main()
