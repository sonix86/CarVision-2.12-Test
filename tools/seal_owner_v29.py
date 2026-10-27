"""Seal the verified TEST build; never replace a previous archive."""
import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    target=ROOT/'outputs/CarVision-2.9-Test/CarVision';out=ROOT/'outputs/exe-v29-verification';check=json.loads((out/'verification.json').read_text());assert check['passed'] and sha(target/'CarVision.exe')==check['exe_sha256']
    shutil.copy2(ROOT/'outputs/tests-v29.log',target/'Development/EV29/tests-v29.log')
    for name in ('verify_owner_exe_v29.py','seal_owner_v29.py','summarize_owner_v29.py'):shutil.copy2(ROOT/'tools'/name,target/'tools'/name)
    manifest_path=target/'TRANSFER_CONTENTS.json';stamp=manifest_path.stat().st_mtime;old=json.loads(manifest_path.read_text())['files'];paths=sorted(p for p in target.rglob('*') if p.is_file() and not any(s in ('outputs','patch-backups','__pycache__') for s in p.relative_to(target).parts) and p.name not in ('TRANSFER_CONTENTS.json','SHA256SUMS.txt'))
    manifest={}
    for p in paths:
        rel=p.relative_to(target).as_posix();manifest[rel]=old[rel] if rel in old and p.stat().st_mtime<=stamp else sha(p)
    manifest_path.write_text(json.dumps(dict(version='2.9 TEST',files=manifest,verification='Development/EV29/RECIPIENT_VERIFICATION.json'),indent=2),encoding='utf-8')
    (target/'SHA256SUMS.txt').write_text('\n'.join(v+'  '+k for k,v in manifest.items())+'\n',encoding='utf-8');paths.extend([manifest_path,target/'SHA256SUMS.txt'])
    archive=ROOT/'CarVision-2.9-Test.zip';assert not archive.exists(),'Never replace a sealed archive'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        z.write(target.parent/'READ_FIRST_RU.txt','READ_FIRST_RU.txt')
        for i,p in enumerate(paths):
            z.write(p,'CarVision/'+p.relative_to(target).as_posix())
            if i%4000==0:print('ZIP WRITE',i,len(paths),flush=True)
    print('ZIP WRITTEN; VERIFY ALL PAYLOADS',flush=True)
    with zipfile.ZipFile(archive) as z:
        for i,info in enumerate(z.infolist()):
            h=hashlib.sha256()
            with z.open(info) as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            rel=info.filename.removeprefix('CarVision/')
            if rel in manifest:assert h.hexdigest()==manifest[rel],rel
            if i%4000==0:print('ZIP VERIFIED',i,len(z.infolist()),flush=True)
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii');report=dict(verified=True,path=str(archive),bytes=archive.stat().st_size,sha256=digest,all_payload_sha256_and_crc_checked=True,test_build_only=True,quality_goal_achieved=False)
    (out/'archive.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
