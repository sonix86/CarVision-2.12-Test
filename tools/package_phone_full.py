"""Create a self-contained PHONE release without changing the sealed TEST ZIP.

Unchanged local ZIP records are copied byte for byte; zipfile writes a fresh
central directory and new/updated records. Every final payload SHA and CRC is
then verified. This avoids recompressing unchanged multi-gigabyte model data.
"""
import copy
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('D:/Projects/TashkentCarVision/outputs')
BASE=WORK/'CarVision-2.12-Test-Full.zip'
PATCH=WORK/'CarVision-2.12-Phone-Patch/payload'
OUT=WORK/'phone-full-verified'
INTRO='''CarVision 2.12 PHONE — полная версия

Распакуйте архив целиком в новую папку. Запустите CarVision/CarVision.exe.
В заголовке должно быть 2.12 PHONE. Патчи и установка Python не нужны.
CarVision.exe, models и _internal должны оставаться вместе.

Телефон: нажмите «С телефона». Подключите телефон и компьютер к одной
Wi-Fi-сети. Выберите адрес Wi-Fi компьютера и отсканируйте QR-код.
«Выбрать фото» — галерея. «Снять камерой» — новый снимок с подтверждением.
После выбора фото анализ запускается сам, ответы видны на обоих устройствах.
При запросе Windows разрешите CarVision доступ к частной сети.
USB и дополнительная последовательная камера: PHONE_INPUT_RU.txt.

Владелец проверил выбор фото из галереи на реальном телефоне и получение
результата на обоих устройствах. Камера открывается; отправка нового снимка
ещё не проверена. Физический USB-режим не проверен.
Веса распознавания прежние, точность от этого обновления не увеличивается.
Предыдущая 2.12 TEST и её ZIP сохранены без изменений.
'''

def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    archive=OUT/'CarVision-2.12-Phone-Full.zip'
    assert not archive.exists(),'Do not overwrite a sealed full release'
    overlays={}
    for f in PATCH.rglob('*'):
        if f.is_file():overlays['CarVision/'+f.relative_to(PATCH).as_posix()]=f
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in filter(None,tracked):
        f=ROOT/name
        if f.is_file() and name not in ('release_assets.json',) and 'CarVision/'+name not in overlays:overlays['CarVision/'+name]=f
    generated={}
    for name in ('READ_FIRST_RU.txt','CarVision/START_HERE_RU.txt','CarVision/TRANSFER.txt','CarVision/CHECKLIST_NEW_PC.txt'):
        generated[name]=INTRO.encode('utf-8-sig')
    generated['CarVision/BUILD_PHONE.json']=json.dumps(dict(version='2.12 PHONE',base='2.12 TEST',build_date='2026-11-03',recognition_weights_unchanged=True,phone_gallery_confirmed_by_owner=True,native_camera_open_confirmed_by_owner=True,captured_photo_submission_confirmed=False,physical_usb_tested=False,exe_sha256=sha_file(PATCH/'CarVision.exe')),indent=2).encode()
    with zipfile.ZipFile(BASE) as source:
        hashes=json.loads(source.read('CarVision/TRANSFER_CONTENTS.json'))['files']
        for name,f in overlays.items():hashes[name.removeprefix('CarVision/')]=sha_file(f)
        for name,data in generated.items():
            if name.startswith('CarVision/'):hashes[name.removeprefix('CarVision/')]=hashlib.sha256(data).hexdigest()
        hashes.pop('TRANSFER_CONTENTS.json',None);hashes.pop('SHA256SUMS.txt',None)
        generated['CarVision/TRANSFER_CONTENTS.json']=json.dumps(dict(version='2.12 PHONE',files=hashes,recognition_weights_unchanged=True,base_release='v2.12-test'),ensure_ascii=False,indent=2).encode()
        generated['CarVision/SHA256SUMS.txt']=''.join(h+'  '+n+'\n' for n,h in sorted(hashes.items())).encode()
        replaced=set(overlays)|set(generated)
        infos=sorted(source.infolist(),key=lambda i:i.header_offset)
        with BASE.open('rb') as raw,zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as dest:
            for i,info in enumerate(infos):
                if info.filename in replaced:continue
                end=infos[i+1].header_offset if i+1<len(infos) else source.start_dir
                raw.seek(info.header_offset);remaining=end-info.header_offset
                new=copy.copy(info);new.header_offset=dest.fp.tell()
                while remaining:
                    b=raw.read(min(8*1024*1024,remaining));assert b;dest.fp.write(b);remaining-=len(b)
                dest.filelist.append(new);dest.NameToInfo[new.filename]=new;dest.start_dir=dest.fp.tell()
            for name,f in sorted(overlays.items()):
                if name not in generated:dest.write(f,name)
            for name,data in generated.items():dest.writestr(name,data)
    print('ARCHIVE WRITTEN; checking every payload hash and ZIP CRC',flush=True)
    with zipfile.ZipFile(archive) as check:
        assert len(check.namelist())==len(set(check.namelist()))
        for i,(name,expected) in enumerate(hashes.items()):
            h=hashlib.sha256()
            with check.open('CarVision/'+name) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==expected,name
            if i%3000==0:print('VERIFIED',i,len(hashes),flush=True)
        assert check.testzip() is None
    manifest=dict(repository='https://github.com/sonix86/CarVision-2.12-Test',release_tag='v2.12-phone',version='2.12 PHONE',original_zip=dict(name=archive.name,bytes=archive.stat().st_size,sha256=sha_file(archive)),parts=[],weights_sha256=hashes['models/dino_make.pt'],exe_sha256=hashes['CarVision.exe'],scope='Self-contained full app including phone input; recognition weights unchanged; previous TEST release preserved')
    with archive.open('rb') as f:
        number=1
        while True:
            first=f.read(8*1024*1024)
            if not first:break
            path=OUT/(archive.name+f'.{number:03}')
            h=hashlib.sha256();written=0
            with path.open('wb') as part:
                data=first
                while data:
                    part.write(data);h.update(data);written+=len(data)
                    if written==1_500_000_000:break
                    data=f.read(min(8*1024*1024,1_500_000_000-written))
            manifest['parts'].append(dict(name=path.name,bytes=written,sha256=h.hexdigest()));number+=1
    (OUT/'release_assets.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    old=(ROOT/'.publication/assets/RESTORE_FULL.ps1').read_text('utf-8-sig')
    start=old.index("$taskManifest = @'");end=old.index("'@ | ConvertFrom-Json",start)
    new=old[:start]+"$taskManifest = @'\n"+json.dumps(manifest,indent=2)+'\n'+old[end:]
    (OUT/'RESTORE_FULL.ps1').write_text(new,encoding='utf-8-sig')
    shutil.copy2(ROOT/'.publication/assets/RESTORE_FULL.cmd',OUT/'RESTORE_FULL.cmd')
    files=[OUT/p['name'] for p in manifest['parts']]+[OUT/x for x in ('release_assets.json','RESTORE_FULL.cmd','RESTORE_FULL.ps1')]
    (OUT/'SHA256SUMS.txt').write_text(''.join(sha_file(p)+'  '+p.name+'\n' for p in files),'ascii')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
