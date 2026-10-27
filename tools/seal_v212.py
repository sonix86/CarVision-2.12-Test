"""Seal the verified 2.12 portable app; never overwrite older releases."""
import hashlib,json,os,zipfile
from pathlib import Path
from tools.seal_v211 import sha,write,read
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'outputs/CarVision-2.12-Test/CarVision'
INTRO='''CarVision 2.12 TEST - полная версия для передачи

1. Windows 10/11, 64 бита. Передайте ZIP целиком.
2. «Извлечь всё» в НОВУЮ папку, например C:\\CarVision212.
3. Откройте папку CarVision и запустите CarVision.exe.
4. В заголовке окна должно быть CarVision 2.12 TEST.
5. Открыть фото -> Анализировать фото. Нажмите контур или строку машины.

Python, pip, интернет, API-ключи и установка прежних патчей для анализа
не нужны. Не запускайте EXE внутри ZIP. Не переносите EXE отдельно:
models и _internal должны оставаться рядом. Со старой версией не смешивайте.
Это самостоятельная полная сборка, а не патч поверх 2.11.

Проверка на другом компьютере: CHECK_ON_NEW_PC.cmd.
PASS подтверждает выполнение анализа, области, экспорта и загрузки весов;
он не означает правильность распознавания всех автомобилей.
Проверено на текущем ПК с исключённым Python из PATH. Второй физический
компьютер ещё не проверен. Первое открытие модели может быть медленнее.

Подробности: GUIDE_2_12_RU.txt, AUDIT_RU.txt.
Включены исправления загрузки модели, запасной ветки детектора и проверок
данных. Веса распознавания такие же, как в 2.11. Рост точности не заявлен.
Ошибки AIQAR, Voyah, Tracker, конкретных моделей и части цветов сохраняются.
Каталог 325 названий не означает 325 обученных моделей. Это версия TEST.
'''

def main():
    archive=ROOT/'outputs/CarVision-2.12-Test-Full.zip'
    assert not archive.exists(),'A sealed archive must not be overwritten'
    v=read(APP/'Development/FIX_2_12/RECIPIENT_VERIFICATION.json')
    stage=read(ROOT/'outputs/stage-v212.json');version=read(APP/'models/model_version.json')
    assert v['passed'] and version['version']=='2.12 TEST'
    assert sha(APP/'CarVision.exe')==stage['exe_sha256']==v['exe_sha256']
    assert sha(APP/'models/dino_make.pt')==stage['model_sha256']==version['selected_model_sha256']
    # Stage shares hard links with previous releases: all changes use atomic replacement.
    for name in ('START_HERE_RU.txt','TRANSFER.txt','CHECKLIST_NEW_PC.txt'):
        write(APP/name,INTRO)
    p=APP/'CHECK_ON_NEW_PC.ps1';write(p,p.read_text('utf-8-sig').replace("version='2.10 TEST'","version='2.12 TEST'"))
    first=APP.parent/'READ_FIRST_RU.txt';write(first,INTRO)
    exclude={'outputs','patch-backups','__pycache__','.git'}
    files=sorted(p for p in APP.rglob('*') if p.is_file() and not any(x in exclude for x in p.relative_to(APP).parts) and p.name not in ('TRANSFER_CONTENTS.json','SHA256SUMS.txt') and not p.name.endswith('.seal211-tmp'))
    hashes={}
    for i,p in enumerate(files):
        hashes[p.relative_to(APP).as_posix()]=sha(p)
        if i%2000==0:print('HASH',i,len(files),flush=True)
    manifest=dict(version='2.12 TEST',files=hashes,weights_unchanged_from='2.11 TEST',scope='Full offline Windows x64 transfer; runtime outputs and caches excluded; experimental recognition')
    write(APP/'TRANSFER_CONTENTS.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    write(APP/'SHA256SUMS.txt',''.join(h+'  '+n+'\n' for n,h in hashes.items()))
    temp=archive.with_suffix('.zip.tmp')
    print('PACK',len(files),sum(p.stat().st_size for p in files),flush=True)
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,p in enumerate(files+[APP/'TRANSFER_CONTENTS.json',APP/'SHA256SUMS.txt']):
            z.write(p,'CarVision/'+p.relative_to(APP).as_posix())
            if i%2000==0:print('ZIP',i,len(files),flush=True)
        z.write(first,'READ_FIRST_RU.txt')
    with zipfile.ZipFile(temp) as z:
        for i,(name,expected) in enumerate(hashes.items()):
            h=hashlib.sha256()
            with z.open('CarVision/'+name) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==expected,name
            if i%2000==0:print('VERIFY',i,len(hashes),flush=True)
        assert json.loads(z.read('CarVision/TRANSFER_CONTENTS.json'))==manifest
        assert z.read('CarVision/SHA256SUMS.txt')==(APP/'SHA256SUMS.txt').read_bytes()
        assert z.read('READ_FIRST_RU.txt')==first.read_bytes()
    os.replace(temp,archive)
    result=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),payload_files=len(files),all_payload_hashes_verified=True,all_zip_entries_crc_checked=True,version='2.12 TEST',exe_sha256=v['exe_sha256'],second_physical_pc_tested=False)
    write(ROOT/'outputs/seal-v212.json',json.dumps(result,indent=2))
    write(archive.with_suffix('.zip.sha256'),result['sha256']+'  '+archive.name+'\n')
    print('SEALED',json.dumps(result),flush=True)

if __name__=='__main__':main()
