"""Full offline transfer ZIP; verify every archived payload SHA256 and CRC."""
import hashlib,json,os,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'outputs/CarVision-2.11-Test/CarVision'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,s):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.seal211-tmp');temp.write_text(s,encoding='utf-8-sig' if p.suffix=='.txt' else 'utf-8');os.replace(temp,p)
def main():
    verified=read(APP/'Development/FIX_2_11/RECIPIENT_VERIFICATION.json');stage=read(ROOT/'outputs/stage-v211.json');version=read(APP/'models/model_version.json');assert verified['passed'] and version['version']=='2.11 TEST';assert sha(APP/'CarVision.exe')==stage['exe_sha256']==verified['exe_sha256'];assert sha(APP/'models/dino_make.pt')==stage['model_sha256']==version['selected_model_sha256'];assert (APP/'_internal').is_dir()
    intro='CarVision 2.11 TEST — полный перенос на другой компьютер\n\n1. Передайте этот ZIP целиком. Windows 10/11, 64 бита.\n2. Распакуйте через «Извлечь всё» в отдельную папку, например C:\\CarVision211.\n3. Откройте полученную папку CarVision и запустите CarVision.exe.\n4. В заголовке должно быть CarVision 2.11 TEST.\n5. Открыть фото -> Анализировать фото.\n\nЗапускать EXE прямо внутри ZIP нельзя. Переносите всю папку: рядом с EXE должны оставаться models и _internal. Старые патчи устанавливать не нужно. С предыдущими версиями файлы не смешивайте. Python, pip, CMD и интернет для обычного анализа не требуются. Дополнительная техническая проверка — CHECK_ON_NEW_PC.cmd. PASS означает работу EXE, весов, анализа, выделения области и экспорта, а не правильность всех ответов. Второй физический компьютер пока не проверен.\n\nИнструкции: CarVision/GUIDE_2_11_RU.txt. Разбор оставшихся ошибок: CarVision/REVIEW_2_11_RU.txt.\n\nЭто версия TEST с частичным улучшением: Zeekr, Leapmotor и BYD на известных новых примерах. AIQAR, Voyah и белый Tracker в разобранных случаях ещё не исправлены; конкретная модель Zeekr007 не обучена. Новые примеры использованы как TRAIN, не независимый экзамен. Цвет может ошибаться. Все325 моделей и100% точность не достигнуты.\n'
    first=APP.parent/'READ_FIRST_RU.txt';write(first,intro)
    excluded={'outputs','patch-backups','__pycache__','.git'};files=sorted(p for p in APP.rglob('*') if p.is_file() and not any(x in excluded for x in p.relative_to(APP).parts) and p.name not in ('TRANSFER_CONTENTS.json','SHA256SUMS.txt') and not p.name.endswith('.seal211-tmp'));hashes={}
    for i,p in enumerate(files):
        hashes[p.relative_to(APP).as_posix()]=sha(p)
        if i%1000==0:print('HASH',i,len(files),flush=True)
    manifest=dict(version='2.11 TEST',files=hashes,scope='Full offline Windows x64 transfer, outputs/cache excluded. Recognition remains experimental.');write(APP/'TRANSFER_CONTENTS.json',json.dumps(manifest,ensure_ascii=False,indent=2));write(APP/'SHA256SUMS.txt',''.join(h+'  '+n+'\n' for n,h in hashes.items()));archive=ROOT/'outputs/CarVision-2.11-Test-Full.zip';assert not archive.exists(),'Never overwrite a sealed archive';temp=archive.with_suffix('.zip.tmp');print('PACK',len(files),sum(p.stat().st_size for p in files),flush=True)
    extras=[APP/'TRANSFER_CONTENTS.json',APP/'SHA256SUMS.txt'];allfiles=files+extras
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,p in enumerate(allfiles):
            z.write(p,'CarVision/'+p.relative_to(APP).as_posix())
            if i%1000==0:print('ZIP',i,len(allfiles),flush=True)
        z.write(first,'READ_FIRST_RU.txt')
    os.replace(temp,archive)
    with zipfile.ZipFile(archive) as z:
        for i,(name,expected) in enumerate(hashes.items()):
            h=hashlib.sha256()
            with z.open('CarVision/'+name) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==expected,name
            if i%1000==0:print('VERIFY',i,len(hashes),flush=True)
        assert json.loads(z.read('CarVision/TRANSFER_CONTENTS.json'))==manifest
        assert z.read('CarVision/SHA256SUMS.txt')==(APP/'SHA256SUMS.txt').read_bytes()
        assert z.read('READ_FIRST_RU.txt')==first.read_bytes()
    result=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),payload_files=len(hashes),all_payload_hashes_verified=True,all_zip_entries_crc_checked=True,exe_sha256=stage['exe_sha256'],model_sha256=stage['model_sha256'],version='2.11 TEST',second_physical_pc_tested=False)
    write(ROOT/'outputs/seal-v211.json',json.dumps(result,indent=2));write(archive.with_suffix('.zip.sha256'),result['sha256']+'  '+archive.name+'\n');print('SEALED',json.dumps(result),flush=True)
if __name__=='__main__':main()
