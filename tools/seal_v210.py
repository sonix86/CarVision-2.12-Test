"""Seal a separate full transfer; ground truth is reviewed labels, never predictions."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'outputs/CarVision-2.10-Test/CarVision'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,s):
    t=p.with_name(p.name+'.seal-tmp');t.write_text(s,encoding='utf-8');os.replace(t,p)
def iou(a,b):
    area=lambda z:max(0,z[2]-z[0])*max(0,z[3]-z[1])
    x=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]);return x/max(1,area(a)+area(b)-x)
def main():
    assert read(APP/'Development/FIX_2_10/RECIPIENT_VERIFICATION.json')['passed']
    old=read(ROOT/'outputs/mashinki-v29/candidate_summary.json');new=read(ROOT/'outputs/mashinki-v210/candidate_summary.json');assert old['complete'] and new['complete']
    excluded=read(ROOT/'models/owner_domain_v29/review_approval.json')['exclusions'];all_labels=read(ROOT/'models/owner_domain_v29/annotations.json')
    labels=[dict(r) for i,r in enumerate(all_labels) if str(i) not in excluded]
    for i in (15,16):
        r=dict(all_labels[i]);r['brand']='Dongfeng';r['colour']='white';r['label_source']='Author-labelled licensed E70 sources and manual visible emblem/body comparison, not model output';labels.append(r)
    assert len(labels)==41
    banks=[{Path(r['source']).name:r for r in d['records']} for d in (old,new)];rows=[]
    for r in labels:
        item=dict(file=Path(r['source']).name,truth_brand=r['brand'],truth_colour=r['colour'],role='Known TRAIN/development scene; not blind accuracy')
        for key,bank in zip(('v2_9','v2_10'),banks):
            car=max(bank[item['file']]['cars'],key=lambda c:iou(c['box'],r['box']),default=None)
            ok=car is not None and iou(car['box'],r['box'])>=.3
            item[key]=dict(matched=ok,brand=car['brand'] if ok else None,model=car.get('model') if ok else None,colour=car['colour'] if ok else None)
            item[key]['brand_correct']=ok and car['brand']==r['brand'];item[key]['colour_correct']=ok and car['colour']==r['colour']
        rows.append(item)
    counts={k:{m:sum(r[k][m] for r in rows) for m in ('matched','brand_correct','colour_correct')} for k in ('v2_9','v2_10')}
    report=dict(processed_photos=len(new['records']),reviewed_objects=41,counts=counts,records=rows,scope='TRAIN/development diagnostics only; historical 35/39 omitted first two and assessed make only. No new independent phone accuracy. First E70, Dolphin and Song Plus now trained. Second E70 view is same car, not blind.')
    write(APP/'Development/FIX_2_10/OWN_REVIEWED_COMPARISON.json',json.dumps(report,ensure_ascii=False,indent=2))
    text='CarVision 2.10 TEST — проверка исправления\n\n'
    text+='В 2.9 были реальные ошибки. Прежние 35/39 — проверка марки на учебных фрагментах, без первых двух спорных фото и без оценки конкретной модели. Это не доказательство готовности.\n\n'
    text+='Проверен настоящий новый EXE, без Python в PATH: два ракурса белого Dongfeng/Aeolus E70, BYD Dolphin, BYD Song Plus; дополнительно Haval и Toyota. Ответы EXE совпали с исходным кодом. Переименование E70 и Song Plus не изменило ответы. Выделение области и экспорт прошли техническую проверку.\n\n'
    text+='Первый седан: независимые авторские фотографии E70 совпадают по эмблеме, кузову и решётке. Старую подпись GAC Aion S Plus исключили. Это визуальная проверка по источникам, а не подтверждение по VIN.\nИсточник: https://commons.wikimedia.org/wiki/File:Aeolus_E70_front,_Dongguan,_China_29_April_2024.jpg\n\n'
    text+='Все 99 файлов обработаны. На 41 размеченном объекте (учебные/известные сцены):\n'
    for key,c in counts.items():text+=f"{key}: верная подтверждённая марка {c['brand_correct']}/41; цвет {c['colour_correct']}/41.\n"
    text+='\nE70 на первом фото, Dolphin и Song Plus добавлены в обучение. Второй ракурс E70 не использован в этом дообучении, но это та же машина. Эти примеры не считаются независимой точностью. Проценты программы не гарантируют правильность. Цвет Dolphin по-прежнему ошибочный; ложные контуры и unknown остаются. Все 325 моделей не обучены.\n\n'
    text+='Для ручной проверки откройте CarVision.exe из этой папки: заголовок CarVision 2.10 TEST. Открыть фото → Анализировать фото. Старое окно 2.9 может оставаться открытым: проверяйте заголовок. Переносите весь новый архив в отдельную папку; Python, CMD и интернет для приложения не нужны.\n'
    write(APP/'RESULTS_2_10_RU.txt',text)
    write(ROOT/'outputs/CarVision-2.10-Test/READ_FIRST_RU.txt',text)
    excluded_parts={'outputs','patch-backups','__pycache__','.git'}
    files=sorted(p for p in APP.rglob('*') if p.is_file() and not any(x in excluded_parts for x in p.relative_to(APP).parts) and p.name not in ('TRANSFER_CONTENTS.json','SHA256SUMS.txt') and not p.name.endswith('.seal-tmp'))
    hashes={p.relative_to(APP).as_posix():sha(p) for p in files}
    write(APP/'TRANSFER_CONTENTS.json',json.dumps(dict(version='2.10 TEST',files=hashes,scope='Full standalone Windows transfer; output/cache excluded'),ensure_ascii=False,indent=2))
    write(APP/'SHA256SUMS.txt',''.join(h+'  '+n+'\n' for n,h in hashes.items()))
    archive=ROOT/'outputs/CarVision-2.10-Test-Full.zip';temp=archive.with_suffix('.zip.tmp');total=sum(p.stat().st_size for p in files);print('SEAL',len(files),'files',total,'bytes',counts,flush=True)
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for index,p in enumerate(files+[APP/'TRANSFER_CONTENTS.json',APP/'SHA256SUMS.txt']):
            z.write(p,'CarVision/'+p.relative_to(APP).as_posix())
            if index%250==0:print('ZIP',index,flush=True)
        z.write(ROOT/'outputs/CarVision-2.10-Test/READ_FIRST_RU.txt','READ_FIRST_RU.txt')
    os.replace(temp,archive)
    with zipfile.ZipFile(archive) as z:
        for index,(rel,expected) in enumerate(hashes.items()):
            h=hashlib.sha256()
            with z.open('CarVision/'+rel) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==expected,rel
            if index%350==0:print('VERIFY ZIP',index,flush=True)
        assert json.loads(z.read('CarVision/TRANSFER_CONTENTS.json'))['files']==hashes
    result=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(hashes),payload_hashes_verified=True,zip_crc_checked_by_full_read=True,counts=counts)
    write(ROOT/'outputs/seal-v210.json',json.dumps(result,indent=2));write(archive.with_suffix('.zip.sha256'),result['sha256']+'  '+archive.name+'\n');print('SEALED',json.dumps(result),flush=True)
if __name__=='__main__':main()
