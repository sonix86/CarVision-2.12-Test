"""Create a distinct local TEST application; preserve sealed 2.10 unchanged."""
import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def replace(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.v211-tmp');shutil.copy2(src,tmp);os.replace(tmp,dst)
def write(dst,s):
    dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.v211-tmp');tmp.write_text(s,encoding='utf-8-sig' if dst.suffix=='.txt' else 'utf-8');os.replace(tmp,dst)
def main():
    assert json.loads((ROOT/'models/phone_v32/regression.json').read_text())['passed'];assert json.loads((ROOT/'outputs/user-error-audit-v32/runtime/summary.json').read_text())['passed_required_controls'];base=ROOT/'outputs/CarVision-2.10-Test/CarVision';dest=ROOT/'outputs/CarVision-2.11-Test/CarVision';build=ROOT/'outputs/build-v211/dist/CarVision';assert (build/'CarVision.exe').is_file() and not dest.exists();old=json.loads((base/'TRANSFER_CONTENTS.json').read_text(encoding='utf-8'))['files'];before={n:sha(base/n) for n in ('CarVision.exe','models/dino_make.pt','models/model_version.json')}
    shutil.copytree(base,dest,copy_function=os.link,ignore=shutil.ignore_patterns('outputs','patch-backups','__pycache__','TRANSFER_CONTENTS.json','SHA256SUMS.txt'))
    for f in build.rglob('*'):
        if f.is_file():
            rel=f.relative_to(build).as_posix()
            if old.get(rel)!=sha(f):replace(f,dest/rel)
    replace(ROOT/'models/phone_v32/scoped_candidate.pt',dest/'models/dino_make.pt');modelsha=sha(dest/'models/dino_make.pt');version=dict(version='2.11 TEST',verify_active_head=True,selected_model_sha256=modelsha,make_labels=82,model_families=43,make_correction_brands=['Zeekr','Leapmotor','BYD'],all325_models_trained=False,quality_goal_achieved=False,known_new_cases_now_train=True,unresolved=['AIQAR absent from taxonomy','Voyah confirmation','White Tracker in garage','Exact Zeekr007 model','Paint and other existing failures'])
    write(dest/'models/model_version.json',json.dumps(version,ensure_ascii=False,indent=2));active=json.loads((base/'Development/ACTIVE_MODEL_SHA256.json').read_text(encoding='utf-8'));active['dino_make.pt']=modelsha;write(dest/'Development/ACTIVE_MODEL_SHA256.json',json.dumps(active,indent=2))
    for folder in ('carvision','tools','tests'):
        for f in (ROOT/folder).glob('*.py'):replace(f,dest/folder/f.name)
    for f in (ROOT/'models/phone_v32').glob('*.json'):replace(f,dest/'Development/FIX_2_11'/f.name)
    replace(ROOT/'outputs/user-error-audit-v32/runtime/summary.json',dest/'Development/FIX_2_11/runtime_summary.json')
    guide='CarVision 2.11 TEST — частичное исправление новых ошибок\n\nЗапуск: CarVision.exe в этой папке. Заголовок окна должен быть 2.11 TEST. Интернет, Python и CMD для анализа не нужны.\n\nДобавлена собственная дополнительная модель марки только для Zeekr, Leapmotor, BYD. Общие DINO/CLIP признаки и прежние решения остальных марок сохранены. Новые фото стали TRAIN-исправлениями; это не независимая точность.\nДве полные замены головы отклонены: ухудшали другие решения. Узкое дополнение откалибровано по глобальному баллу, разнице двух лучших классов и геометрии; минимум3 группы источников на проверке. Пороги вручную не снижались. Все неудачные протоколы сохранены.\n\nПовторный публичный тест разработки942: верных подтверждений марки623->626 на обычных,478->478 на искусственно уменьшенных. Ошибочных подтверждений осталось23 и24 соответственно. Это повторно использованный тест, не новый слепой экзамен.\n\nНа новых известных примерах исправлены марка Zeekr, Leapmotor и BYD слева в гараже. Точная модель Zeekr007 отсутствует в43 семействах — её название не подставляется принудительно. AIQAR на третьем фото отсутствует в82 марках; по официальному каталогу есть AIQAR Arrizo e. Случай не переименован в Chery/Hyundai. Voyah по центру и белый Tracker справа в гараже ещё не исправлены. Цвет и контуры не переобучены. У левого края гаража есть ещё отдельная частично скрытая машина; её контур нельзя сливать с BYD.\n\nВерсия экспериментальная, все325 моделей и100% точность не достигнуты. 2.10 и её ZIP сохранены отдельно. Открыть фото -> Анализировать фото. Проверяйте сначала марку, затем модель и цвет.\nИсточник AIQAR: https://www.aiqar.am/arrizo\n'
    write(dest/'GUIDE_2_11_RU.txt',guide);write(dest/'GUIDE_FULL_2_6_RU.txt',guide)
    identity=dict(target=str(dest),exe_sha256=sha(dest/'CarVision.exe'),model_sha256=modelsha,base210_preserved={n:sha(base/n)==h for n,h in before.items()},not_final_quality=True)
    assert all(identity['base210_preserved'].values());write(ROOT/'outputs/stage-v211.json',json.dumps(identity,indent=2));write(dest/'BUILD_2_11.json',json.dumps(identity,indent=2));print(json.dumps(identity),flush=True)
if __name__=='__main__':main()
