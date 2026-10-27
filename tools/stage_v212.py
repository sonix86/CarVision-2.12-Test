"""Separate audit build; unchanged 2.11 inference weights, corrected runtime and raw data."""
import json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.stage_v211 import sha,replace,write
def main():
    base=ROOT/'outputs/CarVision-2.11-Test/CarVision';dest=ROOT/'outputs/CarVision-2.12-Test/CarVision';build=ROOT/'outputs/build-v212/dist/CarVision';assert (build/'CarVision.exe').exists() and not dest.exists();old=json.loads((base/'TRANSFER_CONTENTS.json').read_text(encoding='utf-8'))['files'];before={n:sha(base/n) for n in ('CarVision.exe','models/dino_make.pt','models/model_version.json')}
    shutil.copytree(base,dest,copy_function=os.link,ignore=shutil.ignore_patterns('outputs','patch-backups','__pycache__','TRANSFER_CONTENTS.json','SHA256SUMS.txt'))
    for f in build.rglob('*'):
        if f.is_file():
            rel=f.relative_to(build).as_posix()
            if old.get(rel)!=sha(f):replace(f,dest/rel)
    for folder in ('carvision','tools','tests'):
        for f in (ROOT/folder).glob('*.py'):replace(f,dest/folder/f.name)
    replace(ROOT/'data/owner_oct25/labels.jsonl',dest/'data/owner_oct25/labels.jsonl')
    for f in (ROOT/'data/reviewed_owner_v212').rglob('*'):
        if f.is_file():replace(f,dest/'data/reviewed_owner_v212'/f.relative_to(ROOT/'data/reviewed_owner_v212'))
    version=json.loads((base/'models/model_version.json').read_text(encoding='utf-8'));version.update(version='2.12 TEST',inference_weights_unchanged_from='2.11 TEST',audit='Correct make head in box-detector path; colour loading/API; data integrity; paired encoder SHA check. No new recognition-accuracy claim.');write(dest/'models/model_version.json',json.dumps(version,ensure_ascii=False,indent=2))
    for f in (ROOT/'outputs/pipeline-audit-v212').glob('*'):
        if f.is_file():replace(f,dest/'Development/FIX_2_12'/f.name)
    for f in ('audit-regressions-before.log','audit-regressions-after.log','tests-audit-v212-final.log'):replace(ROOT/'outputs'/f,dest/'Development/FIX_2_12'/f)
    guide='CarVision 2.12 TEST — проверка кода и данных\n\nЗапуск: CarVision.exe из этой папки. Заголовок 2.12 TEST. Веса распознавания те же, что в2.11; нового прироста точности или80% не заявляется. Старые версии и ZIP сохранены.\n\nИсправлены: выбор обновлённого классификатора при отсутствии сегментации; загрузка отдельной модели цвета по умолчанию; вызов общей модели цвета без сегментации; запрет противоречащих подписей одной области; группировка одинаковых пикселей с разными метаданными в один раздел; проверка парного CLIP-кодировщика по сохранённому SHA.\n\n149 программных тестов прошли. До исправления5 воспроизведений ошибок падали, после исправления прошли. В текущем кэше3851 фрагмента не найдены утечки по группам, исходным SHA и одинаковым пикселям с маской. Парные веса соответствуют.8 пересчитанных признаков совпали с кэшем (ошибка менее0.000001). Это структурная проверка, не визуальная сертификация всех3851 подписей.\n\nУ старого сырого экспорта исправлены подпись Chevrolet вместо Kia и область соседней Kia по ранее вручную проверенной разметке. Создан data/reviewed_owner_v212/labels.jsonl с доступными проверенными TRAIN-примерами. Недоступные оригиналы перечислены в Development/FIX_2_12/label_repairs.json. Этот набор не независимый тест и сам по себе не обновляет веса.\n\nТекущие ошибки распознавания AIQAR, Voyah, белого Tracker, моделиZeekr007 и части цветов сохраняются. Архив2.11неизменён; улучшения его обычных ответов от этого аудита не обещаются. Для роста качества нужны новые проверенные местные данные и обучение, а не только исправление запасных веток.\n'
    write(dest/'GUIDE_2_12_RU.txt',guide);write(dest/'GUIDE_FULL_2_6_RU.txt',guide)
    identity=dict(target=str(dest),exe_sha256=sha(dest/'CarVision.exe'),model_sha256=sha(dest/'models/dino_make.pt'),weights_unchanged=sha(dest/'models/dino_make.pt')==before['models/dino_make.pt'],base211_preserved={n:sha(base/n)==h for n,h in before.items()});assert identity['weights_unchanged'] and all(identity['base211_preserved'].values());write(ROOT/'outputs/stage-v212.json',json.dumps(identity,indent=2));write(dest/'BUILD_2_12.json',json.dumps(identity,indent=2));print(json.dumps(identity),flush=True)
if __name__=='__main__':main()
