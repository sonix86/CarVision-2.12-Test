"""Build a small, version-checked model-only update for the already delivered EXE."""
from pathlib import Path
import json,hashlib,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];candidate=ROOT/'models/ev_patch_v22'
training=json.loads((candidate/'training.json').read_text(encoding='utf-8'));evaluation=json.loads((candidate/'evaluation.json').read_text(encoding='utf-8'));dataset=json.loads((candidate/'dataset.json').read_text(encoding='utf-8'))
if not training['selected']:raise SystemExit('Validation rejected candidate')
historical=evaluation['results']['historical'];overall=evaluation['results']['all'];evcheck=evaluation['results']['ev']
if (historical['candidate']['top1'] < historical['baseline']['top1']-.01
        or overall['candidate']['top1'] < overall['baseline']['top1']
        or evcheck['candidate']['top1'] < evcheck['baseline']['top1']+.03):
    raise SystemExit('Post-selection test regression: do not release or pick a different epoch using test results')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'models/vehicle_model.pt')==evaluation['baseline_sha256']
assert sha(candidate/'vehicle_model.pt')==evaluation['candidate_sha256']
stage=ROOT/'outputs/CarVision-EV-Patch-2.2';archive=ROOT/'CarVision-EV-Patch-2.2.zip'
if stage.exists() or archive.exists():raise SystemExit('Existing patch release; inspect before replacing')
(stage/'payload/models').mkdir(parents=True)
files=[]
for name in ['vehicle_model.pt','visual_references.pt']:
    source=candidate/name;destination=stage/'payload/models'/name;shutil.copy2(source,destination)
    files.append(dict(path='models/'+name,sha256=sha(destination),previous_sha256=sha(ROOT/'models'/name)))
primary=['classifier.pt','detector.pt','colour_model.pt','vehicle_model.pt','segmenter.pt','masked_colour.pt','visual_references.pt','generic_brand_backbone.pt','generic_detector_backbone.pt']
active={name:sha(candidate/name if name in ['vehicle_model.pt','visual_references.pt'] else ROOT/'models'/name) for name in primary}
meta=stage/'payload/Development/ACTIVE_MODEL_SHA256.json';meta.parent.mkdir(parents=True);meta.write_text(json.dumps(active,indent=2),encoding='utf-8')
files.append(dict(path='Development/ACTIVE_MODEL_SHA256.json',sha256=sha(meta),previous_sha256=None))
for name in ['APPLY_EV_PATCH.ps1','APPLY_EV_PATCH.cmd']:shutil.copy2(ROOT/'handoff'/name,stage/name)
manifest=dict(release='EV weights patch 2.2 for CarVision 2.1',exe_sha256=sha(ROOT/'dist/CarVision/CarVision.exe'),files=files,baseline_model_sha256=evaluation['baseline_sha256'],candidate_model_sha256=evaluation['candidate_sha256'])
(stage/'PATCH_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for name in ['dataset.json','training.json','evaluation.json','data_audit.json','review_exclusions.json']:shutil.copy2(candidate/name,stage/name)
(stage/'Development/tools').mkdir(parents=True)
for name in ['collect_ev_patch_v22.py','collect_ev_aliases_v22.py','collect_ev_flickr_v22.py','prepare_patch_v22.py','audit_ev_patch_v22.py','cache_patch_baseline_v22.py','train_patch_v22.py','evaluate_patch_v22.py','package_ev_patch_v22.py']:
    shutil.copy2(ROOT/'tools'/name,stage/'Development/tools'/name)
credits=[]
for folder in ['ev_patch_v22','ev_patch_flickr_v22']:
    p=ROOT/'datasets'/folder/'manifest.json'
    if p.exists():credits+=json.loads(p.read_text(encoding='utf-8'))
(stage/'PHOTO_CREDITS.json').write_text(json.dumps(credits,ensure_ascii=False,indent=2),encoding='utf-8')
names=', '.join(' '.join(p) for p in dataset['new_pairs']) or 'список прежний; обновлены данные существующих моделей'
catalog=['ИЗУЧЕННЫЕ СЕМЕЙСТВА МОДЕЛЕЙ', 'Числа: обучение / validation / test. Наличие в списке не гарантирует распознавание.', '']
for name,counts in sorted(dataset['counts'].items()):
    catalog.append(f"{name}: {counts['train']} / {counts['validation']} / {counts['test']}")
(stage/'СПИСОК_МОДЕЛЕЙ.txt').write_text('\n'.join(catalog)+'\n',encoding='utf-8-sig')
ev=evaluation['results']['ev'];old=evaluation['results']['historical']
text=f'''CARVISION — ПАТЧ МОДЕЛЕЙ 2.2 ДЛЯ ПРОГРАММЫ 2.1
Владелец: София Рахманкулова

УСТАНОВКА НА ДРУГОМ КОМПЬЮТЕРЕ
1. Полностью закройте CarVision.
2. Распакуйте этот ZIP в отдельную папку. Не запускайте файлы прямо из ZIP.
3. Дважды нажмите APPLY_EV_PATCH.cmd.
4. В окне выбора папки укажите существующую папку CarVision, где лежит CarVision.exe.
5. Дождитесь сообщения Patch installed. Затем запустите свой CarVision.exe обычным способом.
Python, интернет, ключи, оплата и повторное обучение не нужны.
Патч меняет models/vehicle_model.pt, models/visual_references.pt и технический список хешей Development/ACTIVE_MODEL_SHA256.json.
EXE и папка _internal остаются прежними. В интерфейсе может остаться версия приложения 2.1.
Не заменяйте только один из двух файлов: веса и банк сравнения согласованы контрольной суммой.
Фотографии, разметка и пользовательские результаты не изменяются.

ОТКАТ
В папке приложения появится patch-backups/ev22-ДАТА-ВРЕМЯ/models.
Закройте программу и скопируйте ОБА файла из этой папки обратно в основную models с заменой.
Также восстановите Development/ACTIVE_MODEL_SHA256.json из той же резервной копии, если он там есть.
Если установщик отказал из-за отличающейся модели или версии EXE, не обходите проверку:
используйте чистую копию ранее переданного CarVision-2.1-Full-Transfer.zip.

ЧТО ОБУЧЕНО
Всего семейств моделей: {len(dataset['pairs'])}.
Новые семейства: {names}.
Список и размеры обучающей/validation/test выборок: dataset.json.
Данные — реальные фотографии с сохранёнными источниками/лицензиями PHOTO_CREDITS.json.
Классификатор обучен в проекте; готовый автомобильный классификатор не скачивался.
Телефонные условия имитируются затемнением, размытием, уменьшением и перекрытиями;
это не заменяет независимую проверку на реальных новых фотографиях.
Ночная фотография пользователя не использовалась для обучения или выбора весов.

ИЗМЕРЕНИЯ
EV, отложенные фото ({ev['count']}): top-1 {ev['baseline']['top1']:.1%} -> {ev['candidate']['top1']:.1%}.
Исторические отложенные фото ({old['count']}): top-1 {old['baseline']['top1']:.1%} -> {old['candidate']['top1']:.1%}.
Полные результаты, включая слабые классы: evaluation.json.
Результаты относятся к корпусным фрагментам, не являются точностью всей сцены/детектора.
Названия конкретных моделей — дополнительные возможности задания; обязательны марка и цвет.
Не все автомобили мира входят в обучение. Знак вопроса означает гипотезу.
Проценты — оценки модели, не гарантия истинности; тёмный кадр/блики/малые объекты могут ошибаться.
Цвет и нейросеть детектора в этом патче не переобучались.

ПРОВЕРКА ПОСЛЕ УСТАНОВКИ
Откройте свои новые фотографии, выполните анализ всего кадра и выделенной машины.
Проверьте экспорт. Пробуйте независимые снимки, а не только встроенные примеры.
CHECK_ON_NEW_PC.cmd можно использовать после обновления: технический список хешей обновлён.
Контрольные суммы новых файлов также находятся в PATCH_MANIFEST.json.
В обучение добавлены 8 размеченных фрагментов собственных фотографий из одной съёмки;
этого недостаточно для представительной собственной базы. Прочие незакрытые требования:
реальная независимая телефонная проверка, подтверждение уникальности архитектуры, GitHub/отчёт.
'''
(stage/'ПРОЧИТАЙ_УСТАНОВКА.txt').write_text(text,encoding='utf-8-sig')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(stage.rglob('*')):
        if p.is_file():z.write(p,stage.name+'/'+p.relative_to(stage).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for r in files:assert hashlib.sha256(z.read(stage.name+'/payload/'+r['path'])).hexdigest()==r['sha256']
archive.with_suffix('.zip.sha256.txt').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
print(archive,archive.stat().st_size)
