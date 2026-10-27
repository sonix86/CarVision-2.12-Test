"""Stage a checked EXE/runtime/weights update for the delivered 2.1/2.2 folder."""
import collections,hashlib,json,shutil,zipfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'models/make_scope_v23'
sys.path.insert(0,str(ROOT))
stage=ROOT/'outputs/CarVision-Make-Update-2.3';payload=stage/'payload'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
training=load(OUT/'training.json');evaluation=load(OUT/'evaluation.json');dataset=load(OUT/'dataset.json');audit=load(OUT/'data_audit.json')
assert audit['ready_for_training'] and audit['dataset_sha256']==training['dataset_sha256']==evaluation['dataset_sha256']
assert sha(OUT/'make_expert.pt')==evaluation['candidate_sha256']
assert load(ROOT/'outputs/frozen-v23-check.json')['passed']
r=evaluation['results'];old=r['historical']['whole'];all_result=r['all']['whole'];new=r['target_scope_new']['whole']
guards=dict(historical_top1=old['top1']>=old['baseline_body_top1']-.02,
            all_top1=all_result['top1']>=all_result['baseline_body_top1']+.03,
            confirmed_precision=all_result['confirmed']>=30 and all_result['confirmed_precision']>=.9,
            confirmed_coverage=all_result['coverage']>=.1,
            new_make_top1=new['count']>=50 and new['top1']>=.5,
            new_confirmed_precision=new['confirmed']>=10 and new['confirmed_precision']>=.9,
            priority_makes=r['priority_byd_li_zeekr']['whole']['top1']>=r['priority_byd_li_zeekr']['whole']['baseline_body_top1']-.02)
(OUT/'release_gate.json').write_text(json.dumps(dict(passed=all(guards.values()),checks=guards,policy='Post-selection stop gates; never choose another epoch using test results.'),indent=2),encoding='utf-8')
assert all(guards.values()),'Heldout release gate rejected the update; stop, do not cherry-pick another epoch.'
assert not stage.exists(),'Existing stage must be inspected, not overwritten'
payload.mkdir(parents=True)
original=zipfile.ZipFile(ROOT/'CarVision-2.1-Full-Transfer.zip');original_names=set(original.namelist());files=[]
recipient=ROOT/'outputs/Transfer extracted София/CarVision'
def add(path,source=None,content=None,required=None,extra_old=()):
    assert source is not None or content is not None
    dest=payload/path;dest.parent.mkdir(parents=True,exist_ok=True)
    if source is not None:shutil.copy2(source,dest)
    else:dest.write_text(content,encoding='utf-8')
    entry='CarVision/'+path;previous=[]
    if entry in original_names:previous.append(hashlib.sha256(original.read(entry)).hexdigest())
    previous+=list(extra_old)
    critical=path=='CarVision.exe' or path.startswith('_internal/') or path in ['models/vehicle_model.pt','models/visual_references.pt']
    files.append(dict(path=path,sha256=sha(dest),previous_sha256=sorted(set(previous)),require_existing=(critical and entry in original_names) if required is None else required))
build=ROOT/'outputs/build-v23/dist/CarVision'
add('CarVision.exe',source=build/'CarVision.exe')
changed=[];required_files=[]
for p in sorted((build/'_internal').rglob('*')):
    if not p.is_file():continue
    relative=p.relative_to(build).as_posix();other=recipient/relative
    if not other.exists() or sha(other)!=sha(p):
        add(relative,source=p,required=other.exists());changed.append(relative)
    else:required_files.append(dict(path=relative,sha256=sha(p)))
for name in ['vehicle_model.pt','visual_references.pt']:
    add('models/'+name,source=ROOT/'models/ev_patch_v22'/name,extra_old=[sha(ROOT/'models/ev_patch_v22'/name)])
add('models/make_expert.pt',source=OUT/'make_expert.pt')
active=load(recipient/'Development/ACTIVE_MODEL_SHA256.json')
active['make_expert.pt']=sha(OUT/'make_expert.pt')
required_files += [dict(path='models/'+name,sha256=digest) for name,digest in active.items() if name not in ['vehicle_model.pt','visual_references.pt','make_expert.pt']]
add('Development/ACTIVE_MODEL_SHA256.json',content=json.dumps(active,indent=2),extra_old=[sha(recipient/'Development/ACTIVE_MODEL_SHA256.json')])
catalogue=load(ROOT/'data/catalogue/ev_target_catalog.json')
from tools.make_taxonomy_v23 import ALIASES,DISPLAY,canonical_make
requested=sorted({e['brand'] for e in catalogue['entries']});trained=dataset['brands']
catalog=dict(version='2.3',requested_brands=requested,aliases=ALIASES,display_names=DISPLAY,
             learned_make_labels=trained,confirmed_gate_enabled=[b for b,v in training['calibration'].items() if v['threshold']<1],
             counts=dataset['counts'],requested_entries=catalogue['entries'],
             not_trained_requested_makes=[b for b in requested if canonical_make(b) not in trained],
             exact_model_decoder_classes=42,limitation='Requested names are a catalogue, not trained model classes. A make label without heldout evidence is provisional and cannot be confirmed.')
add('models/make_catalogue.json',content=json.dumps(catalog,ensure_ascii=False,indent=2))
checker=(ROOT/'handoff/CHECK_ON_NEW_PC.ps1').read_text(encoding='utf-8').replace("version='2.1'","version='2.3'")
add('CHECK_ON_NEW_PC.ps1',content=checker)
add('CHECK_ON_NEW_PC.cmd',source=ROOT/'handoff/CHECK_ON_NEW_PC.cmd')
for p in sorted((ROOT/'carvision').glob('*.py')):add(p.relative_to(ROOT).as_posix(),source=p)
toolnames=['prepare_make_scope_v23','make_taxonomy_v23','make_scope_features_v23','train_make_scope_v23','finetune_make_scope_v23','evaluate_make_scope_v23','audit_make_scope_v23','freeze_make_family_probe_v23','collect_make_scope_v23','collect_make_flickr_v23','cache_make_scope_v23','check_frozen_v23','package_make_scope_v23','verify_make_update_v23','build_ev_target_catalog','audit_ev_catalog_sources','resolve_make_category_aliases_v23']
for name in toolnames:
    p=ROOT/'tools'/(name+'.py')
    if p.exists():add(p.relative_to(ROOT).as_posix(),source=p)
for p in sorted((ROOT/'tests').glob('test_*.py')):add(p.relative_to(ROOT).as_posix(),source=p)
for p in sorted((ROOT/'handoff').glob('*')):
    if p.is_file():add(p.relative_to(ROOT).as_posix(),source=p)
for name in ['dataset.json','data_audit.json','review_approval.json','reviewed_images.json','review_exclusions.json','review_relabels.json','manual_boxes.json','family_probe.json','predeclared_release_policy.json','training.json','evaluation.json','release_gate.json']:
    p=OUT/name
    if p.exists():add('Development/TrainingInputs/'+name,source=p)
for folder in ['make_scope_v23','make_flickr_v23']:
    add('Development/TrainingInputs/'+folder+'_PHOTO_CREDITS.json',source=ROOT/'datasets'/folder/'manifest.json')
for name in ['manifest.json','STATUS.txt','labels.jsonl']:
    owner_file=ROOT/'data/owner_oct25'/name
    if owner_file.exists():add('Development/TrainingInputs/owner_oct25/'+name,source=owner_file)
add('data/catalogue/ev_target_catalog.json',source=ROOT/'data/catalogue/ev_target_catalog.json')
make_count=len(trained);confirmed=len(catalog['confirmed_gate_enabled']);missing=', '.join(catalog['not_trained_requested_makes']) or 'нет'
guide=f'''CarVision 2.3 — обновление определения марки
Владелец: София Рахманкулова

УСТАНОВКА НА ДРУГОМ КОМПЬЮТЕРЕ
1. Полностью закройте CarVision.
2. Распакуйте ЭТОТ ZIP в отдельную папку. Не копируйте его поверх программы вручную.
3. Запустите APPLY_UPDATE_2_3.cmd из распакованной папки обновления.
4. В появившемся окне выберите СТАРУЮ папку CarVision, где уже лежит CarVision.exe.
5. Дождитесь сообщения Update 2.3 installed successfully.
6. Запустите CarVision.exe из старой папки обычным способом.
Установщик сам заменит EXE, согласованные зависимости и обученные веса; предыдущие файлы сохранит в patch-backups/update23-ДАТА.
Поддерживается ранее переданная полная версия 2.1 и её обновление весов 2.2. Ваши фотографии, разметка и результаты сохраняются.
Python, интернет, ключи, оплата и повторное обучение для запуска не нужны. Windows x64.
Это обновление НЕ является самостоятельной полной программой. Для нового компьютера сначала распакуйте CarVision-2.1-Full-Transfer.zip, затем примените данный патч.
Если установщик обнаружит изменённые пользователем файлы, он остановится до замены. Не обходите проверку; сохраните свою работу и используйте чистую копию полной версии.

ПРОВЕРКА
При желании запустите CHECK_ON_NEW_PC.cmd из папки программы: он проверяет веса, анализ, выделение зоны, экспорт и отдельное учебное обучение без изменения активной модели.
Подробности сохраняются в outputs/transfer-selfcheck. Это техническая проверка, а не гарантия точности каждой марки.
Для анализа: Открыть фото → Анализировать фото. Для отдельной машины: Выбрать зону мышью → обвести машину → Анализировать фото.
Нажмите контур машины или строку результата. Сначала показывается марка. Модель появляется только при достаточном подтверждении из изученных 42 семейств.
«Возможно … / марка не подтверждена» означает предположение. Проценты — оценки сходства, не гарантии правильности.

ЧТО ИЗМЕНИЛОСЬ
Обучен отдельный классификатор марки: {make_count} меток. Калиброванный вывод марки включён для {confirmed}; остальные могут появляться только как варианты.
Модель из другой марки больше не заменяет подтверждённую марку. Неопределённая модель явно подписывается.
Каталог охватывает 325 запрошенных записей; это НЕ 325 обученных классов модели. Точный декодер моделей остаётся на 42 семействах.
Запрошенные марки без обученного класса: {missing}.
Детектор/сегментация и классификатор цвета в этом патче не переобучались. Очень маленькие, перекрытые, ночные и неизвестные машины могут определяться неверно.
Нельзя обещать 100% на любых фотографиях или достоверно различать внешне одинаковые комплектации по одному кадру.

ПРОВЕРЕННЫЕ ИЗМЕРЕНИЯ
Отложенные корпусные фрагменты: {all_result['count']}; top-1 марки {all_result['baseline_body_top1']:.1%} → {all_result['top1']:.1%}.
Подтверждённая марка: покрытие {all_result['coverage']:.1%}, правильность среди подтверждённых {all_result['confirmed_precision']:.1%}.
Новые целевые фотографии: {new['count']}; top-1 {new['top1']:.1%}.
Историческая часть: {old['count']}; top-1 {old['baseline_body_top1']:.1%} → {old['top1']:.1%}.
Это результаты по фрагментам с групповым разделением источников, не точность поиска машин на всей сцене и не независимый полевой экзамен в Узбекистане.
По каждой марке, включая слабые, см. Development/TrainingInputs/evaluation.json. Редкие марки с несколькими снимками нельзя считать надёжно изученными.
Выбор весов выполнен по validation; test не использован для выбора эпохи. Повторные ракурсы одной авторской сессии не разделялись между train/validation/test.

ИСХОДНИКИ И УЧЕБНОЕ ОБУЧЕНИЕ
Обновлены исходники carvision, tools, tests. Development/TrainingInputs содержит протоколы, группы и происхождение/лицензии реальных фотографий.
Готовые автомобильные классификаторы и платные API не использовались. Данные не синтезировались.
Тренировка через учебную вкладку создаёт отдельный кандидат по вашей разметке; она не заменяет активный расширенный эксперт марки автоматически.
Рецепт нового эксперта: tools/train_make_scope_v23.py --fine-tune в окружении разработки с PyTorch и подготовленным cache/make_scope_v23.pt. Сами учебные пиксели в патч не включены: старый исследовательский корпус содержит фотографии с ограничениями на распространение. Для повторного расширенного обучения разработчику требуется восстановить исходный корпус и пройти review/audit. Обычному пользователю это не нужно: обученные веса полностью включены и работают офлайн.
Файлы происхождения перечисляют также исключённые исходные результаты поиска; в обучение попадают только строки подготовленного корпуса, прошедшие review/audit.
Дополнительные 99 фотографий из папки владельца «машинки» сохранены в проекте отдельно: вручную проверены 10 подписей марки/цвета/области для пяти марок; остальные 89 снимков ещё требуют разметки. Веса этой версии на них не обучались; они не использованы для независимой оценки. Автоматические ответы не выдаются за правильную разметку.
Преобразования: обрезка, уменьшение, маска, зеркалирование и учебные ухудшения изображения. Исходные лицензии/авторы сохранены в PHOTO_CREDITS.
GitHub/публикация не выполнялись. Независимая проверка на 20 новых телефонных фотографиях и остальные ранее отмеченные учебные требования не считаются выполненными этим техническим патчем.

ОТКАТ
Закройте CarVision. Скопируйте содержимое своей patch-backups/update23-ДАТА обратно в папку программы с заменой. Новые make_expert.pt и make_catalogue.json при полном откате уберите из models.
Если обновление завершилось ошибкой при копировании, установщик автоматически восстанавливает прежние файлы и сообщает результат.
'''
add('UPDATE_2_3_RU.txt',content=guide)
(stage/'READ_FIRST_RU.txt').write_text(guide,encoding='utf-8')
for name in ['APPLY_UPDATE_2_3.ps1','APPLY_UPDATE_2_3.cmd']:shutil.copy2(ROOT/'handoff'/name,stage/name)
manifest=dict(release='CarVision make-first update 2.3',exe_sha256=sha(build/'CarVision.exe'),previous_exe_sha256=[hashlib.sha256(original.read('CarVision/CarVision.exe')).hexdigest()],
              files=files,required_files=required_files,changed_runtime_files=changed,make_model_sha256=evaluation['candidate_sha256'],trained_makes=make_count,confirmed_make_gates=confirmed)
(stage/'UPDATE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('STAGED',len(files),'files; runtime dependencies',len(changed),flush=True)
print('Do not ZIP or deliver until tools/verify_make_update_v23.py passes on the actual recipient EXE.',flush=True)
