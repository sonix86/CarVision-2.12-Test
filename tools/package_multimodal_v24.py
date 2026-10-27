"""Stage only a measured improvement, preserving a strict reversible installer."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'models/multimodal_v24';stage=ROOT/'outputs/CarVision-Update-2.4';payload=stage/'payload'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert load(out/'release_gate.json')['passed'],'Rejected candidate must not be packaged'
evaluation=load(out/'evaluation.json');training=load(out/'training.json')
assert evaluation['candidate_sha256']==sha(out/'multimodal_make.pt')
build=ROOT/'outputs/build-v24/dist/CarVision';frozen=load(ROOT/'outputs/frozen-v24-check.json')
assert frozen['passed'] and frozen['exe_sha256']==sha(build/'CarVision.exe')
assert not stage.exists(),'Inspect an existing stage; never silently overwrite it'
payload.mkdir(parents=True);baseline=ROOT/'outputs/Transfer extracted София/CarVision';original=zipfile.ZipFile(ROOT/'CarVision-2.1-Full-Transfer.zip');original_names=set(original.namelist());files=[];required=[];changed=[]
def add(relative,source=None,content=None,extra_old=()):
    dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if source is not None:shutil.copy2(source,dest)
    else:dest.write_text(content,encoding='utf-8')
    old=[];entry='CarVision/'+relative
    if entry in original_names:old.append(hashlib.sha256(original.read(entry)).hexdigest())
    old+=list(extra_old)
    critical=relative=='CarVision.exe' or relative.startswith('_internal/') or relative in ('models/vehicle_model.pt','models/visual_references.pt')
    files.append(dict(path=relative,sha256=sha(dest),previous_sha256=sorted(set(old)),require_existing=critical and entry in original_names))
add('CarVision.exe',source=build/'CarVision.exe')
for p in sorted((build/'_internal').rglob('*')):
    if not p.is_file():continue
    relative=p.relative_to(build).as_posix();old=baseline/relative
    if not old.exists() or sha(old)!=sha(p):add(relative,source=p);changed.append(relative)
    else:required.append(dict(path=relative,sha256=sha(p)))
for name in ('vehicle_model.pt','visual_references.pt'):add('models/'+name,source=ROOT/'models/ev_patch_v22'/name,extra_old=[sha(ROOT/'models/ev_patch_v22'/name)])
add('models/multimodal_make.pt',source=out/'multimodal_make.pt')
colour=load(out/'colour_evaluation.json') if (out/'colour_evaluation.json').exists() else None
include_colour=bool(colour and colour['passed'])
if include_colour:
    assert colour['candidate_sha256']==sha(out/'illumination_colour.pt')
    add('models/illumination_colour.pt',source=out/'illumination_colour.pt')
if training['rounds'][training['selected_round']-1]['with_text']:
    for p in sorted((out/'ocr').glob('*')):
        if p.is_file():add('models/ocr/'+p.name,source=p)
active=load(baseline/'Development/ACTIVE_MODEL_SHA256.json')
for name in ('vehicle_model.pt','visual_references.pt'):active[name]=sha(ROOT/'models/ev_patch_v22'/name)
active['multimodal_make.pt']=sha(out/'multimodal_make.pt')
if include_colour:active['illumination_colour.pt']=sha(out/'illumination_colour.pt')
required += [dict(path='models/'+n,sha256=s) for n,s in active.items() if n not in ('vehicle_model.pt','visual_references.pt','multimodal_make.pt','illumination_colour.pt')]
add('Development/ACTIVE_MODEL_SHA256.json',content=json.dumps(active,indent=2),extra_old=[sha(baseline/'Development/ACTIVE_MODEL_SHA256.json')])
add('CHECK_ON_NEW_PC.ps1',content=(ROOT/'handoff/CHECK_ON_NEW_PC.ps1').read_text(encoding='utf-8').replace("version='2.1'","version='2.4'"));add('CHECK_ON_NEW_PC.cmd',source=ROOT/'handoff/CHECK_ON_NEW_PC.cmd')
for p in sorted((ROOT/'carvision').glob('*.py')):add(p.relative_to(ROOT).as_posix(),source=p)
for p in sorted((ROOT/'tests').glob('test_*.py')):add(p.relative_to(ROOT).as_posix(),source=p)
for p in sorted((ROOT/'tools').glob('*v24.py')):add(p.relative_to(ROOT).as_posix(),source=p)
for name in ('requirements.txt','requirements-ocr-v24.txt','CarVision-v24.spec','launch.py'):add(name,source=ROOT/name)
for name in ('dataset.json','policy.json','training.json','evaluation.json','release_gate.json','colour_training.json','colour_evaluation.json'):
    if (out/name).exists():add('Development/Multimodal24/'+name,source=out/name)
for name in ('data_audit.json','review_exclusions.json','review_relabels.json','reviewed_images.json','manual_boxes.json'):
    p=ROOT/'models/make_scope_v23'/name
    if p.exists():add('Development/Multimodal24/base_'+name,source=p)
for folder in ('make_scope_v23','make_flickr_v23'):add('Development/Multimodal24/'+folder+'_PHOTO_CREDITS.json',source=ROOT/'datasets'/folder/'manifest.json')
add('Development/Multimodal24/convnext_tiny_SOURCE.json',source=ROOT/'cache/imagenet/convnext_tiny_SOURCE.json')
for name in ('manifest.json','labels.jsonl','STATUS.txt'):add('Development/Multimodal24/owner_oct25/'+name,source=ROOT/'data/owner_oct25'/name)
for name in ('APPLY_UPDATE_2_3.ps1','APPLY_UPDATE_2_3.cmd'):
    content=(ROOT/'handoff'/name).read_text(encoding='utf-8').replace('2.3','2.4').replace('2_3','2_4').replace('update23','update24')
    (stage/name.replace('2_3','2_4')).write_text(content,encoding='utf-8-sig')
result=evaluation['results']['all']['whole'];text_used=training['rounds'][training['selected_round']-1]['with_text']
guide=f'''CarVision 2.4 — обновление распознавания марки
Владелец: София Рахманкулова

УСТАНОВКА
1. Закройте CarVision.
2. Распакуйте этот ZIP В ОТДЕЛЬНУЮ папку обновления.
3. Запустите APPLY_UPDATE_2_4.cmd. Выберите старую папку с CarVision.exe.
4. Дождитесь сообщения Update 2.4 installed successfully.
5. Откройте CarVision.exe из старой папки. Повторно обучать модель не нужно.
Установщик сам заменяет EXE, нужные библиотеки и обученные веса. Просто бросать ZIP рядом с EXE недостаточно.
Старые файлы сохраняются в patch-backups/update24-ДАТА; ваши фото, разметка и результаты сохраняются.
Нужна полная переданная ранее версия 2.1 либо её обновление 2.2. Этот патч сам по себе не является полной программой.
Windows x64. Python, интернет, API-ключи, подписки и платежи для запуска не нужны.
Если проверка файлов остановила установку, не обходите её: сохраните свою работу и возьмите чистую копию исходного полного архива.

ИСПОЛЬЗОВАНИЕ
Открыть фото → Анализировать фото. Для одной машины: Выбрать зону мышью → обвести машину → Анализировать фото.
Выберите контур или строку. Сначала показывается марка; модель — только при достаточных данных.
«Возможно» — предположение. Проценты не гарантируют правильность. Неизвестные и плохо видимые машины могут ошибочно совпасть с изученными.
CHECK_ON_NEW_PC.cmd проверяет запуск, веса, анализ, зону, экспорт и учебное обучение в отдельной папке; активные веса не меняет.

ИЗМЕНЕНИЯ И ИЗМЕРЕНИЯ
Общий ImageNet ConvNeXt Tiny, два визуальных окна (кузов и детали), собственный обученный классификатор {len(training['calibration'])} марок.
Общее офлайн-распознавание текста используется как обученный дополнительный признак: {text_used}.
Новый классификатор освещения/цвета прошёл проверки и включён: {include_colour}.
Фиксированная отложенная выборка: {result['count']} фрагментов. Первая гипотеза марки: {result['baseline_body_top1']:.1%} → {result['top1']:.1%}.
Подтверждено {result['confirmed']} фрагментов; правильность среди них {result['confirmed_precision']:.1%}. Полные результаты и слабые марки — Development/Multimodal24/evaluation.json.
Это проверка по исходным группам публичных фотографий, не независимый полевой экзамен на новых телефонных снимках.
18 проверенных собственных фрагментов добавлены только в обучение. Три текущих снимка с ошибками исключены из обучения; правильные ответы пользователя неизвестны, точность по ним не заявляется.
82 метки марки и каталог 325 названий НЕ означают 325 надёжно распознаваемых моделей. Декодер точной модели остаётся на 42 семействах.
Детектор/сегментация не переобучались этим обновлением. Кратное улучшение, 100% правильность и все модели не заявляются.

СООТВЕТСТВИЕ ЗАДАНИЮ
Готовый автомобильный классификатор и внешние API не использовались. Общий ImageNet/COCO разрешён разделом 5; марку определяет собственная обученная голова. Общий OCR не выдаёт готовый ответ марки.
Данные реальных фотографий, происхождение, лицензии, разделение и параметры обучения сохранены в Development/Multimodal24.
Остаются учебные требования: собственные телефонные данные как основа, независимый прогон 20 новых фото, уникальность архитектуры среди команд, понимание участниками, итоговый PDF до 6 страниц с GitHub URL, реальная история вкладов и тег midterm. Этот технический патч не подтверждает их выполнение.
Исходники включены. Для повторения обучения приватные исходные фото потребуется передать отдельно; они не публикуются автоматически.
'''
(stage/'READ_FIRST_RU.txt').write_text(guide,encoding='utf-8');add('GUIDE_UPDATE_2_4_RU.txt',content=guide)
manifest=dict(release='CarVision update 2.4',exe_sha256=sha(build/'CarVision.exe'),previous_exe_sha256=[hashlib.sha256(original.read('CarVision/CarVision.exe')).hexdigest()],files=files,required_files=required,changed_runtime_files=changed,make_model_sha256=evaluation['candidate_sha256'],include_colour=include_colour)
(stage/'UPDATE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print('STAGED',len(files),'files; verify before sealing or delivery')
