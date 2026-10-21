"""Select by grouped validation, preserve baseline, install a matched local pair."""
import hashlib,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from carvision.core import ROOT
from tools.ev_group_audit import clean_v18_baseline
torch.set_num_threads(3)
def score(metrics):return sum(v['top1']+.5*v['top3'] for v in metrics.values())/len(metrics)
baseline_val=clean_v18_baseline();candidates=[];decisions=[]
for name in ['ev','prototype','hybrid']:
 network=ROOT/f'models/{name}_parts_candidate';reference=ROOT/f'models/{name}_reference_candidate';file=reference/'training.json'
 if not file.exists():decisions.append(dict(candidate=name,accepted=False,reason='No successfully fitted, guarded references'));continue
 info=json.loads(file.read_text(encoding='utf-8'));chosen=info['selected'];legacy=chosen['legacy_validation'];new=chosen.get('new_validation',{})
 accepted=chosen['clean_floor'] and chosen.get('new_floor',False) and score(legacy)>=score(baseline_val)-.01
 decisions.append(dict(candidate=name,accepted=accepted,validation_score=chosen['score'],old_validation_score=score(legacy),baseline_old_validation_score=score(baseline_val),old_important_modes_preserved=chosen['clean_floor'],new_whole_validation=new.get('whole'),reason='Validation only; no test metric participates in this decision'))
 if accepted:candidates.append((chosen['score'],name,network,reference,info))
if candidates:_,name,network,reference,info=max(candidates,key=lambda r:r[0])
else:
 name='occlusion';network=ROOT/'models/occlusion_parts_candidate';reference=ROOT/'models/occlusion_reference_candidate';info=json.loads((reference/'training.json').read_text(encoding='utf-8'))
 if not info['selected']['clean_floor'] or info['selected']['score']<=score(baseline_val):raise SystemExit('No candidate improves guarded validation; retain baseline')
model_path=network/'vehicle_model.pt';bank_path=reference/'visual_references.pt';sha=hashlib.sha256(model_path.read_bytes()).hexdigest();ck=torch.load(model_path,weights_only=True);bank=torch.load(bank_path,weights_only=True)
assert bank['source_model_sha256']==sha and bank['labels']==ck['labels'] and bank['brands']==ck['brands']
# Fixed baseline test is measured only AFTER the validation-only choice is final.
baseline_test=clean_v18_baseline('test')
selection=dict(selected=name,model_sha256=sha,reference_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),labels=len(ck['labels']),decisions=decisions,validation_only=True,baseline_clean_validation=baseline_val,baseline_clean_test=baseline_test,selected_results=info['results'],model_training_recipe=str(network.relative_to(ROOT)),reference_training_recipe=str(reference.relative_to(ROOT)),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),neural_rounds=2,public_photos_downloaded=244,own_student_photos=0)
backup=ROOT/'models/backups/v18_before_v19';backup.mkdir(parents=True,exist_ok=True)
for filename in ['vehicle_model.pt','visual_references.pt','visual_parts_training.json','model_card.json','expanded_model_training.json','expanded_dataset.json']:
 source=ROOT/'models'/filename
 if source.exists() and not (backup/filename).exists():shutil.copy2(source,backup/filename)
if not (backup/'README.md').exists():shutil.copy2(ROOT/'README.md',backup/'README.md')
for source,filename in [(model_path,'vehicle_model.pt'),(bank_path,'visual_references.pt'),(reference/'training.json','visual_parts_training.json'),(network/'training.json','vehicle_model_training_v19.json'),(ROOT/'models/ev_parts_candidate/dataset.json','ev_dataset.json')]:
 target=ROOT/'models'/filename;temp=target.with_suffix(target.suffix+'.new');shutil.copy2(source,temp);temp.replace(target)
(ROOT/'models/release_v19_comparison.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
data=json.loads((ROOT/'models/ev_dataset.json').read_text(encoding='utf-8'));card=json.loads((ROOT/'models/model_card.json').read_text(encoding='utf-8'))
card.update(owner='София Рахманкулова',release='1.9',supported_brands=sorted(set(card['supported_brands'])|set(ck['brands'])),supported_vehicle_models=[dict(brand=b,model=m) for b,m in zip(ck['brands'],ck['labels'])],vehicle_model_test_accuracy=info['results']['test']['whole']['combined']['top1'],vehicle_model_test_count=info['results']['test']['whole']['count'],vehicle_model_threshold=bank['confidence_threshold'],model_rejection_threshold=bank['confidence_threshold'],instance_model_metrics=info['results'],visual_parts=info,expanded_model_training=json.loads((network/'training.json').read_text(encoding='utf-8')),detector='General COCO Mask R-CNN visible-instance masks, instance bridge association, duplicate consolidation and person-pixel exclusion; Faster R-CNN is a fallback. Segmentation neural weights are unchanged this round.',current_scope=f"Selected {name} candidate; {len(ck['labels'])} learned model families, closed set. Unseen/rebadged/occluded cars can remain wrong or unknown; colour is not a model prohibition. Two bounded local neural rounds; no ready-made vehicle-name classifier.",v19_data=dict(real_licensed_photos=244,accepted_new_photos=len(data['new_rows']),new_capture_groups=len({r['group'] for r in data['new_rows']}),group_audit=data['group_audit'],sources='Per-file Commons/Flickr author/title/licence records; Flickr licence verified on source page; no generated photographs',own_student_photos=0),v19_selection=decisions)
(ROOT/'models/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
def pct(value):return f'{100*value:.1f}%'
table='| Видимые данные | v1.8, test389: top1 / top3 | v1.9, те же test389: top1 / top3 |\n|---|---|---|\n'
for kind,label in [('whole','Весь автомобиль'),('upper55','Верхние55% вырезки'),('upper35','Верхние35% вырезки')]:
 old=baseline_test[kind];new=info['results']['test'][kind]['legacy_combined'];table+=f"| {label} | {pct(old['top1'])} / {pct(old['top3'])} | {pct(new['top1'])} / {pct(new['top3'])} |\n"
new_test=info['results']['test']['whole'].get('new_combined');new_count=info['results']['test']['whole']['count']-389
new_line=f"На {new_count} новых test-снимках: top1 {pct(new_test['top1'])}, top3 {pct(new_test['top3'])}. На отдельную модель приходится всего3–5тестовых фото, поэтому это ограниченная оценка." if new_test else 'Расширенный32-классовый кандидат не прошёл защитные проверки; новые фотографии предоставлены для дальнейшей работы. Активны24модели с улучшенным сравнением фрагментов.'
models='; '.join(b+' '+m for b,m in zip(ck['brands'],ck['labels']))
readme=f'''# CarVision1.9 — переносимый Windows-прототип

Владелец: София Рахманкулова. Программа автоматически находит машины и предлагает марку, конкретную модель и цвет. Работает локально, без API, ключей, оплаты и сети при анализе.

## Запуск и перенос
Распакуйте TashkentCarVision-v1.9-Portable.zip целиком. Запустите CarVision/CarVision.exe или Start CarVision.cmd. Для другого Windowsx64-компьютера переносите ВСЮ папку CarVision, включая models и _internal. Установка Python для анализа не требуется. Проверка отдельной копии на этом ПК не доказывает совместимость со всеми другими устройствами.

Откройте фото → «Анализировать фото». Марку выбирать не нужно. «Выбрать зону мышью» → обведите одну машину → распознайте в зоне. Полный контекст сохраняется, соседние видимые маски исключаются из её цвета. Двойной щелчок открывает крупный просмотр. Справа показаны до3гипотез и весь список изученных моделей. Знак вопроса означает предположение, а не установленную модель.

В «Подробных настройках» есть «Объединить части одной машины»: отметьте её ошибочно разделённые контуры. Объединяются только уже видимые пиксели, ответ рассчитывается заново; действие можно отменить. Автоматическая ассоциация использует общее предложение детектора, простая близость машин не является основанием объединения. Ошибки ассоциации всё ещё возможны. Разные цвета контуров различают экземпляры; цвет кузова показан отдельно.

TXT/«Ответ для задания» содержат пары brand colour. Подробный TXT, JSON и JPG содержат модели/варианты, уверенности, контуры и раздельную легенду. Русский текст JPG поддерживает кириллицу. Числа сети относительны и не гарантируют ответ.

## Фотографии и самостоятельное обучение
Отдельный архив CarVision-Real-Chinese-EV-Photos-v1.9.zip содержит244найденные реальные фотографии, авторов, лицензии и train/validation/test принадлежность. Ни одна фотография не сгенерирована.194снимка прошли фильтрацию; из них137train,30validation,27test.146новых групп съёмки. Уличные сцены без разметки всех автомобилей не превращены в подписанные обучающие машины. Дубликаты, интерьер, неверно описанный главный объект и слишком малые/неоднозначные вырезки исключены; причины сохранены.

В «Примеры фото» доступны images(6городских диагностических кадров), Chinese_EV_Test(8детерминированно выбранных новых test-фото) и Street_scenes(8неразмеченных сцен). Новые test-примеры не использовались для обучения/выбора параметров, но их просмотр не заменяет проверку на своих новых фото. Лицензии каждой фотографии — samples/EV_PHOTO_SOURCES.json и отдельный фотоархив. Некоторые фотографические семейства присутствуют только в наборе для дальнейшего расширения, а не в активном классификаторе.

Для задания нужен собственный набор Ташкента. «Учебная разметка» сохраняет ваши марки, цвета и группы. Во вкладке «Обучение» расписаны4шага: разметить → проверить → обучить → оценить. Эта кнопка обновляет отдельный классификатор марок/цветов и резервный детектор рамок; контурный COCO-сегментатор и список названий моделей она не переобучает. Рецепты обучения моделей кузова находятся в source/tools. Старые веса сохраняются. Импорт обычного тестового фото ничего сам по себе не обучает.

## Измерения и границы
Выбран кандидат {name}; активных семейств моделей: {len(ck['labels'])}. Общие ImageNet/COCO-веса разрешены, готовые автомобильные классификаторы марок/моделей/цветов не используются. Выполнены2ограниченных нейросетевых раунда,16+16эпох, плюс обучение/подбор собственного сравнения деталей на train/validation. Кандидаты и параметры выбраны только по validation. Цвет не запрещает возможную модель.

Для честного сравнения обнаружены7старых авторских сессий, попавших в разные split у разных марок.3validation и5test изображений исключены из новых оценок. Исходные старые split не перетасованы, прежние test не перемещены в train. Старый чистый train/validation/test:642/131/389. С новыми данными:779/161/416. Сессия author/date — приближение; настоящие идентификаторы всех машин публичных источников отсутствуют.

{table}
{new_line}

Top1 — первый вариант совпал с подписью; top3 — верный вариант есть среди трёх. Верхние вырезки имитируют отсутствие деталей, не являются полноценным тестом реальной городской окклюзии. Исторические опубликованные оценки1.8на394изображениях не сравниваются напрямую с очищенными389. Все результаты, выбор и размеры выборок: models/release_v19_comparison.json. Ошибки по редким Cobalt/Damas/Nexia и незнакомым семействам остаются; у этих классов недостаточно независимых примеров, для некоторых нет validation. Точного определения любого автомобиля по любой крыше/значку/цвету программа не гарантирует. Отдельный детектор значков не заявляется. Видимая маска не восстанавливает скрытый кузов.

Новые марки, которых нет в прежней18-классовой голове марок, могут быть выведены из уверенного собственного классификатора модели. Семейства Yuan Plus/H6 и другие могут включать поколения и варианты; распознавание имени не определяет двигатель, комплектацию либо чистую электрическую тягу. При слабых данных остаются гипотезы/unknown.

Изученные модели: {models}.

## Исходники и задание
source содержит редактируемый проект и рецепты. Для разработки создайте Python3.12venv, установите requirements-lock.txt, собирайте build.ps1. Для повторного исследования нужны исходные открытые данные/кэши по provenance, они целиком не вложены в программу; новые244фото есть в отдельном архиве. Переносимый анализ использует включённые веса и сеть не требует.

Собственных учебных фото студента пока0; публичные снимки это требование PDF не заменяют. GitHub-ссылка не предоставлена, публикации не выполнялись. Очный телефонный цикл/20новых снимков, запуск на другом физическом ПК, подтверждение уникальности ML-семейства преподавателем и финальные сведения о вкладе ещё не выполнены. Шестистраничный PDF в docs сохраняет исторический протокол1.5, не является обновлённым отчётом1.9или свидетельством полной сдачи. Никакие данные, история работы и результаты не выдуманы.
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
prefix=f"# Обновление1.9\n\nСофия Рахманкулова. Активны {len(ck['labels'])} модельных семейств; выбран {name}. Автообъединение фрагментов, исключение людей, ручная корректировка/пересчёт/отмена, цветные номера, поиск по видимой маске.244реальных лицензированных фото,194новых пригодных,8новых модельных семейств в кандидате. Точные текущие измерения и ограничения — README.md и models/release_v19_comparison.json. Ниже сохранён предыдущий протокол; его исторические числа не описывают новую сборку.\n\n"
for filename in ['INSTANCE_ANALYSIS.md','MODEL_AND_COLOUR.md','REQUIREMENTS_AUDIT.md','CHECKLIST.md']:
 file=ROOT/filename;text=file.read_text(encoding='utf-8')
 if not text.startswith('# Обновление1.9'):file.write_text(prefix+text,encoding='utf-8')
third=ROOT/'THIRD_PARTY.md';text=third.read_text(encoding='utf-8');text+='\n\nCarVision1.9:244real Commons/Flickr photos, per-file author/licence/source/changes in samples/EV_PHOTO_SOURCES.json and CarVision-Real-Chinese-EV-Photos-v1.9.zip. Flickr licences verified on original source pages. No generated photographs; public photo ownership is not attributed to the student. See CC BY/CC BY-SA/CC0/PD conditions per image.\n';third.write_text(text,encoding='utf-8')
transfer=ROOT/'TRANSFER.txt';text=transfer.read_text(encoding='utf-8').replace('v1.8','v1.9').replace('1.8','1.9');transfer.write_text('Новая сборка1.9: София Рахманкулова. Переносите всю распакованную папку CarVision. Новые реальные фото — отдельный CarVision-Real-Chinese-EV-Photos-v1.9.zip; его распаковка не обязательна для анализа.\n\n'+text,encoding='utf-8')
print('SELECTED',name,'labels',len(ck['labels']),'model',sha,'validation-only decision, matched pair installed',flush=True)
