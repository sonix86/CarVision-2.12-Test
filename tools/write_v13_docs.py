"""Documentation is derived from the shipped checkpoints and measured results."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Classifier

brand=torch.load(ROOT/'models/classifier.pt',weights_only=True);colour=json.loads((ROOT/'models/colour_training.json').read_text(encoding='utf-8'));test=json.loads((ROOT/'models/evaluation/public_test.json').read_text(encoding='utf-8'));regional=json.loads((ROOT/'models/regional_brand_training.json').read_text(encoding='utf-8'));data=json.loads((ROOT/'data/regional/provenance.json').read_text(encoding='utf-8'));brands=brand['brands'];names=', '.join(brands)
card=dict(owner='София Рахманкулова',release='1.3',brand_parameters=sum(p.numel() for p in Classifier(len(brands)).parameters()),brand_architecture=f'MobileNetV3 Small, own avg/max pooled {len(brands)}-brand head, regional fine-tuning, mirrored-view probability mean and development temperature',detector='Generic Faster R-CNN ResNet50 FPN v2 COCO, min800/max1333; default whole frame, optional tiles',general_initialization='General ImageNet/COCO explicitly authorized; no ready-made car-brand/vehicle-colour classifier',supported_brands=brands,image_size=160,colour=colour['method'],brand_test_accuracy=test['brand_top1_accuracy'],brand_test_images=test['images'],brand_scope=test['brand_scope'],own_tashkent_training_photos=0,regional_test_accuracy=regional['regional_test_accuracy'],regional_test_images=regional['test_regional'],colour_test_accuracy=colour['test_accuracy'],colour_test_images=colour['test'],colour_classes=colour['classes'],colour_source=colour['source'],ready_made_colour_model_used=False)
(ROOT/'models/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
readme=f'''# Tashkent CarVision 1.3

Владелец: **София Рахманкулова**. Локальное распознавание марки и одного из восьми цветов. Windows 10/11 x64. Интернет, платные сервисы, ключи и установленный Python для EXE не нужны.

## Запуск и перенос

Полностью распакуйте **TashkentCarVision-v1.3-Portable.zip** и запустите **CarVision/CarVision.exe**. Можно использовать Start CarVision.cmd. Переносите всю папку CarVision, включая _internal, models, data и samples. Папка должна разрешать запись; результаты сохраняются в outputs. Не запускайте EXE прямо из ZIP.

## Как распознать машину

1. Нажмите «Открыть фото» или «Примеры Ташкента».
2. Для всей фотографии оставьте «Весь кадр» и нажмите «Анализировать».
3. Для нужного участка выберите «Машины в области» и обведите его мышью: детектор выполнит поиск только внутри участка.
4. Для конкретного автомобиля выберите «Одна машина в области», обведите именно его и нажмите «Анализировать». Классификатор работает напрямую с выбранной вырезкой, даже если общий детектор машину пропустил. В область должна попадать машина; этот режим не проверяет, что вы обвели автомобиль.
5. На фотографии показаны короткие номера. Марка и цвет находятся в таблице. Щёлкните рамку/номер/строку; двойной щелчок или «Увеличить машину» открывает отдельный просмотр. В нём можно повторно распознать эту машину.
6. «Весь кадр» снимает область. При открытии другого фото область сбрасывается.
7. «Копировать ответ» копирует список; «Экспортировать файлы» сохраняет TXT, JSON и JPG. Названия на экспортном JPG находятся в отдельной легенде и не закрывают автомобили.

Правильную марку и цвет при анализе вводить **не нужно**. Вкладка «Учебная разметка» предназначена для подготовки учебных ответов. По заданию определяется **марка**, а не точное коммерческое название модели вроде Cobalt/Tracker.

Начальный режим - весь кадр: он даёт меньше фрагментных повторов. Balanced добавляет перекрывающиеся фрагменты, detail также отражённый кадр; они полезны для мелких машин, но медленнее. Внутренние обрезанные рамки и вложенные фрагменты разных проходов отбрасываются; NMS 0.45. Настройки порога рамки и марки меняют уверенность, а не правильный ответ. Ниже порога марки выводится «Не уверен» / unknown.

## Обученные модели и реальные измерения

Общий COCO-детектор только находит автомобили и фургоны (car/truck); bus исключён. Размер 800/1333. Классификатор марок обучен в этом проекте с общей ImageNet-инициализацией, затем дообучен на публичных вариантах автомобилей, встречающихся в регионе. Готовые brand/vehicle-colour classifiers не использованы. Общие веса разрешены владельцем и заданием.

**{len(brands)} обученных марок:** {names}.

- Stanford official test: **{test['brand_top1_accuracy']:.1%}**, {test['images']} эталонных вырезок, 13 марок в этом конкретном тесте.
- Дополнительный региональный holdout: **{regional['regional_test_accuracy']:.1%}**, {regional['test_regional']} вырезок; старая модель на тех же вырезках **{regional['baseline_regional_test_accuracy']:.1%}**.
- Левая/правая половина автомобиля (искусственные 50% вырезки holdout): **{regional['regional_half_vehicle_test_accuracy']:.1%}**, старая модель **{regional['baseline_regional_half_vehicle_test_accuracy']:.1%}**. Это не тест реальных уличных перекрытий.
- Цвет VCoR official test: **{colour['test_accuracy']:.1%}**, {colour['test']} изображений. Это другой набор, а не совместная точность улиц Ташкента.

Цвет: {colour['method']}. Восемь классов: white/black/grey/red/blue/green/brown/yellow; silver -> grey, beige/gold/tan -> brown, orange -> yellow. Псевдометки HSV из Stanford не используются для обучения цвета. Тестовые метки не выбирают checkpoint или смесь.

## Ограничения и данные

Сильное перекрытие, очень маленькая машина, неразличимая эмблема, тени и отражения могут дать ошибку. Неизвестная марка иногда получает уверенный неправильный известный класс. Уверенность не гарантирует правильность. Шесть городских примеров и две присланные сцены использованы как диагностика разработки, не независимый финальный тест; ответы не подставляются по пути/имени файла. По цвету дополнительно использована небольшая AI-размеченная development-подборка; она не заменяет независимую человеческую проверку.

Региональные публичные данные: train {regional['train_regional']}, validation {regional['validation_regional']}, test {regional['test_regional']}. Явные категории/названия Commons, источники и CC-лицензии сохранены в data/regional. Автоматические рамки проверяются фильтрами размера/неоднозначности, near-dHash дубликаты удалены, автор/дата съёмки группируются. Настоящие идентичности машин не доступны; ошибки категорий возможны. Публичные исходные фото не включены в переносимый пакет.

**Собственных фотографий Софии пока 0.** Требование PDF о собственном наборе Ташкента публичными фотографиями не закрывается. GitHub ещё не предоставлен; уникальность стратегии среди команд и полный телефон -> передача -> ответ <=120 секунд требуют фактической проверки участником. Отчёт в docs - честный шестистраничный черновик, не заявление о завершённой учебной сдаче.

## Свои фотографии

В «Учебной разметке» добавляйте фото, обводите машины, указывайте правильные марки/цвета, source=own и группу машины/сессии. Повторные кадры одной машины держите в одной группе. Файлы находятся в data/images и data/labels.jsonl. «Обучение» проверяет данные, обучает головы и детектор с локальной общей основой; прежние веса сохраняются в models/backups. Небольшой набор может ухудшить модель. После обучения нужна повторная оценка.

## Исходники и воспроизведение

Редактируемый код включён в source. Для разработки: Python 3.12, venv, requirements.txt; для скачивания датасетов также requirements-data.txt. Из папки source скопируйте соседнюю папку models в source/models. EXE уже использует свои models.

Базовый эксперимент: tools/download_prototype_data.py -> prepare_prototype_data.py -> prepare_generic_detector.py --accurate -> train_public_brand.py --imagenet --epochs 40 --size 160. Цвет: download_colour_data.py -> train_colour_model.py. Адаптация: collect_regional_data.py -> prepare_regional_data.py -> train_regional_brand.py. Сравнение цвета: train_robust_colour.py, train_paint_colour.py, fit_colour_evidence_ensemble.py, finalize_v13_model_metrics.py. Точные истории, критерии и выбранные варианты сохранены в models/*.json. Скачивание/обучение требует времени, интернета и места; запуск переносимого EXE этого не требует.

Инженерные проверки - VALIDATION.md; раздельные показатели и ограничения - PROTOTYPE_VALIDATION.md; лицензии - THIRD_PARTY.md и licenses. Точный ответ программы также доступен в CLI: CarVision.exe predict photo.jpg --region 100 100 800 600 --single --output outputs.
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
validation=f'''# Измерения CarVision 1.3

Все значения относятся к фактически поставляемым весам; готовый классификатор марок/цветов не использован. Генератор: tools/write_v13_docs.py. Подробные записи: models/evaluation/public_test.json, models/regional_brand_training.json, models/colour_training.json.

| Проверка | v1.2 | v1.3 | Область применимости |
|---|---:|---:|---|
| Марка / региональный holdout | {regional['baseline_regional_test_accuracy']:.1%} | {regional['regional_test_accuracy']:.1%} | {regional['test_regional']} вырезок Commons, несколько региональных вариантов |
| Марка / половина автомобиля | {regional['baseline_regional_half_vehicle_test_accuracy']:.1%} | {regional['regional_half_vehicle_test_accuracy']:.1%} | Левая и правая50%, синтетическая видимость |
| Марка / Stanford official test | 84.6% | {test['brand_top1_accuracy']:.1%} | {test['images']} эталонных вырезок, 13 представленных марок |
| Цвет / VCoR official test | 91.6% | {colour['test_accuracy']:.1%} | {colour['test']} изображений, 8 цветов |

Региональная адаптация: {regional['train_regional']} train + {regional['validation_regional']} validation при replay {regional['train_stanford']} базовых train-фото. {len(regional['history'])} реальных эпох, {regional['augmentation']}. Температура уверенности {regional['temperature']:.3f} обучена на development, не test. {len(brands)} классов: {names}.

Цветовая модель: {colour['method']}. Train {colour['train']}, validation {colour['validation']}; исходные метки VCoR, все8 цветов, exact SHA256/identical dHash очистка. {colour['selection']}. Дополнительная development-диагностика23 вырезок: {colour['cross_domain_development_accuracy']:.1%}; это использованная при выборе AI-разметка, не независимый итоговый эталон. Сравнения сохранены, неудачные варианты не выдаются за улучшения.

Public benchmark не доказывает совместную точность всех машин/цветов на фото Ташкента. В Commons split сгруппированы автор/дата и удалены похожие вырезки; истинные vehicle IDs не доступны. Ошибки публичных категорий/автоматических рамок возможны. Шесть демо-фотографий исключены из тренировочного списка по именам и проверке похожести, но используются для разработки и регрессий; они не невиденный тест.

Нужны собственные фотографии Ташкента, независимая проверка цвета человеком, реальные перекрытия и новые китайские модели, полный120с цикл с телефона и запуск на втором физическом Windows-компьютере. Собственных фото:0. GitHub ожидается. Не все академические требования PDF закрыты.
'''
(ROOT/'PROTOTYPE_VALIDATION.md').write_text(validation,encoding='utf-8')
for filename in ['CHECKLIST.md','REQUIREMENTS_AUDIT.md']:
    text=(ROOT/filename).read_text(encoding='utf-8');text=text.replace('13 обученных марок, настраиваемый порог unknown. Китайские марки в данных отсутствуют, известны ограничения открытого распознавания.',f"{len(brands)} обученных марок; список/реальные метрики в model_card.json. Порог unknown, ограниченное открытое распознавание.");text=text.replace('Отсутствие покрытия явно указано. Нужны собственные примеры и расширение обучения.','Реальный охват указан в model_card.json. Дополнительные публичные варианты обучены; нужны собственные данные и дальнейшее расширение.');(ROOT/filename).write_text(text,encoding='utf-8')
text=(ROOT/'THIRD_PARTY.md').read_text(encoding='utf-8').replace('The 13-brand decision head is initialized randomly and trained by this project on real labelled photographs.',f'The {len(brands)}-brand decision head is trained by this project; new regional class rows were randomly initialized.').replace('An MLP colour head was independently initialized and trained in this project over frozen general ImageNet features and original RGB/HSV spatial statistics.',colour['method']+'. Colour decisions are independently trained in this project.')
text+='\n\n## Additional regional photographs / v1.3\n\nSupplementary Commons category-labelled vehicle photographs, locally resized and automatically cropped. Authors, source URLs, licence names/URLs, changes and hashes are preserved in data/regional/source_manifest_snapshot.json and labels.jsonl. They are public photographs, never attributed to the student; raw training photographs are excluded from the portable package. Author/date session split, near-dHash deduplication and development-demo exclusion are documented in provenance.json.\n'
(ROOT/'THIRD_PARTY.md').write_text(text,encoding='utf-8')
(ROOT/'TRANSFER.txt').write_text(f'CARVISION 1.3 / София Рахманкулова\n\nПолностью распаковать TashkentCarVision-v1.3-Portable.zip.\nЗапустить CarVision/CarVision.exe.\nПереносить всю папку CarVision; не отделять EXE от _internal/models/data/samples.\nWindows10/11 x64, папка с правом записи; Python/API/ключи/сеть не нужны для анализа.\nФото -> Весь кадр или Одна машина в области -> обвести -> Анализировать.\nМарка и цвет выбираются автоматически; номера соответствуют таблице.\n{len(brands)} обученных марок; список и реальные метрики в README.md.\nOwn-фото пока0; итоговая учебная сдача ещё требует ваших данных/GitHub/live test.\n',encoding='utf-8')
print('v1.3 documentation/model card written from actual checkpoints and evaluations.')
