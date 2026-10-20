"""Generate release instructions from completed REAL measurements, not planned results."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
training=json.loads((root/'models/public_brand_training.json').read_text())
colour_training=json.loads((root/'models/colour_training.json').read_text())
test=json.loads((root/'models/evaluation/public_test.json').read_text())
validation=json.loads((root/'models/evaluation/public_validation.json').read_text())
brands=', '.join(test['brands']);accuracy=f"{test['brand_top1_accuracy']:.1%}";epochs=len(training['history'])
detector=json.loads((root/'models/generic_detector.json').read_text())['architecture']
detector_name='Faster R-CNN ResNet50 FPN v2 COCO (640/1024 px)' if detector.startswith('fasterrcnn') else 'SSDLite320 MobileNetV3 Large COCO'
import torch,sys
sys.path.insert(0,str(root))
from carvision.core import Classifier
classifier=Classifier(len(test['brands']))
card=dict(owner='София Рахманкулова',release='1.2',brand_parameters=sum(p.numel() for p in classifier.parameters()),brand_architecture='MobileNetV3 Small + average/max pooling + 1152->256 Hardswish/dropout + 13-brand head',detector=detector_name,general_initialization='ImageNet/COCO explicitly authorized by owner; no ready-made car-brand classifier',supported_brands=test['brands'],image_size=160,colour='Original HSV body-region method; unused colour neural head is NOT trained',brand_test_accuracy=test['brand_top1_accuracy'],brand_test_images=test['images'],brand_scope=test['brand_scope'],own_tashkent_training_photos=0)
(root/'models/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
card.update(colour='Independently trained VCoR colour MLP over general ImageNet features and original RGB/HSV statistics',colour_test_accuracy=colour_training['test_accuracy'],colour_test_images=colour_training['test'],colour_classes=colour_training['classes'],colour_source=colour_training['source'],ready_made_colour_model_used=False)
(root/'models/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['epoch'] for r in training['history']],[r['train_accuracy'] for r in training['history']],label='Train');ax.plot([r['epoch'] for r in training['history']],[r['validation_accuracy'] for r in training['history']],label='Development validation');ax.set(xlabel='Epoch',ylabel='Known-brand accuracy',ylim=(0,1),title='Actual 13-brand training history');ax.legend();fig.tight_layout();fig.savefig(root/'models/evaluation/training_history.png',dpi=150);plt.close(fig)
colour=test.get('colour_visual_review',{});colour_text=f"{colour_training['test_accuracy']:.1%} на {colour_training['test']} тестовых изображениях VCoR"
scene=test.get('dominant_car_full_photo_check',{})
scene_text=(f"На {scene['images']} целых публичных тестовых фото найдена основная размеченная машина в {scene['detected']} случаях; марка основной машины с учётом пропусков/unknown верна в {scene['correct_brand']} случаях. Это проверка основной машины с IoU≥0,5, а не точность всех объектов улицы; дополнительные машины в Stanford Cars не размечены." if scene else 'Отдельная проверка детектора целых фотографий отсутствует.')
readme=f'''# Tashkent CarVision 1.2 — переносимый прототип

Владелец: **София Рахманкулова**. Обученный классификатор включён в пакет. Откройте фотографию и нажмите «Анализировать»: марка и цвет определяются автоматически, выбирать правильную марку не нужно. Windows 10/11 x64, CPU, без API, ключей, подписок и интернета при работе.

## Запуск и перенос

1. Полностью распакуйте TashkentCarVision-Portable.zip в папку с разрешением на запись.
2. Запустите CarVision.exe или Start CarVision.cmd. Python устанавливать не нужно.
3. «Анализ фото» → «Открыть фото» → «Анализировать». При необходимости выберите режим «Детальный».
4. Результаты видны в таблице и на фотографии. Выбор строки выделяет машину. «Копировать ответ» и «Экспортировать файлы» сохраняют результат.
5. На другой компьютер переносите всю папку CarVision, вместе с _internal и models. Один EXE без соседних файлов не работает.

Поддерживаются JPG/JPEG, PNG, BMP, WebP, EXIF-поворот. HEIC предварительно экспортируйте в JPEG. Ctrl+O — открыть фото, Ctrl+Enter — анализ. Результаты TXT/JSON/JPG автоматически сохраняются в outputs с уникальными именами.

## Что обучено и что измерено

Классификатор распознаёт **13 марок**: {brands}.

Обучение: {training['train_cars']} реальных машин, проверка при выборе модели: {training['validation_cars']}, {epochs} эпох. Источник — публичный Stanford Cars. Итоговый официальный test: **{test['images']} фотографий, точность марки {accuracy} на эталонных вырезках**. Это не точность всей уличной сцены. Матрица ошибок, покрытие при разных порогах, результаты по каждой марке и история обучения находятся в models/evaluation и models/public_brand_training.json.

Детектор — общий torchvision {detector_name}; марок он не определяет. Классификатор марки — MobileNetV3 Small с общей ImageNet-основой, объединением average/max pooling и собственной обученной головой на 13 марок. Готовый классификатор автомобильных марок не используется. Общие веса ImageNet/COCO явно разрешены владельцем 20.10.2026 и допускаются PDF задания.

Цвет — отдельная собственная обученная голова: общие ImageNet-признаки MobileNetV3, оригинальные пространственные статистики RGB/HSV и MLP. Обучение на {colour_training['train']} изображениях VCoR с исходными метками; validation {colour_training['validation']}; итоговая оценка — **{colour_text}**. Тест не использовался для выбора checkpoint. Восемь классов white, black, grey, red, blue, green, brown, yellow; silver→grey, beige/gold/tan→brown, orange→yellow. Pink/purple исключены как не относящиеся к заданию. Точные SHA256 и одинаковые dHash-дубликаты удалены между частями. Идентичности машин в исходном наборе не доступны, поэтому split по автомобилям подтвердить нельзя. Исходная цветовая голова внутри brand-классификатора не используется; отдельный colour_model.pt содержит реально обученную цветовую голову. HSV-пороговый метод сохранён только как резерв при отсутствии отдельной модели.

## Ограничения, которые нужно учитывать

Китайские марки, Kia и Lada не входят в 13 обученных классов. Марка ниже порога уверенности выводится как unknown; неизвестная марка иногда может быть ошибочно названа известной. Начальный порог марки 0,5 выбран на dev-validation до итогового теста; при необходимости увеличьте его. Числовой «Порог марки» — порог уверенности, а не ручной выбор производителя. Softmax не является гарантией правильности.

Далёкие, перекрытые, вечерние машины, строгий боковой ракурс и локальные модели Chevrolet/Daewoo могут определяться хуже. COCO-категории car и truck включены для автомобилей, пикапов и фургонов; детектор может также предложить грузовик. Автобусы отдельной категории не включены. Цвет чувствителен к стеклу, теням, отражениям, фону и неполной рамке. Любая фотография принимается как вход, но правильное распознавание всех машин на любых фото не гарантируется.

Режимы: fast — весь кадр, balanced — кадр и перекрывающиеся фрагменты, detail — дополнительно отражённый кадр. Внутренние обрезанные рамки отбрасываются; NMS убирает повторения. Новые ракурсы обрабатываются моделью, а не таблицей заранее записанных ответов.

## Свои данные и обучение

Для работы поставляемого прототипа переобучение не нужно. Раздел «Учебная разметка» нужен для развития модели и выполнения требования собственных фотографий Ташкента. В нём ручные правильные марки/цвета обязательны как учебные ответы, но в анализ они не передаются.

Добавьте свои фото, обведите все различимые машины, сохраните марку по эмблеме, цвет, источник own и группу съёмки. Повторные кадры одной машины/сессии объединяйте одной группой. Можно вводить новые марки. Данные сохраняются в data/images и data/labels.jsonl. Проверка групп и SHA256 предотвращает разделение одинаковых снимков между train/validation. Нужны минимум две марки и независимые группы.

«Обучение» последовательно обучает собственные головы марок/цветов и двухклассовый детектор на вашей разметке с локальной общей основой. Старые модели сохраняются в models/backups. После переобучения повторно измерьте качество; маленький набор способен ухудшить модель. «Проверка качества» показывает опубликованные измерения прототипа, если личный набор пуст; иначе запускает оценку личного набора.

```powershell
.\\CarVision.exe predict photo.jpg --mode balanced --output outputs
.\\CarVision.exe audit --labels data/labels.jsonl
.\\CarVision.exe train --labels data/labels.jsonl --output models --epochs 25 --detector-epochs 25
.\\CarVision.exe evaluate --labels data/labels.jsonl --models models --output models/evaluation
```

## Фотографии и исходники

Отдельные реальные JPG Ташкента: samples/images/01.jpg … 06.jpg и архив Tashkent-JPG-Photos.zip рядом с проектом. Авторы, источники и лицензии: samples/sources.json. Эти шесть примеров не участвовали в обучении. BYD на примере не входит в поддерживаемые марки. Подборка PDF — вспомогательный материал, не итоговый отчёт.

Редактируемый код включён в source. Для разработки создайте Python 3.12 venv и установите requirements.txt. Для запуска GUI из папки source скопируйте соседнюю папку models в source/models; готовый EXE уже использует модели из своей папки. Для повторного получения публичного набора дополнительно requirements-data.txt; это требует интернета и места, но не требуется конечному пользователю EXE. Скрипты download_prototype_data.py → prepare_prototype_data.py → prepare_generic_detector.py --accurate → train_public_brand.py --imagenet --epochs 40 --size 160 → evaluate_public_prototype.py --split validation и затем --split test воспроизводят публичный эксперимент. Источники/численности: data/public_prototype/provenance.json; метки доступны в исходном проекте, исходные публичные изображения не включены в переносимый пакет.

Для воспроизведения цветовой модели после подготовки общих весов: python tools/download_colour_data.py, затем python tools/train_colour_model.py. Это загрузит только публичные фотографии и метки VCoR, а цветовая голова будет обучена локально. Исходный test не используется для выбора checkpoint; данные не включены в переносимый пакет. Итоговые цветовые показатели — models/colour_training.json, матрица — models/evaluation/colour_confusion.png.

```mermaid
flowchart LR
    A[New photo] --> B[General COCO detector]
    B --> C[Tiles and NMS]
    C --> D[Car crop]
    D --> E[Own trained brand classifier]
    D --> F[General features and RGB HSV statistics]
    F --> G[Own trained colour head]
    E --> H[Brand and colour list]
    G --> H
```

Черновик отчёта с реальными данными: docs/CarVision_Prototype_Report.pdf (6 страниц). Он явно отмечен как прототип; GitHub и собственные фотографии пока отсутствуют.

## Соответствие заданию

Это работающий публичный прототип, **не окончательно выполненное задание**. Собственных учебных фотографий Ташкента пока 0; необходимая GitHub-ссылка ожидается. Нужны собственные данные, независимый тест телефоном до 120 секунд, подтверждение уникальности архитектуры среди команд, окончательный отчёт до 6 страниц с реальными вкладами/рефлексией/AI disclosure и публикация с тегом midterm. Подробная сверка — REQUIREMENTS_AUDIT.md. История коммитов Codex не выдаётся за историю работы студентов.
'''
(root/'README.md').write_text(readme,encoding='utf-8')
validation_doc=f'''# Реальные измерения прототипа 1.2

София Рахманкулова · 20 октября 2026. Windows x64, CPU Intel Xeon E3-1230 v3, Python 3.12.14, torch 2.6.0+cpu, torchvision 0.21.0.

- Обучение марки: {training['train_cars']} машин; dev-validation {training['validation_cars']}; {epochs} эпох; best epoch {training['best_epoch']}; dev accuracy {training['best_validation_accuracy']:.1%}.
- Отдельный официальный test: {test['images']} эталонных вырезок; brand top-1 **{accuracy}**. Этот test не участвовал в градиентах, выборе checkpoint или настройке порога.
- Цвет: **{colour_text}** с исходной разметкой VCoR; train {colour_training['train']}, validation {colour_training['validation']}. Все восемь цветов представлены. Матрица ошибок и история: models/colour_training.json. Сравнения на маленькой AI-визуальной подборке Stanford служили диагностикой разработки и не являются независимым итоговым тестом.
- Общий детектор COCO, собственный brand head, ImageNet-признаки. Готовых весов классификатора марок/цветов нет.
- 13 марок: {brands}.

Обучение полностью завершено; сохранён checkpoint с лучшей dev-validation. AdamW lr=0.0003, weight decay=0.01, cosine decay; weighted CE, smoothing=0.03; batch=32, вход160; отражения, случайный crop, яркость. Полная история: models/public_brand_training.json. Источник данных: Stanford Cars, точное сопоставление исходного изображения по Git blob SHA1 и проверка марки/official split; точные дубликаты удалены по SHA256. Источник цвета в labels.jsonl — явно отмеченная псевдометка алгоритма, которая не использовалась для обучения или измерения цветовой головы.

Ранее проверены scratch MobileNet, компактная residual-сеть и собственный HOG-детектор. Их качество не было достаточным; эти экспериментальные веса не являются рабочим классификатором релиза. После явного разрешения владельца использована общая ImageNet/COCO-основа. Сырые patch-метрики HOG не выдаются за качество поиска машин на улицах.

{scene_text}

Объём и точность на вырезках не доказывают качество всех сцен Ташкента. Китайские марки/Kia/Lada отсутствуют в обученных классах; неизвестные марки иногда ошибочно получают известную метку. Собственные фотографии и независимый тест телефоном ещё нужны. Программные тесты/переносимость описаны в VALIDATION.md; итоговый протокол сравнения сборки должен читаться вместе с этим файлом.
'''
(root/'PROTOTYPE_VALIDATION.md').write_text(validation_doc,encoding='utf-8')
audit='''# Сверка с CSAI-3210 Midterm

Владелец: София Рахманкулова. Источник: предоставленный PDF задания. Статус: обученный переносимый прототип готов для локального тестирования; полное выполнение учебного задания ещё не подтверждено.

| Требование | Реализовано / что осталось |
|---|---|
| Автоматически находить несколько машин и определять марку/цвет | Реализовано: общий COCO-детектор, фрагменты/NMS, собственный brand head, собственный метод цвета. Ручной ввод правильной марки при анализе отсутствует. |
| Любые ракурсы, освещение и дистанция | Вход поддерживается; корректность всех случаев не гарантируется. Требуется собственный разнообразный набор и проверка сложных сцен. |
| Восемь цветов и объединённые оттенки | Реализовано; отдельная собственная голова обучена на VCoR с исходными метками и отдельным тестом. Идентичности машин для проверки группировки в исходном наборе не доступны. |
| Открытый набор марок и unknown | 13 обученных марок, настраиваемый порог unknown. Китайские марки в данных отсутствуют, известны ограничения открытого распознавания. |
| Offline, без облака/API, до 120 секунд | Offline-код и EXE; никаких платных ключей. Полная передача реального снимка с телефона до выдачи ответа за 120 секунд пока не измерена. |
| Собственные фотографии Ташкента обязательны | НЕ ВЫПОЛНЕНО: собственных снимков 0. Публичные Stanford Cars и шесть городских примеров не заменяют их. |
| Публичные datasets только дополнение | Для прототипа использованы публичные данные; перед сдачей необходим собственный набор. Источники/метки/splits сохранены. |
| Своя обученная марка, общий pretrained разрешён | Выполнено для публичного прототипа. ImageNet/COCO разрешены владельцем; готовый car-brand/colour classifier не использован. |
| Уникальная архитектура среди команд | Выбранная архитектура описана; совпадения с другими командами должен проверить участник/преподаватель. |
| Настоящее обучение, данные/counts, параметры/время, ошибки/матрица | Сохранены в models и PROTOTYPE_VALIDATION.md. Публичные crop-метрики не подменяют live street test. |
| Китайские марки и анализ ошибок | Отсутствие покрытия явно указано. Нужны собственные примеры и расширение обучения. |
| Итоговый PDF ≤6 страниц: участники, GitHub на первой странице, данные, схема/обоснование/параметры, обучение/метрики/ошибки, phone-tests, запуск, reflection/AI | Генератор есть, но окончательный отчёт не готов: GitHub и собственные данные/телефонные результаты отсутствуют. Подборка фото не является итоговым отчётом. |
| GitHub: исходники/deps, weights, dataset/links+labels, реальная история | Исходники/веса/метки публичного эксперимента локально есть. URL ожидается; публикации пока нет. Коммиты Codex не выдаются за историю студентов. |
| midterm tag до 27.10.2026 23:59 Ташкент | Финальный тег пока не ставится: окончательная сдача не готова. |
| Очный тест 20 новых фото, знание кода, без посторонней помощи | Выполняется участником на экзамене; программой заранее не подтверждается. |

Пункты, требующие собственных данных, GitHub, подтверждения других команд и личного участия, нельзя честно считать завершёнными по наличию EXE.
'''
(root/'REQUIREMENTS_AUDIT.md').write_text(audit,encoding='utf-8');(root/'CHECKLIST.md').write_text(audit,encoding='utf-8')
(root/'TRANSFER.txt').write_text('''CARVISION 1.2 / София Рахманкулова

Распакуйте весь TashkentCarVision-Portable.zip на Windows 10/11 x64.
Откройте CarVision/CarVision.exe или Start CarVision.cmd.
Откройте фотографию → нажмите «Анализировать».
Выбирать правильную марку и предварительно обучать поставленный прототип не нужно.
Python, API-ключи, подписки и интернет для анализа не нужны.
Переносите всю папку вместе с _internal и models, а не отдельный EXE.

Поддерживаются 13 марок; список и измеренная точность — README.md.
Детектор общий COCO; марочный классификатор обучен отдельно; цвет — свой алгоритм.
Программа может ошибаться, особенно на неизвестных марках/плохих ракурсах.
Результаты: outputs. Отдельные фото: samples/images. Код: source.
Для разработки нужны исходники и зависимости; для готового EXE ничего устанавливать не нужно.

Прототип обучен на публичном наборе. Требование своих фото Ташкента для учебного задания ещё не выполнено.
''',encoding='utf-8')
third='''# Libraries, general weights and public photographs

Owner: София Рахманкулова. OpenAI Codex assisted with code, training, data preparation, visual colour review, tests, documentation and packaging. No student-collected data is claimed.

## Libraries

Python 3.12 (PSF), PyTorch 2.6.0 CPU (BSD-style), torchvision 0.21.0 (BSD-3-Clause), NumPy/SciPy (BSD), Pillow (HPND), Matplotlib (PSF-based), ReportLab (BSD), OpenCV (Apache-2.0), PyInstaller (GPL with bootloader exception). Installed license notices are included in licenses/. PyArrow is used only for optional public-data preparation, not required by inference.

## General pretrained parameters

The owner explicitly authorized general ImageNet/COCO weights on 2026-10-20. The assignment allows them. Official torchvision Faster R-CNN ResNet50 FPN v2 COCO_V1 supplies generic car/truck boxes. A separate SSDLite general backbone is included for optional local training on personal annotations. Official MobileNetV3 Small IMAGENET1K_V1 supplies the feature initialization. Its original ImageNet classification head is not used. The 13-brand decision head is initialized randomly and trained by this project on real labelled photographs. The colour algorithm is original code; no ready-made brand or colour model is used.

- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html

## Public training data

Stanford Cars, Krause et al. 2013, used as a public research dataset. Image mirror: https://huggingface.co/datasets/tanganke/stanford_cars . Original-image and annotation mirror: https://github.com/jhpohovey/StanfordCars-Dataset . Exact file hashes and original human bounding boxes were joined; split/brand conflicts excluded; original SHA256 duplicates removed. data/public_prototype/provenance.json records counts and mapping. This project does not own those photographs; raw public training images are excluded from the portable distribution. Research use does not establish unrestricted commercial image rights. Brand labels are original dataset labels; colour pseudo-labels are explicitly not human truth.

## City examples

Six photographs in samples/images come from Wikimedia Commons; exact authors, source pages, dates, license names and license URLs are in samples/sources.json. Preserve those credits and the applicable CC BY / CC BY-SA terms when redistributing photographs. They were not used to train or choose the brand model, and are not student-collected photos. The supplementary PDF contains resized reproductions with credits, not a final academic report.
'''
third+='''
## Independently trained colour decision

VCoR colour photographs and provided labels were obtained from https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset . The authors permit research and education use and request citation: Panetta, K., Kezebou, L., Oludare, V., Intriligator, J., Agaian, S. (2021), Artificial Intelligence for Text-Based Vehicle Search, Recognition, and Continuous Localization in Traffic Videos, AI 2(4), 684–704, https://doi.org/10.3390/ai2040041 . The dataset authors describe images collected from Google and subsequently cleaned/annotated. This project does not own or redistribute those raw training images in the portable package. Source metadata, archive SHA256, retained image hashes and exact colour mappings are preserved in data/public_prototype and source/data/public_prototype.

An MLP colour head was independently initialized and trained in this project over frozen general ImageNet features and original RGB/HSV spatial statistics. No ready-made vehicle-colour weights were obtained. Original train/validation/test folders are retained after removing exact SHA256 or identical dHash duplicates; vehicle identities are not available, so identity-separated generalization is not certified. Beige/gold/tan map to brown, silver to grey, orange to yellow; pink/purple are excluded. All eight required colours are represented. Reported VCoR test accuracy is a dataset benchmark, not a guarantee for Tashkent streets.
'''
(root/'THIRD_PARTY.md').write_text(third,encoding='utf-8')
print('Release documentation generated from actual completed training and held-out measurements.')
