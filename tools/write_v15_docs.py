"""Create honest v1.5 metrics, plots and portable instructions."""
import json,sys,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,COLOURS
from carvision.masked_colour import MaskedColourClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=ROOT
def read(path):return json.loads((root/path).read_text(encoding='utf-8'))
model=read('models/instance_vehicle_calibration.json');adaptation=read('models/instance_vehicle_training.json');fusion=read('models/mask_fusion.json');paint=read('models/paint_forest_experiment.json');segmenter=read('models/segmenter.json');card=read('models/model_card.json');brands=card['supported_brands']
matrix=np.zeros((len(brands),len(brands)),int)
for r in torch.load(root/'cache/visible_vehicle_evidence.pt',weights_only=True)['rows']:
 if r['split']=='test':matrix[r['brand_truth'],int(r['clean']['brand_mask'].argmax())]+=1
colours=MaskedColourClassifier(torch.load(root/'models/masked_colour.pt',weights_only=True))
rows=[r for r in torch.load(root/'cache/masked_colour_v15.pt',weights_only=True)['rows'] if r['split']=='test']
pred=colours.feature_probabilities(torch.stack([r['x'] for r in rows]));prob=torch.stack([colours.blend(r['baseline'],v) for r,v in zip(rows,pred)]);cm=np.zeros((8,8),int)
for r,p in zip(rows,prob):cm[COLOURS.index(r['colour']),int(p.argmax())]+=1
for name,m,names,title in [('instance_brand_confusion',matrix,brands,'Visible-mask brand test:80 Commons crops'),('instance_colour_confusion',cm,COLOURS,'Visible-mask colour test:240 VCoR cars')]:
 fig,ax=plt.subplots(figsize=(10,8));ax.imshow(m,cmap='Blues');ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,xlabel='Prediction',ylabel='Reference',title=title);plt.setp(ax.get_xticklabels(),rotation=65,ha='right',fontsize=8);plt.setp(ax.get_yticklabels(),fontsize=8)
 for i in range(len(names)):
  for j in range(len(names)):ax.text(j,i,str(m[i,j]),ha='center',va='center',fontsize=6 if len(names)>10 else 10,color='white' if m[i,j]>m.max()/2 else '#243746')
 fig.tight_layout();fig.savefig(root/'models/evaluation'/f'{name}.png',dpi=160);plt.close(fig)
chinese={b:dict(test_count=int(matrix[brands.index(b)].sum()),correct=int(matrix[brands.index(b),brands.index(b)])) for b in ['BYD','Chery','Haval']}
(root/'models/evaluation/instance_metrics.json').write_text(json.dumps(dict(brand_confusion=matrix.tolist(),brand_test_accuracy=float(np.trace(matrix)/matrix.sum()),brand_test_count=80,chinese=chinese,colour_confusion=cm.tolist(),colour_test_accuracy=float(np.trace(cm)/cm.sum()),colour_test_count=240),indent=2))
card.update(release='1.5',segmentation=segmenter,instance_brand_accuracy=fusion['brand_test']['clean']['accuracy'],instance_model_metrics=model['metrics']['test'],instance_colour_accuracy=paint['test_accuracy'],instance_colour_test_count=paint['test_count'],model_rejection_threshold=model['threshold'],visible_masks=True)
(root/'models/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
text=f"""# CarVision 1.5 — видимая форма автомобиля

Откройте фото → «Анализировать фото». Программа выделяет видимые контуры отдельных машин. Один видимый пиксель принадлежит только одной машине; цвет соседнего кузова не включается в её цветовую область. Координатные прямоугольники остаются в JSON для вырезки, но не являются масками цвета.

## Выбор машины и настройки

«Выбрать зону мышью» → обведите нужную машину → «Распознать в зоне». Прямоугольник указывает, какую машину искать. Граница определяется автоматически с учётом полного кадра, цвет и классификация относятся к выбранной машине внутри зоны. Если контур не найден, выполняется поиск на увеличенном участке, затем приблизительный GrabCut с предупреждением. Это не подтверждает, что выбран автомобиль. Скрытые части не восстанавливаются.

«Вся фотография» снимает зону. Нажмите контур или строку результата для выбора; «Крупнее» показывает изолированный автомобиль. Подробные настройки сначала скрыты. Уверенность поиска: больше значение — меньше сомнительных объектов. Уверенность марки: ниже выбранного значения марка не установлена. Начните с0.5. Большие пороги могут скрыть реальные машины. Проценты сети не гарантируют правильность.

Справа показываются до трёх изученных вариантов модели, включая неустановленную марку. Изучено16 вариантов. «?» — предположение, «≈» — сомнительный цвет; также виден второй цвет. Для части из18 марок нет классификатора конкретных моделей.

## Формат задания и перенос

«Марки / цвета для задания» и основной TXT содержат ровно пары brand colour. «Копировать варианты» и .details.txt содержат модели и уверенность. JSON: все варианты, маски RLE, контуры и координаты; JPG: контуры и отдельная легенда без длинных надписей поверх машин.

Распакуйте TashkentCarVision-v1.5-Portable.zip целиком; запускайте CarVision/CarVision.exe. Переносите всю папку с _internal/models/data/samples. Windows10/11 x64, папка с правом записи. Интернет, API, ключи и установленный Python для распознавания не нужны.

## Реальное обучение

Общий COCO Mask R-CNN ResNet50 FPN v2, {segmenter['parameters']:,} параметров: только поиск/видимые маски car/truck. Это разрешённые общие веса. Готового классификатора автомобильных марок, моделей или цветов нет.

Своя сеть марки MobileNetV3 Small обучена на Stanford+Commons. Сеть конкретных моделей дообучена24 эпохи на374 Commons train, split374/82/80 сохранён по группам. Лучшая эпоха {adaptation['best_epoch']}. AdamW feature0.00006/head0.0003, wd0.015, batch32, CE smoothing0.05, cosine24. Аугментации:60% изолированных видов,50% искусственных перекрытий другой тренировочной машиной, crop65–100%, отражение/jitter. Уверенность калибруется только на validation.

Цвет: свои прежние VCoR-головы плюс собственный лес64 деревьев по572 признакам HSV/Lab видимых пикселей. Лес обучен с нуля на640 VCoR train; validation160/test240, по80/20/30 машин каждого цвета. Это поднаборы прежнего VCoR, не новые собственные фото. Общий COCO дал1035/1040 масок,5 — GrabCut. Перед лесом проверена своя MLP; эксперименты сохранены. Выбор леса, смеси и порогов — по validation без городских примеров. sklearn нужен только для обучения; EXE использует экспортированные массивы деревьев.

## Измерения и ограничения

- Марка после масок: {fusion['brand_test']['clean']['accuracy']:.1%} на80 публичных вырезках,7 марок представлены.
- Модель,80 чистых вырезок: top-1 {model['metrics']['test']['clean']['top1']:.1%}; правильная модель среди трёх {model['metrics']['test']['clean']['top3']:.1%}.
- Искусственное перекрытие27% нижнего угла: top-1 {model['metrics']['test']['partial']['top1']:.1%}, до адаптации {adaptation['baseline']['test']['partial']:.1%}; top-3 {model['metrics']['test']['partial']['top3']:.1%}. Это искусственный тест, не настоящие уличные перекрытия.
- Цвет на240 сбалансированных масочных VCoR-вырезках: {paint['test_accuracy']:.1%}; уверенный охват {paint['test_coverage']:.1%}, точность среди принятых {paint['test_accepted_accuracy']:.1%}. Старые93.6% получены на1314 полных VCoR-изображениях: протоколы различаются.

Китайские бренды в дополнительном тесте: {chinese}. Малые числа не дают надёжной уличной оценки. Многие китайские марки ещё не обучены. Тени, отражения, маленькие/скрытые машины и неизвестные поколения всё ещё дают ошибки. Маски не заменяют достаточный набор данных. Городские примеры служат диагностикой разработки, не независимым тестом; они не участвуют в обучении или подстановке по имени файла.

## Остаток академических требований

Собственных учебных фото пока0; GitHub ожидается. Уникальность среди команд, полный телефонный цикл≤120с,20 невиденных фото и запуск на другом физическом компьютере требуют фактической проверки. PDF пока шестистраничный черновик. Владелец — София Рахманкулова; помощь Codex раскрыта. Исходники, подготовка данных и обучение — source; источники/лицензии — data, models/*.json, THIRD_PARTY.md и samples/sources.json.
"""
(root/'INSTANCE_ANALYSIS.md').write_text(text,encoding='utf-8')
(root/'README.md').write_text("""# Tashkent CarVision 1.5

Владелец: София Рахманкулова. Локальный Windows10/11 x64 прототип: контуры автомобилей, автоматические марки, варианты моделей и восемь цветов. Без платных API, ключей и сети при распознавании.

Распакуйте **TashkentCarVision-v1.5-Portable.zip** целиком и запустите **CarVision/CarVision.exe**. Переносите всю папку; не запускайте EXE прямо из ZIP.

1. «Открыть фото» → «Анализировать фото» для всего кадра.
2. «Выбрать зону мышью» → обведите машину → «Распознать в зоне».
3. Нажмите контур или строку: справа три варианта модели и варианты цвета. Проценты не гарантируют правильность.
4. «Вся фотография» снимает зону; «Крупнее» показывает изолированную машину.
5. «Подробные настройки» раскрывают режим мелких машин и пояснённые пороги. Начните с0.5.
6. «Марки / цвета для задания» копирует требуемые пары. «Копировать варианты» — подробности. Экспорт: TXT, details.txt, JSON, JPG.

Метод, обучение, измерения и ограничения: [INSTANCE_ANALYSIS.md](INSTANCE_ANALYSIS.md). Проверки: VALIDATION.md. Собственных учебных фото0; GitHub ожидается. Не все академические условия закрыты; отчёт docs/CarVision_Prototype_Report.pdf — шестистраничный черновик.

«Учебная разметка» сохраняет свои фото, марки/цвета, источник и группу машины/сессии. Для разработки из source скопируйте соседнюю папку models в source/models, создайте Python-окружение и установите requirements.txt; для инструментов подготовки/обучения также requirements-data.txt. Маски: prepare_instance_segmenter.py. Модели: train_vehicle_model_network.py → train_instance_vehicle_models.py → calibrate_instance_model_confidence.py. Цвет: train_masked_colour.py → calibrate_masked_colour_gate.py → train_paint_forest.py. Split/источники/метрики: models/*.json и data/*.jsonl. Скачанные публичные учебные фотографии в EXE не включены.

Предыдущие измерения1.3/1.4 сохранены в PROTOTYPE_VALIDATION.md и MODEL_AND_COLOUR.md как история. Актуальный метод — INSTANCE_ANALYSIS.md. Лицензии: THIRD_PARTY.md, licenses и samples/sources.json. CLI: CarVision.exe predict photo.jpg --mode fast --region 100 100 800 600 --single --output outputs.
""",encoding='utf-8')
(root/'TRANSFER.txt').write_text("""CARVISION 1.5 / София Рахманкулова
Распаковать TashkentCarVision-v1.5-Portable.zip целиком; запустить CarVision/CarVision.exe.
Переносить всю папку, включая _internal/models/data/samples.
Фото → Анализировать фото. Зона → Выбрать зону мышью → обвести → Распознать в зоне.
Модели и цвет определяются автоматически. Справа — три вероятные модели.
«?» — предположение, «≈» — сомнительный цвет. Маски описывают видимые части.
Пары brand colour копируются отдельной кнопкой, варианты — «Копировать варианты».
Без установленного Python, сети и API. Windows10/11 x64, папка с правом записи.
Ошибки сохраняются; собственные фото/GitHub/телефонный тест ещё нужны.
""",encoding='utf-8')
print('v1.5 docs and actual confusion plots created',flush=True)
