import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'models/vehicle_model_training.json').read_text(encoding='utf-8'));c=json.loads((root/'models/colour_calibration.json').read_text(encoding='utf-8'))
card=json.loads((root/'models/model_card.json').read_text(encoding='utf-8'));card.update(release='1.4',supported_vehicle_models=[b+' '+name for b,name in zip(m['brands'],m['labels'])],vehicle_model_test_accuracy=m['test_accuracy'],vehicle_model_test_count=m['test_count'],vehicle_model_threshold=m['threshold'],colour_test_accuracy=c['test_accuracy'],colour_reliability_threshold=c['threshold']);(root/'models/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
colour=json.loads((root/'models/colour_training.json').read_text(encoding='utf-8'));colour.update(test_accuracy=c['test_accuracy'],confusion_matrix=c['test_confusion_matrix'],cross_domain_development_accuracy=c['best']['development_accuracy'],v14_calibration=c,method='Own VCoR heads; validated full/body panel mixture retained from v1.3; calibrated uncertain-colour indication in v1.4.');(root/'models/colour_training.json').write_text(json.dumps(colour,ensure_ascii=False,indent=2),encoding='utf-8')
models=', '.join(card['supported_vehicle_models'])
text=f"""# CarVision 1.4: модели и цвет

Марка и конкретная модель выбираются автоматически. Ничего вводить для распознавания не нужно. Пример результата: Chevrolet Cobalt. Если данных или уверенности мало, модель остаётся неопределённой, но показывается наиболее вероятный вариант с «?», например Chevrolet Cobalt?. Полное имя включено в таблицу, подробности, буфер обмена и TXT/JSON/JPG. На изображении остаются номера без перекрывающихся названий.

## Реальное обучение модели автомобиля

{models}.

Своя сеть MobileNetV3 Small с avg/max pooling и головой 1152→256→16. Инициализация собственной сетью марки; готовый автомобильный model classifier не загружался. 40 эпох, AdamW features 0.00012/head 0.001, weight decay 0.02, cosine scheduler; crop 65–100%, отражение и мягкий цветовой jitter, повторная выборка для редких классов. Train/validation/test: 374/82/80 вырезок Commons с прежними группами автора/сессии. Лучший checkpoint: эпоха {m['best_epoch']}. BICUBIC resize176 и центральная вырезка160, как при обучении; среднее вероятностей обычного и зеркального вида, температура {m['temperature']} по dev.

- Validation top-1: {m['validation_accuracy']:.1%}.
- Test top-1: {m['test_accuracy']:.1%}, {m['test_count']} машин.
- При пороге {m['threshold']:.2f}: охват {m['test_coverage']:.1%}, правильных среди принятых {m['test_accepted_accuracy']:.1%}.

Это небольшой закрытый публичный тест, не точность на любых улицах. Damas: только 3 обучающих примера, без validation/test; Chevrolet Nexia: 4 train и 1 test; Cobalt:12 train, нет validation и 2 test. Для этих моделей надёжность не установлена. Другие поколения одной модели также могут отличаться. Неизвестная модель иногда ошибочно получает известное имя. Уверенность не является гарантией. Диагностические городские фотографии исключены из обучения; ответы по имени файла не подставляются. Ранняя попытка линейной головы сохранена как эксперимент, релиз использует дообученную сеть.

## Цвет

В версии 1.4 сохранены цветовые веса версии 1.3; добавлена калиброванная проверка уверенности. Свои головы обучены на исходных метках VCoR, не на псевдометках Stanford. Проверка объединяет полную машину и участки кузова, чтобы уменьшить влияние стёкол и фона. Смесь и степень доверия выбираются на validation и ранее объявленных 23 development-вырезках; тест не выбирает коэффициенты.

- VCoR test: {c['test_accuracy']:.1%}, n={c['test_count']}.
- Порог уверенного цвета выбран на validation: {c['threshold']:.2f}.
- На тесте охват уверенных цветов {c['test_reliable_coverage']:.1%}, точность среди них {c['test_reliable_accuracy']:.1%}.

«≈» в таблице означает, что цвет требует проверки; JSON содержит colour_reliable и colour_confidence. Отражения неба, соседняя машина внутри рамки и сильные тени могут давать неправильный цвет. При перекрытии выделите нужную машину через «Одна машина в области». Метод не восстанавливает невидимую часть кузова.

## Перенос и требования

Распакуйте TashkentCarVision-v1.4-Portable.zip целиком и запустите CarVision/CarVision.exe. Нужна вся папка; интернет и установленный Python для распознавания не требуются. Подробные веса, метрики и история обучения находятся в models; исходники и тесты — source. Владельцем указана София Рахматулаева, реализация/обучение выполнены с Codex.

Собственных учебных фотографий пока0; GitHub, реальный телефонный цикл и второй физический компьютер ещё не проверены. Эти требования исходного задания не выдаются за выполненные.
"""
(root/'MODEL_AND_COLOUR.md').write_text(text,encoding='utf-8')
p=root/'README.md';s=p.read_text(encoding='utf-8').replace('**93.6%**',f"**{c['test_accuracy']:.1%}**");p.write_text(s,encoding='utf-8')
p=root/'PROTOTYPE_VALIDATION.md';s=p.read_text(encoding='utf-8');s=s if s.startswith('Актуальные дополнения') else 'Актуальные дополнения и измерения версии 1.4: MODEL_AND_COLOUR.md. Ниже сохранено сравнение выпуска 1.3.\n\n'+s;p.write_text(s,encoding='utf-8')
