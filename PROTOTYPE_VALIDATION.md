Актуальные дополнения и измерения версии 1.4: MODEL_AND_COLOUR.md. Ниже сохранено сравнение выпуска 1.3.

# Измерения CarVision 1.3

Все значения относятся к фактически поставляемым весам; готовый классификатор марок/цветов не использован. Генератор: tools/write_v13_docs.py. Подробные записи: models/evaluation/public_test.json, models/regional_brand_training.json, models/colour_training.json.

| Проверка | v1.2 | v1.3 | Область применимости |
|---|---:|---:|---|
| Марка / региональный holdout | 18.8% | 78.8% | 80 вырезок Commons, несколько региональных вариантов |
| Марка / половина автомобиля | 13.8% | 60.6% | Левая и правая50%, синтетическая видимость |
| Марка / Stanford official test | 84.6% | 80.4% | 4026 эталонных вырезок, 13 представленных марок |
| Цвет / VCoR official test | 91.6% | 93.6% | 1314 изображений, 8 цветов |

Региональная адаптация: 374 train + 82 validation при replay 3102 базовых train-фото. 18 реальных эпох, Random resized crop with 55–100% visible area, horizontal flip, mild colour jitter, random erasing up to 25%. Температура уверенности 0.861 обучена на development, не test. 18 классов: Audi, BMW, BYD, Chery, Chevrolet, Daewoo, Ford, Haval, Honda, Hyundai, Kia, Lada, Mercedes-Benz, Nissan, Tesla, Toyota, Volkswagen, Volvo.

Цветовая модель: Собственные обученные цветовые головы: полная вырезка и участки кузова; веса видов выбраны по development. Готовые vehicle-colour classifiers не используются.. Train 6210, validation 1319; исходные метки VCoR, все8 цветов, exact SHA256/identical dHash очистка. Coarse simplex weights in 0.25 steps over three own trained heads and full/lower-body views; maximize mean VCoR validation and 23 separate development accuracy, VCoR>=88%; city examples and both tests excluded. Дополнительная development-диагностика23 вырезок: 91.3%; это использованная при выборе AI-разметка, не независимый итоговый эталон. Сравнения сохранены, неудачные варианты не выдаются за улучшения.

Public benchmark не доказывает совместную точность всех машин/цветов на фото Ташкента. В Commons split сгруппированы автор/дата и удалены похожие вырезки; истинные vehicle IDs не доступны. Ошибки публичных категорий/автоматических рамок возможны. Шесть демо-фотографий исключены из тренировочного списка по именам и проверке похожести, но используются для разработки и регрессий; они не невиденный тест.

Нужны собственные фотографии Ташкента, независимая проверка цвета человеком, реальные перекрытия и новые китайские модели, полный120с цикл с телефона и запуск на втором физическом Windows-компьютере. Собственных фото:0. GitHub ожидается. Не все академические требования PDF закрыты.
