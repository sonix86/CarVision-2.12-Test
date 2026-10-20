# Tashkent CarVision

Переносимое Windows x64 приложение для CSAI-3210. Все вычисления при распознавании выполняются на CPU локально. Подписок, API-ключей и облачных вызовов нет.

## Текущее состояние — прочитайте сначала

Детектор машин COCO установлен. **Обученного классификатора марок пока нет: пользовательский датасет не предоставлен.** До обучения марки честно выводятся как `unknown`, цвет — экспериментальный HSV baseline. Это НЕ завершённая экзаменационная модель. Никакая точность на реальных машинах пока не заявляется. Для выполнения задания нужны собственные фотографии Ташкента, обучение, оценка и реальные телефонные тесты. Синтетические инженерные тесты не заменяют эти этапы.

## Запуск без Python и интернета

1. Распакуйте **всю** папку `CarVision` из `TashkentCarVision-Portable.zip` в доступное для записи место на Windows 10/11 x64.
2. Запустите `CarVision.exe` или `Start CarVision.cmd`. EXE использует включённый Python, устанавливать Python отдельно не надо.
3. «Распознавание» → «Открыть фото» → «Распознать». Список содержит отдельную запись на каждую найденную машину. «Копировать ответ» копирует `BYD white; Chevrolet grey`.
4. В `outputs` сохраняются TXT, JSON (рамки, уверенность, время), JPG с рамками.

Переносите EXE **вместе с** `_internal`, `models`, `cache`, `data`: один отдельный EXE не содержит все необходимые файлы. Для запуска подготовленного пакета сеть не нужна. Классификатор в `models/classifier.pt` появится только после настоящего обучения. После обучения пересоберите пакет или перенесите актуальную папку `models` в переносимую сборку.

Ориентир по памяти: 4 GB RAM минимум, 8 GB предпочтительно; около 1 GB места. Производительность и работоспособность на другом физическом компьютере необходимо проверить. Для фото HEIC сначала экспортируйте JPEG с телефона. JPEG/PNG/BMP/WebP поддерживаются. Окно консоли используется для диагностики CLI.

## Данные и разметка

1. Соберите собственные фотографии Ташкента, включая китайские марки, перед/зад/бок, расстояния, день/вечер, повторную съёмку фото с экрана телефоном. Не снимайте специально лица и номера. Публичные наборы допустимы только дополнительно, с лицензией и ссылкой.
2. Во вкладке «Разметка данных» добавьте фотографию. Она сохраняется в `data/images` после исправления ориентации EXIF.
3. Обведите каждую различимую машину; выберите марку по **эмблеме**, цвет и сохраните. Марки можно вписывать дополнительно. Не объединяйте Jetour/Chery, Omoda/Chery, Daewoo/Chevrolet. Пишите English manufacturer badge names consistently.
4. Укажите `source`: `own`, `team:Name`, `public:DatasetURL`. Укажите `group`: одинаковый идентификатор для одной машины/сессии и повторной съёмки одного изображения с экрана. Поменяйте группу для независимой съёмки. На одном фото все машины должны оставаться в одной группе. Метки сохраняются сразу, редактируются в `data/labels.jsonl`; ошибочную строку удалите/исправьте вручную.

Одна строка JSON на одну машину:

```json
{"image":"images/example.png","box":[100,80,600,400],"brand":"BYD","colour":"white","source":"own","group":"session-2026-10-20-01"}
```

Координаты в исходном изображении после EXIF: `[left, top, right, bottom]`. 8 цветов: `white, black, grey, red, blue, green, brown, yellow`. Silver→grey, beige/gold→brown, orange→yellow. Все машины на одном фото размечайте; слишком маленькие/нераспознаваемые исключайте. Наугад марку не ставьте. Открытие фото в разметке без сохранения меток не добавляет его в обучение.

## Обучение и оценка

В EXE: вкладка «Обучение и проверка» → «Проверить данные» → «Начать обучение». Журнал показывает реальные эпохи, потери и validation accuracy. Данные должны содержать минимум две марки и независимые группы с марками, присутствующими в train. Для практической точности ориентируйтесь на несколько сотен машин на марку, как советует задание. При недостаточном покрытии validation сборку нельзя считать проверенной.

```powershell
.\CarVision.exe audit --labels data/labels.jsonl
.\CarVision.exe train --labels data/labels.jsonl --output models --epochs 25
.\CarVision.exe evaluate --labels data/labels.jsonl --models models --output models/evaluation
.\CarVision.exe predict phone.jpg --models models --output outputs
```

Приложение создаёт `classifier.pt`, `split.json`, `training.json`, `evaluation/metrics.json`, `evaluation/brand_confusion.png`. Не меняйте датасет между обучением и оценкой: при изменении меток переобучите модель. Валидируются реальные вырезки машин, **не** качество детекции на целых фото. Проверяйте полный конвейер отдельно на новых фотографиях и запишите пропуски/лишние машины, итоговый балл и время. Порог `--threshold 0.65` для unknown и `--detection 0.6` — начальные значения; подбирайте на validation до экзамена. Уверенность softmax не является калиброванной вероятностью правильности.

## Архитектура

```text
Phone JPEG → EXIF correction, downscale ≤1920
 → SSDLite320 / MobileNetV3 Large COCO car detector (category 3)
 → confidence filter + NMS IoU 0.45 → crops
 → fine-tuned ImageNet MobileNetV3 Small features
 → concatenate global average + max pooling (1152)
 → Linear 256 + Hardswish + dropout 0.3
 → team-trained brand head AND 8-colour head
 → unknown threshold → list + boxes + JSON/TXT
```

AdamW, LR 3e-4, weight decay .01, cosine schedule; weighted brand cross-entropy + 0.5 colour cross-entropy; smoothing .05; 25 epochs, batch 16, seed 3210. Flip, mild affine, brightness and contrast; hue augmentation не используется, чтобы сохранять цвет. Лучший checkpoint выбирается по `.6 brand_accuracy + .4 colour_accuracy` на validation. Нет независимого test-set результата до ваших телефонных испытаний. General COCO detector permitted; ready-made brand/colour models are not used. HSV baseline is our own histogram of saturation/value/hue in central body region; it is vulnerable to windows, reflections and lighting and is replaced by the trained colour head after training.

Проверьте уникальность backbone-family + multi-car strategy у преподавателя: из PDF нельзя узнать архитектуры других команд. Если другой команде уже назначен MobileNet + detector/classifier, согласуйте другой backbone и адаптируйте код до обучения.

## Исходники: установка на чистом компьютере

Windows x64, Python 3.12.14 x64 (или более свежий исправленный 3.12), интернет только для установки/первичной загрузки общих весов. Python 3.12.0 не используйте для EXE: при проверке возникла ошибка упаковки PyTorch.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch.py setup
.\.venv\Scripts\python.exe launch.py gui
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -ExecutionPolicy Bypass -File build.ps1
```

В переносимую сборку веса и ImageNet cache уже включены. `setup` не требуется для её распознавания. Пакет CPU совместим с компьютерами без видеокарты. Инференс никогда не вызывает `setup` автоматически и завершится понятной ошибкой при отсутствии детектора. Все загрузки отделены от predict. Полные фактические версии записаны в `requirements-lock.txt`.

## Отчёт и сдача

`CHECKLIST.md` сопоставляет требования задания с реализацией и оставшимися действиями. Итоговый отчёт нельзя заполнить выдуманными данными.

После реального обучения заполните JSON с полями `brand_choice, collection_labelling, errors, chinese_brands, phone_tests, reflection` (все значения — ваш содержательный текст). Затем:

```powershell
.\CarVision.exe report --team "Real student names" --repo "https://github.com/USER/REPO" --contributions "Actual contributions" --analysis analysis.json --output outputs/report.pdf
```

Шаблон генератора содержит шесть логических страниц. **Проверьте реальное число страниц ≤6 и вид каждой страницы**: большие таблицы/длинный текст могут вызвать переполнение. При необходимости сократите текст и образцы. Включите ошибки китайских марок, реальные результаты пересъёмки экрана телефоном и AI-use disclosure. Проверьте точность всех утверждений и личные вклады. Если нужны русские имена/текст в PDF, генератор использует системный шрифт Windows Arial с поддержкой кириллицы.

Сдача: единственный PDF на WorldClassRoom до **27 октября 2026, 23:59 (Ташкент)**; полный GitHub URL на первой странице. Код, веса, данные/ссылки и label file в репозитории. Крупные файлы не помещайте в обычный Git (>100 MB): используйте Git LFS или доступные ссылки/релизы. Историю коммитов создают настоящие участники по мере работы. После проверки сами поставьте `git tag midterm` и отправьте тег. Не выдавайте AI-коммиты за вклад студентов. Live test 28 октября, NH311: без AI-помощи, один запуск на фото, 2 минуты, оба участника должны объяснить весь код.

## Файлы

`core.py` — inference, модель, HSV; `data.py` — проверка/групповой split; `training.py` — обучение/оценка; `gui.py` — GUI/разметка; `scoring.py` — оптимальное сопоставление по формуле задания; `reporting.py` — отчёт из результатов; `launch.py` — CLI; `tests` — инженерные проверки; `build.ps1` — переносимая сборка.
