"""Assemble release evidence from completed checks, never invent success."""
import hashlib,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=json.loads((root/'outputs/release-runtime-v19/checks.json').read_text(encoding='utf-8'));portable=json.loads((root/'outputs/portable-v19-check.json').read_text(encoding='utf-8'));frozen=json.loads((root/'outputs/frozen-v19-check.json').read_text(encoding='utf-8'));card=json.loads((root/'models/model_card.json').read_text(encoding='utf-8'));split=json.loads((root/'models/ev_split_audit.json').read_text(encoding='utf-8'));e2e=json.loads((root/'models/ev_end_to_end.json').read_text(encoding='utf-8'))
assert all(source[k] for k in ['network_connections_blocked','filename_independence','actual_single_car_roi_passed','actual_scene_roi_passed'])
assert portable['passed'] and portable['single_car_roi_passed'] and frozen['passed'] and len(frozen['modules'])==12
assert card['owner']=='София Рахманкулова' and card['release']=='1.9' and split['passed'] and split['group_intersections']==0
n=len(card['supported_vehicle_models']);ranked=[r for p in source['photos'] for r in p['cars'] if r.get('model_ranking')];assert ranked and all(len(r['model_ranking'])==n for r in ranked)
for name,sha in source['model_sha256'].items():
 assert sha==portable['model_sha256'][name]==hashlib.sha256((root/'models'/name).read_bytes()).hexdigest()==hashlib.sha256((root/'dist/CarVision/models'/name).read_bytes()).hexdigest()
tests=(root/'outputs/tests-v19.log').read_text(encoding='utf-8');assert tests.strip().endswith('OK');count=int(re.search(r'Ran (\d+) tests',tests)[1]);assert count>=65
text=f'''# Проверка переносимой сборки1.9

Владелец: София Рахманкулова. {count} инженерных тестов прошли. Точность распознавания ими не измеряется.

- Все12модулей в EXE совпадают с исходниками по рекурсивному сравнению байткода.
- Семь реально используемых файлов весов совпадают в исходном проекте, EXE и отдельной переносимой копии.
- Реальный EXE проверен в другой папке с кириллицей/пробелами: {portable['path']}. В PATH только Windows, PYTHONHOME/PYTHONPATH очищены. Подтверждены совпадение предсказаний, ROI одной машины и экспорт.
- Шесть городских сцен обработаны при заблокированном socket.connect, проверены переименование файла, кеш, групповой ROI, сохранение цвета белой машины рядом с зелёной Lada. Это диагностические кадры, не обучающие ответы и не выбор гиперпараметров.
- RLE-маски сохраняют отверстия и масштаб. Проверены исключение соседнего цвета, людей, объединение видимых частей с общим предложением детектора, сохранение разных соседних машин, ручное объединение без заполнения промежутка и пересчёт ответа. Ручное действие отменяется.
- Исправлено сохранение известной марки (например, Audi), если конкретные модели этой марки не представлены в классификаторе моделей. Альтернативы других семейств при этом не подтверждают название.
- Активно {n} семейств. Выбор гибридного классификатора сделан по очищенной validation; не по имени файла, демонстрационному изображению или test. Две ограниченные нейросетевые попытки завершены; дополнительные раунды не запускались.
-244реальных лицензированных фото,194пригодных новых изображения; авторы/лицензии/исключения в отдельном JPG-архиве. Групповых пересечений учтённых сессий0. Общие train/validation/test:779/161/416. Нет утверждения о наличии истинных vehicle-ID для всех публичных изображений.

Важный практический предел: на27новых целых test-фото найден хотя бы один автомобиль во всех кадрах; правильная модель первой {e2e['top1_count']}/27, среди трёх {e2e['top3_count']}/27. Это проверка главного видимого объекта, не всех машин/контуров сцены. Единственное подтверждённое имя оказалось ошибочным; китайские модели остаются экспериментальными. Цвет на новых фото не имеет проверенной целевой разметки и его новая точность не заявляется. На старом test улучшились целая машина и верхние55%, но самые узкие35% немного ухудшились. Все цифры, включая регрессии, в README.md и models/release_v19_comparison.json.

Скорость шести городских сцен на этом CPU при параллельной работе: {min(p['seconds'] for p in source['photos']):.2f}–{max(p['seconds'] for p in source['photos']):.2f}с. Это не полный телефонный цикл задания и не измерение другого ПК. Вывод «всегда точно распознаёт любой автомобиль» не сделан.

Собственных учебных фото студента0. Публичные фото не закрывают это условие PDF. GitHub-ссылка, очные20новых фото/телефонный цикл, второй физический ПК, согласование ML-семейства и окончательная сдача остаются неподтверждёнными. Исторический шестистраничный PDF1.5не заменяет отчёт о1.9. Бесплатные локальные вычисления и открытые источники; платные API, покупки, кредитные/платные сбросы лимита и генерация фото не использовались.
'''
(root/'VALIDATION.md').write_text(text,encoding='utf-8');print('RELEASE CHECKS PASSED:',count,'tests,12frozen modules,7model hashes,copiedEXE,offline/ROI/group audit; limitations retained')
