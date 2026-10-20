import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=json.loads((root/'outputs/release-runtime/checks.json').read_text(encoding='utf-8'));portable=json.loads((root/'outputs/portable-v14-check.json').read_text(encoding='utf-8'));frozen=json.loads((root/'outputs/frozen-v14-check.json').read_text(encoding='utf-8'))
assert source['network_connections_blocked'] and source['filename_independence'] and source['actual_single_car_roi_passed'] and source['actual_scene_roi_passed'];assert portable['passed'] and frozen['passed']
for name,digest in source['model_sha256'].items():
 assert digest==portable['model_sha256'][name]
 assert digest==hashlib.sha256((root/'models'/name).read_bytes()).hexdigest()
 assert digest==hashlib.sha256((root/'dist/CarVision/models'/name).read_bytes()).hexdigest()
tests=(root/'outputs/tests-v14.log').read_text(encoding='utf-8');assert 'Ran 35 tests' in tests and tests.strip().endswith('OK')
text=f"""# Проверки CarVision 1.4 — 21 октября 2026

35 инженерных тестов прошли. Они проверяют поведение программы, а не точность нейросети.

- Реальные веса четырёх модулей: детектор, марка, модель, цвет; контрольные суммы совпадают в исходном проекте, проверенном EXE и переносимой копии.
- Восемь модулей внутри EXE совпадают с исходниками по рекурсивному сравнению Python-байткода.
- Шесть городских фотографий: распознавание при заблокированном socket.connect, без интернет-запросов.
- Переименование исходной фотографии не меняет ответы.
- Область поиска: координаты возвращаются в исходный кадр. Одна машина: ровно одна вырезка, без зависимости от детектора.
- JPG с отдельной легендой, JSON с моделью/кандидатами/уверенностью/colour_reliable, TXT с полным названием. Сомнительные цвета помечены в экспорте.
- Фактический EXE скопирован в {portable['path']}. PATH содержит только Windows, PYTHONHOME и PYTHONPATH очищены. Предсказания модели, марки, цвета и рамки совпадают с исходниками. Проверены оба экспорта и single-car ROI.
- Проверка ZIP: CRC, обязательные файлы, SHA256 каждого упакованного файла, совпадение четырёх весов с реально протестированными.

Время полного кадра на этом CPU: {min(r['seconds'] for r in source['photos']):.2f}–{max(r['seconds'] for r in source['photos']):.2f} с. Скорость на другом компьютере может отличаться.

Это проверка другой папки на том же физическом компьютере, не свидетельство запуска на втором устройстве. Полный телефонный цикл и очный тест20 новых фотографий не проводились.

Качество и ограничения конкретных моделей: MODEL_AND_COLOUR.md. На части присланных сцен ошибки цвета и модели сохраняются; знак вопроса обозначает предположение, а не установленную модель. Публичные данные не заменяют собственный набор для исходного задания. GitHub ожидается.
"""
(root/'VALIDATION.md').write_text(text,encoding='utf-8');print('Source, frozen, copied EXE and model checks agree.')
