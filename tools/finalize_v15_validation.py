import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=json.loads((root/'outputs/release-runtime-v15/checks.json').read_text(encoding='utf-8'))
portable=json.loads((root/'outputs/portable-v15-check.json').read_text(encoding='utf-8'))
frozen=json.loads((root/'outputs/frozen-v15-check.json').read_text(encoding='utf-8'))
assert source['network_connections_blocked'] and source['filename_independence'] and source['actual_single_car_roi_passed'] and source['actual_scene_roi_passed']
assert portable['passed'] and portable['single_car_roi_passed'] and frozen['passed'] and len(frozen['modules'])==11
for name,digest in source['model_sha256'].items():
    assert digest==portable['model_sha256'][name]
    assert digest==hashlib.sha256((root/'models'/name).read_bytes()).hexdigest()
    assert digest==hashlib.sha256((root/'dist/CarVision/models'/name).read_bytes()).hexdigest()
tests=(root/'outputs/tests-v15.log').read_text(encoding='utf-8')
assert 'Ran 43 tests' in tests and tests.strip().endswith('OK')
text=f'''# Проверки CarVision 1.5 — 21 октября 2026

43 инженерных теста прошли. Они проверяют поведение программы, а не точность нейросети.

- Шесть файлов весов совпадают в исходном проекте, проверенном EXE и переносимой копии: собственные классификаторы марки, модели и цвета, собственная голова видимого цвета, общий COCO-сегментатор и резервный общий детектор.
- Одиннадцать модулей внутри EXE совпадают с исходниками по рекурсивному сравнению Python-байткода.
- Шесть городских фотографий обработаны при заблокированном socket.connect. Переименование фотографии не меняет ответы.
- Проверены кодирование/восстановление масок, отверстия и масштаб, выбор мышью по видимой форме, устранение дубликатов и исключение соседних пикселей.
- Ручная зона одной машины использует полный контекст перекрытий. Реальный ROI соседнего с зелёной Lada белого автомобиля согласуется с анализом всего кадра. Проверена также групповая зона с возвратом координат в оригинал.
- До трёх вариантов модели доступны даже при неизвестной марке. Неуверенность видна; варианты не гарантируют правильный ответ.
- JPG содержит контуры, номера и отдельную легенду. JSON содержит маски RLE, контуры, кандидатов и уверенности. Основной TXT и кнопка задания выдают только пары brand colour; подробности сохранены отдельно в .details.txt.
- Фактический EXE проверен в отдельной папке с пробелами и кириллицей: {portable['path']}. PATH содержит только Windows, PYTHONHOME и PYTHONPATH очищены. Предсказания марки, моделей-кандидатов, цвета, координат и масок совпадают с исходниками; экспорт и single-car ROI работают.
- Архив проверяется отдельным tools/verify_packaged_release.py: CRC, обязательные файлы, SHA256 каждого упакованного файла и совпадение шести файлов весов с реально протестированными.

Время полного кадра на этом CPU: {min(r['seconds'] for r in source['photos']):.2f}–{max(r['seconds'] for r in source['photos']):.2f} с. Скорость на другом компьютере может отличаться.

Это проверка другой папки на том же физическом компьютере, не подтверждение запуска на втором устройстве. Полный телефонный цикл и очный тест 20 новых фотографий не проводились.

Метрики и ограничения нового анализа: INSTANCE_ANALYSIS.md и шестистраничный отчёт. На части сцен сохраняются ошибки цвета, марки, модели и контура. Синтетическое перекрытие не заменяет реальный уличный тест. Собственных учебных фотографий пока 0; публичный набор не закрывает это требование. GitHub, согласование архитектуры, реальные сведения о вкладе и финальная сдача ожидаются.
'''
(root/'VALIDATION.md').write_text(text,encoding='utf-8')
print('43 tests, 11 frozen modules, six model hashes and copied EXE checks agree.')
