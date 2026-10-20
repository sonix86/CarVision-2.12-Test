"""Publish engineering evidence only after the actual model and portable checks passed."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
checks=json.loads((root/'outputs/release-runtime/checks.json').read_text())
portable=json.loads((root/'outputs/portable-v12-check.json').read_text())
tests=(root/'outputs/tests-v12.log').read_text();assert 'Ran 22 tests' in tests and '\nOK' in tests
assert checks['network_connections_blocked'] and checks['filename_independence'] and not checks['synthetic_weights']
assert portable['passed']
assert json.loads((root/'outputs/frozen-v12-check.json').read_text())['passed']
for name,digest in checks['model_sha256'].items():
    assert hashlib.sha256((root/'models'/name).read_bytes()).hexdigest()==digest,'Runtime check used different model: '+name
    assert portable['model_sha256'][name]==digest,'Portable model differs: '+name
times=[p['seconds'] for p in checks['photos']]
text=f'''# Инженерные проверки CarVision 1.2 — 20 октября 2026

Владелец: София Рахматулаева. Windows x64; Python 3.12.14, torch 2.6.0+cpu, torchvision 0.21.0. Общие веса ImageNet/COCO разрешены владельцем; классификатор марок обучен в этом проекте.

- **22 регрессионных тестов прошли**: автоматический вывод марки/цвета без ручного выбора, категории COCO car/truck с исключением bus, координаты фрагментов и швы/NMS, восемь цветов на контрольных полях, split дубликатов, некорректные рамки, уникальные экспорты, формула оценки и эквиваленты марок, ограничения итогового отчёта.
- GUI: четыре раздела, загрузка своего фото/примеров, корректные диалоги и пути, копирование/экспорт, сохранение и отмена метки, группы, масштабирование координат, размещение кнопок при минимальном окне, команды audit/evaluate/train и остановка дочернего процесса проверены программно во временной папке.
- Настоящие поставляемые модели проверены на **{len(checks['photos'])} отдельных городских JPG**, с программно заблокированным socket.connect. Файлы weights загружаются локально; скрытого обращения к интернету нет.
- Одинаковое изображение после переименования даёт те же результаты: имя/путь не служат таблицей готовых ответов.
- Время режима fast на этих шести изображениях на данном CPU: от **{min(times):.2f} до {max(times):.2f} с**, без передачи с телефона. Это измерение скорости, а не подтверждённая точность всех машин на этих улицах.
- EXE перенесён в отдельную папку с пробелами в пути; PYTHONHOME/PYTHONPATH очищены, PATH содержит только Windows. Проверены реальные weights, inference и создание TXT/JSON/JPG. Исходный Python или папка разработки не нужны.
- Запуск собранного кода сравнивается с исходным по результатам на том же снимке. Дополнительно сигнатуры bytecode модулей core/gui/training/detector_training/reporting внутри PyInstaller проверены на соответствие исходникам.
- Реальное обучение/точность/матрица ошибок и ограничения тестовой выборки описаны отдельно в PROTOTYPE_VALIDATION.md и models/evaluation/public_test.json. Синтетические демонстрационные веса не включены в релиз.

Чего эти проверки не подтверждают: запуск на другом физическом компьютере, отсутствие любых ошибок на любых фото, 20 новых экзаменационных сцен, человеческую независимую точность цвета и полный цикл телефон → передача → ответ за 120 секунд. Собственных фото Ташкента пока нет. Визуальный захват GUI системным инструментом ранее завершался FrameArrived timed out; проверка layout/кнопок программная, визуальная приёмка всего окна не заявляется.

Исходный PDF требований сверён в REQUIREMENTS_AUDIT.md. Итоговая учебная сдача требует собственного набора, GitHub, уникальности среди команд, отчёта и личного live test.
'''
(root/'VALIDATION.md').write_text(text,encoding='utf-8')
print('Validation document written from successful real-model and portable checks.')
