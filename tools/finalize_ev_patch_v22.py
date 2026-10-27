"""Attach actual recipient evidence and honest scope limitations to the verified patch."""
from pathlib import Path
import hashlib,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]
stage=ROOT/'outputs/CarVision-EV-Patch-2.2';archive=ROOT/'CarVision-EV-Patch-2.2.zip'
verification=json.loads((ROOT/'outputs/ev22-recipient-verification/verification.json').read_text(encoding='utf-8'))
assert verification['passed']
manifest=json.loads((stage/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
assert verification['candidate_sha256']==manifest['candidate_model_sha256']
shutil.copy2(ROOT/'outputs/ev22-recipient-verification/verification.json',stage/'RELEASE_VERIFICATION.json')
shutil.copy2(ROOT/'TARGET_EV_COVERAGE_RU.txt',stage/'TARGET_EV_COVERAGE_RU.txt')
shutil.copy2(ROOT/'data/catalogue/ev_target_catalog.json',stage/'ev_target_catalog.json')
inventory=ROOT/'datasets/ev_catalog_inventory/inventory.json'
if inventory.exists():shutil.copy2(inventory,stage/'source_inventory.json')
for name in ['build_ev_target_catalog.py','audit_ev_catalog_sources.py','verify_ev_patch_v22.py','finalize_ev_patch_v22.py']:
    shutil.copy2(ROOT/'tools'/name,stage/'Development/tools'/name)
text='''СТАТУС ПАТЧА 2.2 И НОВОГО СПИСКА — 25 ОКТЯБРЯ 2026

Установщик и фактический EXE проверены на отдельной полной копии приложения
на этом же физическом компьютере без Python в PATH. Это не проверка на втором ПК.
Пройдены анализ, выделенная область, экспорт, изолированное учебное обучение
и оценка. Пользовательская разметка сохранена. 81 автоматический тест пройден.

Обучение: 10 проходов; выбран 8-й только по validation. Тест не выбирал версию.
42 семейства, 1592 фрагмента: 923 train / 213 validation / 456 test.
EV: 62 отложенных фрагмента, top-1 29,0% -> 72,6%; top-3 37,1% -> 85,5%.
Ранее поддерживаемые EV: 27 фрагментов, top-1 66,7% -> 74,1%;
top-3 при этом снизился с 85,2% до 81,5%.
Историческая общая выборка: 416 фрагментов, top-1 83,2% -> 83,4%.
На всех 456 фрагментах ложные Captiva: 14 -> 10, ложные BYD: 14 -> 8.
Это результаты классификации подготовленных фрагментов, не точность всего
детектора, не 100% распознавание и не независимый экзамен на телефонных фото.
По отдельным редким классам всего несколько тестовых изображений.

ВАЖНО: новый ночной снимок пользователя всё ещё ошибочный.
Для передней машины первый вариант остаётся Captiva, теперь около 9%.
Он не участвовал в обучении или выборе версии. Нельзя считать проблему
произвольных личных фотографий полностью решённой этим патчем.

Новый большой список пользователя содержит 325 записей семейств/кузовов.
Из них только 16 присутствуют как классы 2.2, 309 ещё не обучены.
Полный перечень и статус: TARGET_EV_COVERAGE_RU.txt / ev_target_catalog.json.
Наличие в списке не означает наличие обученной сети или хорошей точности.
Алиасы/комплектации из запроса требуют проверки по источникам до обучения.
Визуально одинаковые варианты двигателя, батареи и комплектации не должны
определяться принудительно по одному кузову. Названия не являются ответами
по файлу, ценам или месту съёмки. Цвет не ограничивает жёстко модель.

Детектор контуров и цвет в этом патче не переобучались.
Инструкция установки и отката: ПРОЧИТАЙ_УСТАНОВКА.txt.
Сервисы, ключи, платежи, Python и интернет при запуске не нужны.
'''
(stage/'PATCH_STATUS_RU.txt').write_text(text,encoding='utf-8-sig')
temporary=archive.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(stage.rglob('*')):
        if p.is_file():z.write(p,stage.name+'/'+p.relative_to(stage).as_posix())
with zipfile.ZipFile(temporary) as z:
    assert z.testzip() is None
    for entry in manifest['files']:
        assert hashlib.sha256(z.read(stage.name+'/payload/'+entry['path'])).hexdigest()==entry['sha256']
temporary.replace(archive)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256.txt').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest,verified=True)))
