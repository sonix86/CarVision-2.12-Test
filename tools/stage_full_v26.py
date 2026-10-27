"""Overlay the selected verified model and new runtime onto a full copied app."""
import hashlib,json,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];target=ROOT/'outputs/CarVision-2.6-Full/CarVision';out=ROOT/'models/ev_views_v26';load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert target.is_dir() and (target/'_internal').is_dir(),'Complete base copy first'
with (out/'stage_once.json').open('x') as f:json.dump(dict(target=str(target)),f)
decision=load(out/'release_decision.json');model=ROOT/decision['selected_model'];assert sha(model)==decision['selected_model_sha256'];build=ROOT/'outputs/build-v26/dist/CarVision';frozen=load(ROOT/'outputs/frozen-v26-check.json');assert frozen['passed'] and frozen['exe_sha256']==sha(build/'CarVision.exe')
for p in build.rglob('*'):
    if p.is_file():
        dest=target/p.relative_to(build);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha(dest)!=sha(p):shutil.copy2(p,dest)
shutil.copy2(model,target/'models/semantic_make.pt')
for directory in ('carvision','tools','tests'):
    for p in (ROOT/directory).glob('*.py'):
        dest=target/directory/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
for name in ('launch.py','CarVision-v26.spec','requirements-semantic-v25.txt','CHECKLIST.md'):
    shutil.copy2(ROOT/name,target/name)
dev=target/'Development/EV26';dev.mkdir(parents=True,exist_ok=True)
for p in out.glob('*.json'):shutil.copy2(p,dev/p.name)
for name in ('manifest.json','run_budget.json','summary.json'):
    p=ROOT/'datasets/ev_views_v26'/name
    if p.exists():shutil.copy2(p,dev/('sources_'+name))
# The small newly licensed public corpus may travel with its attribution.
public=target/'datasets/ev_views_v26';public.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'datasets/ev_views_v26').glob('*'):
    if p.is_file() and p.suffix in ('.jpg','.json'):shutil.copy2(p,public/p.name)
ck=torch.load(model,weights_only=True);base_summary=load(out/'dataset.json');counts={}
data=torch.load(ROOT/'cache/ev_views_v26_data.pt',weights_only=True)
for r in data['rows']:
    counts.setdefault(r['brand'],dict(train=0,validation=0,test=0));counts[r['brand']][r['split']]+=1
catalog=load(target/'models/make_catalogue.json');catalog.update(version='2.6',learned_make_labels=ck['brands'],confirmed_gate_enabled=[b for b in ck['brands'] if ck['thresholds'][b]<1 or ck.get('retrieval',{}).get('fallback_thresholds',{}).get(b,1)<1]);(target/'models/make_catalogue.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
version=dict(version='2.6',owner='София Рахманкулова',head_kind=ck.get('head_kind','cosine'),selected_model_sha256=sha(model),new_training_promoted=decision['new_model_promoted'],decision=decision,make_counts=counts,exact_model_families=42,requested_catalogue_entries=325,all_requested_models_learned=False,own_data_primary=False)
(target/'models/model_version.json').write_text(json.dumps(version,ensure_ascii=False,indent=2),encoding='utf-8')
active=load(target/'Development/ACTIVE_MODEL_SHA256.json');active['semantic_make.pt']=sha(model);(target/'Development/ACTIVE_MODEL_SHA256.json').write_text(json.dumps(active,indent=2),encoding='utf-8')
check=(ROOT/'handoff/CHECK_ON_NEW_PC.ps1').read_text(encoding='utf-8').replace("version='2.1'","version='2.6'");(target/'CHECK_ON_NEW_PC.ps1').write_text(check,encoding='utf-8-sig');shutil.copy2(ROOT/'handoff/CHECK_ON_NEW_PC.cmd',target/'CHECK_ON_NEW_PC.cmd')
guide=f'''CarVision 2.6 — полная переносимая учебная версия
София Рахманкулова

ЗАПУСК НА ДРУГОМ КОМПЬЮТЕРЕ
1. Скопируйте CarVision-2.6-Full.zip на Windows 10/11 x64.
2. Распакуйте ВЕСЬ архив в новую папку, например C:\\CarVision26.
3. Откройте папку CarVision и запустите CarVision.exe.
Нельзя запускать EXE внутри ZIP или переносить только EXE: рядом нужны
_internal, models, data, samples и другие включённые папки.
Устанавливать старые патчи, запускать APPLY*.cmd, EXAM.cmd, pip или Python
для обычного анализа не нужно. Веса уже обучены и находятся в models.
Внешний API, интернет, ключи и оплата для анализа не нужны.
Старую папку программы оставьте отдельно; поверх неё этот архив не распаковывайте.

АНАЛИЗ
Открыть фото -> Анализировать фото. Контуры показывают найденные машины;
нажмите контур или строку, чтобы посмотреть марку, цвет и варианты.
Сначала определяется марка. Точная модель выводится, когда она изучена и
достаточно подтверждена. Каталог 325 названий не означает знание всех моделей.
Unknown/«марка не подтверждена» означает, что подтверждённого ответа нет.
Проценты и верхняя гипотеза не гарантируют правильность.
Для нужной машины: Выбрать зону мышью -> обвести её -> Анализировать фото.
«Крупнее / выбранная машина» позволяет посмотреть выделенный автомобиль.
«Ответ для задания» копирует пары марка-цвет. «Копировать варианты» добавляет
гипотезы; «Сохранить результаты» создаёт файлы в outputs.
В подробных настройках можно выбрать быстрый, сбалансированный или подробный
поиск. Подробный режим дольше обрабатывает сложные сцены с мелкими машинами.

ПРОВЕРКА ПОСЛЕ ПЕРЕНОСА
По желанию запустите CHECK_ON_NEW_PC.cmd и дождитесь PASS.
Он проверяет EXE, контрольные суммы весов, анализ, зону, экспорт и учебное
обучение в отдельной папке. Рабочие веса эта проверка не переобучает.
Затем отключите Wi-Fi и проверьте новое фото вручную.
Технический PASS не является оценкой правильности ответов на вашей фотографии.

ОБУЧЕНИЕ
Загрузка фотографии для анализа сама по себе не обучает сеть.
В «Учебной разметке» сохраните правильные рамки, марки, цвета и группы съёмок.
Одну машину из серии и близкие кадры одной поездки не делите между обучением
и проверкой. Подписи должны быть проверены, а не взяты из догадок программы.
Кнопка «Обучить учебного кандидата» создаёт отдельную учебную архитектуру в
models/own_training_candidate. Она не заменяет расширенную рабочую модель
автоматически. Это ограничение текущей учебной функции, а не обучение всех
325 моделей одной кнопкой. Новая основная модель обучалась по рецептам tools/*v26.py;
для повторения исследования нужны соответствующие исходные данные и Python-среда.

ЧТО ВЫБРАНО
Новая обученная модель включена: {decision['new_model_promoted']}.
Архитектура головы марки: {ck.get('head_kind','cosine')}.
Причина выбора и результаты: Development/EV26/release_decision.json.
Данные после проверки: {base_summary['rows']} фрагментов, {len(ck['brands'])} меток марки.
Дополнительные данные этой итерации: {base_summary['new_rows']} фрагментов;
происхождение/лицензии публичных фото сохранены в datasets/ev_views_v26/manifest.json.
Виды одной марки могут отличаться; ошибки на новых телефонных фото остаются возможны.
Не обещаются 100% ответов, все 325 моделей или гарантированная скорость на любом ПК.
Реальные контрольные ответы и замеры этой сборки — Development/EV26/RECIPIENT_VERIFICATION.json.

ДЛЯ СДАЧИ
CHECKLIST.md содержит открытые требования. Собственные телефонные данные пока
не основа корпуса; слепой экзамен 20 новых фото ещё не проведён. Публичный старый
test повторно использован в нескольких итерациях. Новые группы публичных фото
не заменяют независимую проверку на телефоне в Узбекистане.
Ранее включённые PDF являются черновиками. Итоговый PDF до 6 страниц нужно
обновить фактическими результатами и ссылкой GitHub; нужны реальные вклады,
история, тег midterm и раскрытие AI. Уникальность архитектуры среди команд и
понимание кода участниками проверяются отдельно. Полная сдача не подтверждена.
Готовый автомобильный классификатор не скачивался: используются общий CLIP
экстрактор изображения и собственная обученная голова, общий COCO-сегментатор,
собственные классификаторы цвета/моделей. Код и часть разметки подготовлены с AI;
проверка ассистентом не считается независимой проверкой человеком.
'''
for name in ('GUIDE_FULL_2_6_RU.txt','START_HERE_RU.txt','HANDOFF_GUIDE_RU.txt','TRANSFER.txt'):(target/name).write_text(guide,encoding='utf-8')
(target/'GUIDE_UPDATE_2_5_3_RU.txt').write_text('Это полная версия 2.6. Старый патч 2.5.3 устанавливать не нужно. Прочитайте GUIDE_FULL_2_6_RU.txt.',encoding='utf-8')
(ROOT/'outputs/CarVision-2.6-Full/READ_FIRST_RU.txt').write_text(guide,encoding='utf-8')
print('FULL VERSION STAGED',str(target),flush=True)
