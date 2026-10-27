"""Assemble a separate test build, preserving all sealed prior versions."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=ROOT/'outputs/CarVision-2.9-Test/CarVision';build=ROOT/'outputs/build-v29/dist/CarVision';research=ROOT/'models/owner_domain_v29'
    assert json.loads((research/'evaluation.json').read_text())['passed']
    assert (target/'models/dino_make.pt').exists() and (build/'CarVision.exe').exists()
    for p in build.rglob('*'):
        if p.is_file():
            dest=target/p.relative_to(build);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for folder in ('carvision','tests','tools'):
        for p in (ROOT/folder).glob('*.py'):
            dest=target/folder/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(ROOT/'launch.py',target/'launch.py')
    dev=target/'Development/EV29';dev.mkdir(exist_ok=True)
    for p in research.glob('*.json'):shutil.copy2(p,dev/p.name)
    (target/'licenses').mkdir(exist_ok=True);shutil.copy2(ROOT/'licenses/DINOv2-Apache-2.0.txt',target/'licenses/DINOv2-Apache-2.0.txt')
    active={p.name:sha(p) for p in (target/'models').glob('*.pt')}
    (target/'Development/ACTIVE_MODEL_SHA256.json').write_text(json.dumps(active,indent=2),encoding='utf-8')
    checker=(ROOT/'handoff/CHECK_ON_NEW_PC.ps1').read_text(encoding='utf-8').replace("version='2.1'","version='2.9 TEST'")
    (target/'CHECK_ON_NEW_PC.ps1').write_text(checker,encoding='utf-8-sig')
    guide='''CarVision 2.9 TEST — отдельная проверочная сборка
София Рахманкулова

ЗАПУСК: открыть папку CarVision и запустить CarVision.exe.
Слева должна быть версия 2.9 TEST. Открыть фото -> Анализировать фото.
Командная строка, EXAM.cmd, старые патчи, Python и интернет для анализа не нужны.
Держите всю папку вместе: EXE, _internal, models и остальные файлы.
Не заменяйте один файл EXE и не накладывайте эту папку на старую 2.6.
Проверка переноса: CHECK_ON_NEW_PC.cmd. PASS означает техническую
работоспособность, а не правильность всех ответов.

ИЗМЕНЕНИЯ: собственная обученная голова марки над общими DINOv2+CLIP.
Исправлены подтверждённые ошибки учебных подписей и фрагментов.
Новая голова цвета не прошла проверку; сохранён цвет из 2.6.
Детектор и голова точной модели остаются прежними. Возможны лишние
машины в салоне, ошибки марки/цвета и отсутствие конкретной модели.
Проценты — оценки сети; верхняя гипотеза не равна подтверждённому ответу.

ПРОВЕРКА КАЧЕСТВА: одинаковые937 публичных изображений, не новый экзамен.
Марка top1: 81.54% -> 83.14%; искусственно ухудшенные: 71.29% ->76.95%.
Верные подтверждённые ответы:576 ->619; ухудшенные:446 ->478.
Точность среди подтверждённых:96.42% и95.22%; ошибки остаются.
Средняя по всем маркам метрика на обычных снимках снизилась68.55% ->67.35%.
Это исследовательский компромисс, а не улучшение каждого класса.
39 проверенных собственных фрагментов этой итерации использованы для
обучения. Их успешное повторное распознавание не независимая точность.
Первая пара ранее названных GAC снимков спорная и не размечена заново.
Dongfeng пока не имеет разрешённого порога подтверждения: возможен unknown
даже при правильной верхней гипотезе. Новые20 слепых телефонных фото нужны.

ОГРАНИЧЕНИЯ:82 метки марки не означают все325 моделей. Точных семейств42.
Гарантии100% нет. Порог не понижен ради выдачи уверенных неверных ответов.
Условия сдачи ещё требуют собственные данные как основу, слепую проверку,
актуальный отчёт с вкладом участников и настоящей ссылкой GitHub.
История прежних экспериментов сохранена в Development; текущая ветка EV29.
'''
    for name in ('GUIDE_FULL_2_6_RU.txt','GUIDE_2_9_RU.txt','QUALITY_STATUS_RU.txt','CHECKLIST_NEW_PC.txt'):(target/name).write_text(guide,encoding='utf-8-sig')
    (target.parent/'READ_FIRST_RU.txt').write_text(guide,encoding='utf-8-sig')
    # Previous release manifests would incorrectly describe this new test build.
    files={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file() and not any(s in ('outputs','__pycache__') for s in p.relative_to(target).parts) and p.name not in ('TRANSFER_CONTENTS.json','SHA256SUMS.txt')}
    (target/'TRANSFER_CONTENTS.json').write_text(json.dumps(dict(version='2.9 TEST',files=files),indent=2),encoding='utf-8')
    (target/'SHA256SUMS.txt').write_text('\n'.join(v+'  '+k for k,v in files.items())+'\n',encoding='utf-8')
    print('ASSEMBLED',target,'EXE',sha(target/'CarVision.exe'),flush=True)
if __name__=='__main__':main()
