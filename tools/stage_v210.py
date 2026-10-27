"""Stage a separate app; replace hardlinked files atomically, never edit base."""
import hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def replace(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.v210-tmp');shutil.copy2(src,tmp);os.replace(tmp,dest)
def write(dest,text):
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.v210-tmp');tmp.write_text(text,encoding='utf-8' if dest.suffix=='.json' else 'utf-8-sig');os.replace(tmp,dest)
def main():
    base=ROOT/'outputs/CarVision-2.9-Test/CarVision';dest=ROOT/'outputs/CarVision-2.10-Test/CarVision';build=ROOT/'outputs/build-v210/dist/CarVision';out=ROOT/'models/truth_v31';assert json.loads((out/'scoped_regression.json').read_text())['passed'];assert (build/'CarVision.exe').is_file() and not dest.exists();baseline={name:sha(base/name) for name in ('CarVision.exe','models/dino_make.pt','models/model_version.json','TRANSFER_CONTENTS.json')};old=json.loads((base/'TRANSFER_CONTENTS.json').read_text())['files']
    def copy_file(src,dst):
        if Path(src).suffix.lower() in ('.json','.jsonl','.py','.txt','.md','.ps1','.cmd','.csv','.toml'):return shutil.copy2(src,dst)
        os.link(src,dst);return dst
    shutil.copytree(base,dest,copy_function=copy_file,ignore=shutil.ignore_patterns('outputs','patch-backups','__pycache__'))
    for p in build.rglob('*'):
        if p.is_file():
            rel=p.relative_to(build).as_posix()
            if not (dest/rel).exists() or old.get(rel)!=sha(p):replace(p,dest/rel)
    replace(out/'scoped_candidate.pt',dest/'models/dino_make.pt');modelsha=sha(dest/'models/dino_make.pt')
    version=dict(version='2.10 TEST',owner='София Рахманкулова',verify_active_head=True,selected_model='dino_make.pt',selected_model_sha256=modelsha,make_head='2.9 DINO+CLIP retained; family fallback only Dongfeng on abstention',family_head='Own43-family shared-feature cosine head; own1/4/5nowTRAIN',colour_head='2.9 unchanged',make_labels=82,model_families=43,all325_models_trained=False,own_data_primary=False,quality_goal_achieved=False,development_regression_not_blind=True)
    write(dest/'models/model_version.json',json.dumps(version,ensure_ascii=False,indent=2))
    active=json.loads((base/'Development/ACTIVE_MODEL_SHA256.json').read_text());active['dino_make.pt']=modelsha;write(dest/'Development/ACTIVE_MODEL_SHA256.json',json.dumps(active,indent=2))
    for folder in ('carvision','tools','tests'):
        for p in (ROOT/folder).glob('*.py'):replace(p,dest/folder/p.name)
    replace(ROOT/'launch.py',dest/'launch.py');dev=dest/'Development/FIX_2_10';dev.mkdir(exist_ok=True)
    for research in ('truth_v30','truth_v31'):
        for p in (ROOT/'models'/research).glob('*.json'):replace(p,dev/research/p.name)
    for p in (ROOT/'datasets/e70_truth_v30').glob('*'):
        if p.is_file():replace(p,dest/'datasets/e70_truth_v30'/p.name)
    write(dest/'datasets/e70_truth_v30/NOTICE.txt','Each photo retains its author, date, source and licence in manifest.json. Source Commons thumbnails resized; instance crops/masks and contact sheets are research derivatives. Licence terms of each source image apply to its derivatives. No synthetic car photos or classifier-generated labels.\n')
    replace(ROOT/'outputs/tests-210-final.log',dev/'tests.log')
    checker=(ROOT/'handoff/CHECK_ON_NEW_PC.ps1').read_text(encoding='utf-8').replace("version='2.1'","version='2.10 TEST'");write(dest/'CHECK_ON_NEW_PC.ps1',checker)
    guide='''CarVision 2.10 TEST — исправление проверки марки и модели
София Рахманкулова

ЗАПУСК: откройте папку CarVision и запустите CarVision.exe.
В ЗАГОЛОВКЕ ОКНА должно быть «CarVision 2.10 TEST», слева тоже2.10 TEST.
Открыть фото -> Анализировать фото. CMD/EXAM/интернет/ключ/API/Python для
обычного анализа не нужны. Для переноса нужна вся папка либо полный архив.
Старые2.9/2.6оставьте отдельно. Подменять только модель или толькоEXE нельзя.
После переноса можно запустить CHECK_ON_NEW_PC.cmd; PASS означает
техническую работу, а не правильность всех распознаваний.

ПОЧЕМУ2.9ОШИБАЛАСЬ
1. Марка была обновлена, а конкретная модель определялась прежней головой.
   Поэтому BYD распознавался, а Song Plus получал3%нижеHan9%.
2. Касание рамкой края фото безусловно запрещало подтверждать модель.
   Теперь исключение разрешено только для принятой новой головы, проверенной
   на обрезанных кадрах; малые/узкие/плохо видимые фрагменты остаются ограничены.
3. Первый белый седан ранее ошибочно назвалиGAC Aion S Plus. По эмблеме,
   кузову и независимым фотографиям это Dongfeng/Aeolus E70. В2.9он был
   исключён из обучения и не входил в цифру35/39. Та цифра не оценивала модели.

ЧТО ИЗМЕНЕНО
Своя новая голова43семейств на общих DINO+CLIP признаках; готовый чужой
классификатор марок/моделей не используется. Для неизвестной марки новый
путь подтверждения включён только дляDongfeng; прочие решения марки сохранены.
Новая голова моделей используется для поддержанных семейств всех марок.
База325названий не означает обучение325моделей. Цвет и детектор не заменены.
JSON результатов содержит runtime_identity: сборку, версию весов, ихSHA256
и число семейств. Программа проверяет, что веса соответствуют этой сборке.

ПРОВЕРКИ И ГРАНИЦЫ
БелыйE70(11223344), Dolphin(photo4) и Song Plus(photo5) использованы как
проверенные обучающие исправления. Их правильный ответ не независимая точность.
photo2—другой ракурс того жеE70—не использован в этом дообучении, но это
известная контрольная сцена, а не слепой экзамен на других машинах.
Повторная проверка разработки:942публичных изображения марки и461модели.
Марка:619->623верных подтверждений; конкретная модель:73->299.
У новых подтверждённых моделей осталось4ошибки на обычных изображениях.
Эта проверка повторно использовала прежний публичный тест после исследования;
она не является новым независимым экзаменом и не гарантирует100%на улице.
Полный вариант подстановки всех марок был отклонён; подробности FIX_2_10.
Цвет на собственных фото пока слабый; ошибки/unknown/лишние контуры остаются.
Собственные данные ещё не основа корпуса; свежий слепой тест20телефонных фото,
актуальный отчёт с вкладом участников и GitHub ещё нужны для полной сдачи.

ОБУЧЕНИЕ
Загрузка фото для анализа не обучает модель. Размечайте правильные марку,
модель(если уверены), цвет и группы съёмки. Близкие кадры одной машины нельзя
разделять между обучением и независимой проверкой. Кнопка учебного обучения
создаёт отдельного кандидата и не заменяет автоматически текущую расширенную
голову. Рецепты и протоколы исправления сохранены в tools и Development/FIX_2_10.
'''
    for name in ('GUIDE_2_10_RU.txt','GUIDE_FULL_2_6_RU.txt','QUALITY_STATUS_RU.txt','CHECKLIST_NEW_PC.txt'):write(dest/name,guide)
    write(dest.parent/'READ_FIRST_RU.txt',guide)
    for name,value in baseline.items():assert sha(base/name)==value,'Base2.9 unexpectedly changed: '+name
    stage=dict(target=str(dest),exe_sha256=sha(dest/'CarVision.exe'),model_sha256=modelsha,base29_preserved=baseline,not_final_quality=True);write(ROOT/'outputs/stage-v210.json',json.dumps(stage,indent=2));print('STAGED',json.dumps(stage),flush=True)
if __name__=='__main__':main()
