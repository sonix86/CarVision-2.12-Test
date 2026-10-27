"""Stage measured fallback as explicitly experimental; never overwrite 2.5."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'tools/package_semantic_v25.py').read_text(encoding='utf-8')
source=source.replace("stage=ROOT/'outputs/CarVision-Update-2.5'","stage=ROOT/'outputs/CarVision-Update-2.5.3'")
source=source.replace('outputs/build-v25','outputs/build-v253').replace('frozen-v25-check.json','frozen-v253-check.json')
source=source.replace("assert evaluation['candidate_sha256']==sha(out/'semantic_make.pt')", "prior_evaluation=evaluation\nevaluation=load(ROOT/'models/fallback_v253/evaluation.json')\nassert evaluation['passed']\nfor cohort in ('all','historical'):\n    evaluation['results'][cohort]['whole']['baseline_body_top1']=prior_evaluation['results'][cohort]['whole']['baseline_body_top1']\nmodel_path=ROOT/'models/fallback_v253/semantic_make.pt'\nassert evaluation['candidate_sha256']==sha(model_path)")
source=source.replace("source=out/'semantic_make.pt'","source=model_path").replace("active['semantic_make.pt']=sha(out/'semantic_make.pt')","active['semantic_make.pt']=sha(model_path)")
# Human-facing version identifiers and installer names only; retain provenance
# directories named Semantic25 for the two original gradient training rounds.
source=source.replace("version='2.5'","version='2.5.3'").replace('CarVision 2.5 —','CarVision 2.5.3 —').replace('CarVision update 2.5\'','CarVision update 2.5.3\'').replace('GUIDE_UPDATE_2_5_RU','GUIDE_UPDATE_2_5_3_RU').replace('APPLY_UPDATE_2_5.cmd','APPLY_UPDATE_2_5_3.cmd')
source=source.replace("replace('2.3','2.5').replace('2_3','2_5').replace('update23','update25')","replace('2.3','2.5.3').replace('2_3','2_5_3').replace('update23','update253')").replace("name.replace('2_3','2_5')","name.replace('2_3','2_5_3')").replace('Update 2.5 installed','Update 2.5.3 installed').replace('patch-backups/update25-','patch-backups/update253-')
source=source.replace("guide=f'''CarVision", "guide=f'''ЭКСПЕРИМЕНТАЛЬНАЯ СБОРКА: реальные контрольные фото BYD и Toyota всё ещё не распознаны.\nЭто измеренное улучшение, а не готовое решение задачи безошибочного распознавания.\n\nCarVision")
source=source.replace('УСТАНОВКА\n1.', 'УСТАНОВКА\nДля сравнения сначала скопируйте всю старую папку CarVision в отдельную папку. В установщике выбирайте эту копию.\n1.')
source=source.replace('Полные результаты и слабые марки — Development/Semantic25/evaluation.json.','Результаты новой маршрутизации — Development/Fallback253/evaluation.json. Исходное обучение — Development/Semantic25.\nСохранены подтверждённые ответы основной модели; проверенное сравнение с учебными примерами дополняет только отказы. Дополнительного нейросетевого обучения в этой калибровке не было.')
source=source.replace("manifest=dict(release=", """for p in sorted((ROOT/'models/fallback_v253').glob('*.json')):add('Development/Fallback253/'+p.name,source=p)
for name in ('calibrate_fallback_v253.py','evaluate_fallback_v253.py','package_fallback_v253.py','check_frozen_v253.py'):
    add('tools/'+name,source=ROOT/'tools'/name)
for name in ('calibrate_views_v251.py','calibrate_retrieval_v252.py'):
    add('tools/'+name,source=ROOT/'tools'/name)
limitations='AI helped write/review code and label private crops. Private labels are not independent human ground truth. Public heldout was reused across iterations. Only 18 additional private training crops; own phone data is not yet primary. The BYD/Toyota owner controls remain failures. All 325 exact models and 100% accuracy are not achieved. No API, paid service or ready automotive classifier is used. Generic CLIP image backbone plus our trained head and labelled TRAIN references only.'
add('AI_USE_AND_LIMITATIONS_RU.txt',content=limitations)
add('CHECKLIST.md',source=ROOT/'CHECKLIST.md')
(stage/'QUALITY_STATUS_RU.txt').write_text('Общая первая гипотеза марки: 80.2% на 910 ранее отложенных фрагментах. Это не точность детектора сцены и не полевой экзамен. Контрольные BYD и Toyota пользователя: unknown. Задача полного распознавания не решена. Исходная установка 2.2 не изменена. Сравнивайте на копии полной программы.',encoding='utf-8')
old_stage=ROOT/'outputs/CarVision-Update-2.5/payload'
for entry in files:
    previous=old_stage/entry['path']
    if previous.is_file():entry['previous_sha256']=sorted(set(entry['previous_sha256']+[sha(previous)]))
manifest=dict(release=""")
source=source.replace("previous_exe_sha256=[hashlib.sha256(original.read('CarVision/CarVision.exe')).hexdigest()]", "previous_exe_sha256=[hashlib.sha256(original.read('CarVision/CarVision.exe')).hexdigest(),'7ece8ec4a4e9a3fbc5eef826e142bcf6b929ee3752682f510eb60a2b85df6dc8']")
exec(compile(source,str(ROOT/'tools/package_semantic_v25.py'),'exec'),dict(__file__=str(ROOT/'tools/package_semantic_v25.py'),__name__='__main__'))
