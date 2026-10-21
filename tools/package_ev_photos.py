"""Real photographs + licence/provenance/split membership; no generated photos."""
import csv,hashlib,io,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT
src=ROOT/'datasets/ev_commons_v19';data=json.loads((ROOT/'models/ev_parts_candidate/dataset.json').read_text(encoding='utf-8'));review=json.loads((ROOT/'models/ev_annotation_review.json').read_text(encoding='utf-8'))
rows=json.loads((src/'manifest.json').read_text(encoding='utf-8'))+json.loads((src/'thumb_data/manifest.json').read_text(encoding='utf-8'))
accepted={r['source']['source']:r for r in data['new_rows']}
out=ROOT/'outputs/Real_Chinese_EV_Photos_v19';out.mkdir(exist_ok=True)
credits=[];csv_rows=[];seen=set()
for r in rows:
 path=src/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];target=out/'ev_commons_v19'/r['image'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
 a=accepted.get(r['source']);split=a['split'] if a else 'street_scene_unlabelled' if not r['brand'] else 'review_only_not_used';reason=review['excluded_sources'].get(r['source'],'Not included in this training cohort (deduplication, group sufficiency or crop quality).') if not a and r['brand'] else ''
 record=dict(r,training_split=split,training_group=a['group'] if a else None,reviewed_model=a['model'] if a else None,review_note=reason);credits.append(record);csv_rows.append([r['image'],r['brand'],a['model'] if a else r['model'],split,a['group'] if a else '',r['license'],r['author'],r['source'],reason])
 # One heldout example per NEW model family, selected by deterministic file
 # hash ordering BEFORE running the application; never cherry-picked by accuracy.
 if a and a['split']=='test' and (a['brand'],a['model']) in [tuple(p) for p in data['pairs'][24:]] and (a['brand'],a['model']) not in seen:
  seen.add((a['brand'],a['model']));folder=ROOT/'samples/Chinese_EV_Test';folder.mkdir(exist_ok=True);shutil.copy2(path,folder/Path(r['image']).name)
 if not r['brand']:
  folder=ROOT/'samples/Street_scenes';folder.mkdir(exist_ok=True);shutil.copy2(path,folder/path.name)
(out/'SOURCES_AND_SPLITS.json').write_text(json.dumps(credits,ensure_ascii=False,indent=2),encoding='utf-8')
with (out/'SOURCES_AND_SPLITS.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['image','brand','model_family','split','capture_group','license','author','source','review_note']);w.writerows(csv_rows)
for file in ['manifest.json','thumb_data/manifest.json']:
 target=out/'ev_commons_v19'/file;target.parent.mkdir(exist_ok=True);shutil.copy2(src/file,target)
shutil.copy2(ROOT/'models/ev_annotation_review.json',out/'ANNOTATION_REVIEW.json')
(out/'README.txt').write_text(f'''Реальные фотографии китайских автомобилей и уличных сцен — CarVision1.9
Файлов JPG: {len(rows)}. Изображения найдены, а не сгенерированы. Фото уменьшены до500/1024px; авторство не приписывается Софии Рахманкуловой.
Лицензия, автор, оригинальная страница и изменения указаны для КАЖДОГО снимка в SOURCES_AND_SPLITS.csv/json. Соблюдайте соответствующую CC BY/CC BY-SA/CC0/PD лицензию, включая атрибуцию и ShareAlike для фотографий, когда применимо.
Для проверки программы откройте любой JPG через «Открыть фото». Не выбирайте марку вручную. Встроенный список моделей ограничен инструкцией к выбранной сборке, а некоторые модели могут быть только в наборе для будущего расширения.
train: использован при обучении; validation: использован при выборе параметров; test: не использован для обучения/подбора. Для независимой проверки выбирайте test либо свои новые фотографии. Просмотр обучающих снимков не измеряет качество.
street_scene_unlabelled: настоящая дорожная сцена без достоверной разметки каждой машины, исключена из обучения марок/моделей. review_only_not_used: дубликаты, интерьер, недостаток групп либо неподходящий кадр; причины указаны в метаданных. Исходное название автора не всегда описывает главную машину: обнаруженный смешанный кадр с MG Cyberster исключён из обучения Tiggo8.
Подготовлено194новых пригодных снимка с групповыми train/validation/test; фильтрация и точные количества — models/ev_parts_candidate/dataset.json в исходном проекте. Собственные фото студента по заданию этим публичным набором не заменяются.
Yuan Plus, H6 и другие — семейства моделей; точное поколение/комплектация и тип силовой установки не гарантируются. Название модели не доказывает, что машина электрическая.
Для повторной подготовки данных сохранена структура ev_commons_v19/. Её можно поместить в datasets/ исходного проекта; рецептам также нужны прежние общедоступные данные/кэши, отсутствующие в этом фотоархиве. Самой переносимой программе для анализа обучение и интернет не нужны.
''',encoding='utf-8')
files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt');(out/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(out).as_posix()+'\n' for p in files),encoding='utf-8')
destination=ROOT/'CarVision-Real-Chinese-EV-Photos-v1.9.zip'
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for p in out.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(out))
with zipfile.ZipFile(destination) as z:
 assert z.testzip() is None
 for line in z.read('SHA256SUMS.txt').decode().splitlines():
  digest,path=line.split('  ',1);assert hashlib.sha256(z.read(path)).hexdigest()==digest
 assert len([p for p in z.namelist() if p.lower().endswith('.jpg')])==len(rows)
(ROOT/'samples/EV_PHOTO_SOURCES.json').write_text(json.dumps([r for r in credits if Path(r['image']).name in {p.name for folder in ['Chinese_EV_Test','Street_scenes'] for p in (ROOT/'samples'/folder).glob('*.jpg')}],ensure_ascii=False,indent=2),encoding='utf-8')
print('VERIFIED',destination,'photos',len(rows),'test examples',len(seen),'bytes',destination.stat().st_size,flush=True)
