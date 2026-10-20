"""Download reusable public photo examples and retain source/author/license metadata.

These are visual test fixtures, NOT student-collected training data or verified labels.
"""
import html
import json
import re
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    ('01','Chevrolet cars, Toshkent.jpg','Несколько машин: перекрытия и разные расстояния'),
    ('02','BYD Seagull, UZ.jpg','Китайская марка: вечерний снимок'),
    ('03','Chevrolet Cobalt. Chilonzor-5 avenue, Tashkent, Uzbekistan.jpg','Одиночный автомобиль: проверьте рамку и цвет'),
    ('04','Chevrolet Malibu in Beshchinar Mahallah, Tashkent.jpg','Уличный кадр: ракурс и фон'),
    ('05','Nexia-3 vehicle (Tashkent, Uzbekistan).jpg','Проверка марки по эмблеме: название модели не заменяет марку'),
    ('06','Ташкент, ВАЗ-2101 на Нуронийлар.jpg','Другой тип кузова и возраст автомобиля'),
]
HEADERS={'User-Agent':'TashkentCarVision/1.1 (educational photo examples; no automated training)'}
def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=60) as response:return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429,503) or attempt==3:raise
            time.sleep(3*(attempt+1))
def clean(s):return html.unescape(re.sub('<[^>]*>','',s)).strip()

out=ROOT/'samples';(out/'images').mkdir(parents=True,exist_ok=True);records=[]
for ident,title,purpose in FILES:
    query=urllib.parse.urlencode(dict(action='query',format='json',prop='imageinfo',iiprop='url|extmetadata',titles='File:'+title))
    page=next(iter(json.loads(get('https://commons.wikimedia.org/w/api.php?'+query))['query']['pages'].values()))
    info=page['imageinfo'][0];meta=info['extmetadata']
    license=clean(meta.get('LicenseShortName',{}).get('value',''))
    if not license.startswith('CC BY'):raise ValueError('Unverified reuse license: '+title+' '+license)
    path=out/'images'/(ident+'.jpg')
    if not path.exists():path.write_bytes(get(info['url'].split('?')[0]))
    records.append(dict(id=ident,title=title,purpose=purpose,image='images/'+path.name,source=info['descriptionurl'],author=clean(meta['Artist']['value']),license=license,license_url=meta.get('LicenseUrl',{}).get('value',''),date=clean(meta.get('DateTimeOriginal',{}).get('value','')),description=clean(meta.get('ImageDescription',{}).get('value','')),changes='No edits to downloaded photo; scaled proportionally in PDF.'))
    print(ident,title,license,flush=True)
    time.sleep(1)
(out/'sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'README.txt').write_text('Общедоступные фотографии Ташкента для ручной проверки приложения.\nНе являются собственными фотографиями Софии Рахматулаевой, обучающим набором или проверенной разметкой.\nАвторы, ссылки и лицензии: sources.json. Не загружать эти примеры в train/validation при измерении на них качества.\nМарка определяется по видимой эмблеме, а не по имени файла.\n',encoding='utf-8')
