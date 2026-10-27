"""One bounded Commons category inventory; it does not create model labels."""
import hashlib,json,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/ev_catalog_inventory';OUT.mkdir(exist_ok=True)
catalog=json.loads((ROOT/'data/catalogue/ev_target_catalog.json').read_text(encoding='utf-8'))['entries']
def categories(r):
    b,m=r['brand'],r['family'];names=[b+' '+m]
    if b=='GAC Aion':names.append('Aion '+m)
    if b=='Li Auto':names.append('Li '+m)
    if b=='BYD' and m=='Chazor':names.append('BYD Destroyer 05')
    if b=='BYD' and m=='Yuan Plus':names.append('BYD Atto 3')
    if b=='Geely' and m=='Galaxy E5':names.append('Geely EX5')
    return ['Category:'+n for n in names]
titles=sorted({n for r in catalog for n in categories(r)});found={};last=0
for first in range(0,len(titles),40):
    params=dict(action='query',format='json',prop='categoryinfo',titles='|'.join(titles[first:first+40]))
    url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)
    cache=OUT/('api_'+hashlib.sha256(url.encode()).hexdigest()[:20]+'.json')
    if not cache.exists():
        time.sleep(max(0,10-(time.monotonic()-last)));last=time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TashkentCarVision/2.2 educational source inventory'}),timeout=35) as response:raw=response.read()
        except urllib.error.HTTPError as error:
            if error.code==429:
                (OUT/'deferred.json').write_text(json.dumps(dict(reason='HTTP429; stopped without retry',retry_after=error.headers.get('Retry-After'))))
            raise
        cache.write_bytes(raw)
    for page in json.loads(cache.read_text(encoding='utf-8')).get('query',{}).get('pages',{}).values():
        if 'categoryinfo' in page:found[page['title']]=page['categoryinfo']
    print('BATCH',first,len(titles),flush=True)
rows=[]
for r in catalog:
    matches=[dict(category=n,**found[n],url='https://commons.wikimedia.org/wiki/'+urllib.parse.quote(n.replace(' ','_'))) for n in categories(r) if n in found]
    rows.append(dict(brand=r['brand'],family=r['family'],category_candidates=matches,
                     status='category_found_requires_per_image_licence_and_label_review' if matches else 'category_not_found_under_queried_names_not_proof_of_absence'))
report=dict(entries=rows,queried_categories=len(titles),families_with_category=sum(bool(r['category_candidates']) for r in rows),
            downloaded_images=0,trained_models=0,
            limitations='Inventory only. Category counts include interiors, duplicates and mislabels. No licence, session independence or recognition guarantee can be derived from counts.')
(OUT/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',report['families_with_category'],len(rows),flush=True)
