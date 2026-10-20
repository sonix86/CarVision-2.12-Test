"""Fetch only public images/labels, never pretrained classifier weights."""
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'datasets/stanford_cars';root.mkdir(parents=True,exist_ok=True)
base='https://huggingface.co/datasets/tanganke/stanford_cars'
def read(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TashkentCarVision-research-prototype/1.2'}),timeout=60) as r:return r.read()
card=read(base+'/raw/main/README.md');(root/'README.source.md').write_bytes(card)
annotation_urls={
    'original_github_tree.json':'https://api.github.com/repos/jhpohovey/StanfordCars-Dataset/git/trees/c7421951b5a2292c8951ef69394231be7fc32474?recursive=1',
    'cars_train_annos.mat':'https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/c7421951b5a2292c8951ef69394231be7fc32474/stanford_cars/devkit/cars_train_annos.mat',
    'cars_test_annos_withlabels.mat':'https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/c7421951b5a2292c8951ef69394231be7fc32474/stanford_cars/cars_test_annos_withlabels.mat',
    'cars_annos.mat':'https://raw.githubusercontent.com/alitourani/deep-learning-vehicle-detection/master/cars_annos.mat'}
for name,url in annotation_urls.items():
    if not (root/name).exists():(root/name).write_bytes(read(url))
(root/'annotation_sources.json').write_text(json.dumps(annotation_urls,indent=2),encoding='utf-8')
entries=json.loads(read('https://huggingface.co/api/datasets/tanganke/stanford_cars/tree/main/data?expand=false'))
files=[r for r in entries if '/train-' in r['path'] or '/test-' in r['path']]
(root/'download_manifest.json').write_text(json.dumps(dict(source=base,files=files,purpose='Public-data prototype, NOT own Tashkent collection. Original source: Stanford Cars, Krause et al. 2013. Not a pretrained model.'),indent=2),encoding='utf-8')
def fetch(row):
    destination=root/Path(row['path']).name
    if destination.exists() and destination.stat().st_size==row['size']:print('Cached',destination.name,flush=True);return
    for attempt in range(4):
        try:
            request=urllib.request.Request(base+'/resolve/main/'+row['path']+'?download=true&cache='+str(int(time.time())),headers={'User-Agent':'TashkentCarVision-research-prototype/1.2'})
            with urllib.request.urlopen(request,timeout=90) as response,destination.with_suffix('.partial').open('wb') as output:
                while chunk:=response.read(1024*1024):output.write(chunk)
            if destination.with_suffix('.partial').stat().st_size!=row['size']:raise IOError('Incomplete download')
            destination.with_suffix('.partial').replace(destination);print('Downloaded',destination.name,row['size'],flush=True);return
        except Exception as error:
            print('Retry',destination.name,attempt+1,str(error)[:180],flush=True)
            if attempt==3:raise
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(fetch,files))
print('Public prototype dataset download complete',flush=True)
