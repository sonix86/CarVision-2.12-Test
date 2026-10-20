"""Public VCoR images and their provided annotations, not a pretrained colour model."""
import hashlib,json,time,urllib.request,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'datasets/vcor';root.mkdir(parents=True,exist_ok=True)
api='https://www.kaggle.com/api/v1/datasets/'
reference='landrykezebou/vcor-vehicle-color-recognition-dataset'
headers={'User-Agent':'TashkentCarVision/1.2 educational research'}
with urllib.request.urlopen(urllib.request.Request(api+'view/'+reference,headers=headers),timeout=60) as response:metadata=json.load(response)
(root/'source_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
file=root/'vcor.zip'
if not file.exists():
    partial=root/'vcor.partial';downloaded=0;reported=0
    with urllib.request.urlopen(urllib.request.Request(api+'download/'+reference,headers=headers),timeout=120) as response,partial.open('wb') as out:
        while chunk:=response.read(2*1024*1024):
            out.write(chunk);downloaded+=len(chunk)
            if downloaded-reported>=100*1024*1024:print('Downloaded MiB',round(downloaded/1024**2),flush=True);reported=downloaded
    if partial.read_bytes()[:4]!=b'PK\x03\x04':raise ValueError('Expected a public data ZIP, not an authentication page')
    partial.replace(file)
with zipfile.ZipFile(file) as archive:
    for item in archive.infolist():
        destination=(root/'images'/item.filename).resolve()
        if not destination.is_relative_to((root/'images').resolve()):raise ValueError('Archive path escapes dataset directory')
    archive.extractall(root/'images')
(root/'download_provenance.json').write_text(json.dumps(dict(source='https://www.kaggle.com/datasets/'+reference,archive_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),ready_made_model_used=False,purpose='Research/education; original public provided colour annotations. No student-collected photographs claimed.'),indent=2))
print('VCoR data ready',flush=True)
