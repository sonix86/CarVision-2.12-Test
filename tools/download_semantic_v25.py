"""One pinned public general-purpose CLIP download. No API key or inference API."""
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'cache/semantic_v25';out.mkdir(exist_ok=True)
meta=json.loads((out/'upstream_metadata.json').read_text(encoding='utf-8'));revision=meta['sha'];repository='laion/CLIP-ViT-B-32-laion2B-s34B-b79K';started=time.time();records=[]
for name in ('README.md','open_clip_config.json','open_clip_model.safetensors'):
    entry=next(r for r in meta['siblings'] if r['rfilename']==name);url=f'https://huggingface.co/{repository}/resolve/{revision}/{name}';target=out/name
    expected=entry.get('lfs',{}).get('sha256')
    if not target.exists():
        temporary=target.with_suffix(target.suffix+'.partial')
        with urllib.request.urlopen(url,timeout=45) as response,temporary.open('wb') as f:
            count=0
            while True:
                assert time.time()-started<600,'Finite download deadline exceeded'
                chunk=response.read(1024*1024)
                if not chunk:break
                f.write(chunk);count+=len(chunk)
                if count%(50*1024*1024)==0:print(name,count,'bytes',flush=True)
        assert temporary.stat().st_size==entry['size']
        if expected:assert hashlib.sha256(temporary.read_bytes()).hexdigest()==expected,'Upstream SHA mismatch'
        temporary.replace(target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    assert target.stat().st_size==entry['size'] and (not expected or digest==expected)
    records.append(dict(file=name,source=url,sha256=digest,bytes=target.stat().st_size,revision=revision))
report=dict(repository=repository,revision=revision,license='MIT (upstream model card)',generic_training='LAION-2B English general image/text corpus',ready_made_car_classifier=False,records=records)
(out/'SOURCE_LICENSES.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('GENERIC RESOURCE VERIFIED',flush=True)
