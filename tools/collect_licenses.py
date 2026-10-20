import importlib.metadata
import sys
from pathlib import Path

out=Path(sys.argv[1])/'licenses';out.mkdir(parents=True,exist_ok=True)
for dist in importlib.metadata.distributions():
    name=dist.metadata.get('Name','package')
    for file in dist.files or []:
        if 'license' in file.name.lower() or file.name.lower().startswith('copying'):
            source=Path(dist.locate_file(file))
            if source.is_file():
                target=out/name/str(file).replace('../','').replace('..\\','')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
pylicense=Path(sys.base_prefix)/'LICENSE.txt'
if pylicense.exists():(out/'Python-LICENSE.txt').write_bytes(pylicense.read_bytes())
print('Collected installed third-party license notices.')
