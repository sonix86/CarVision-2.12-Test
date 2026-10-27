"""Compare every delivered application module to its code inside the new EXE build."""
import json,types,marshal
from pathlib import Path
from PyInstaller.loader.pyimod01_archive import ZlibArchiveReader
from PyInstaller.archive.readers import CArchiveReader
ROOT=Path(__file__).resolve().parents[1]
reader=ZlibArchiveReader(str(ROOT/'outputs/build-v23/work/CarVision/PYZ-00.pyz'))
def signature(code):
    return (code.co_code,tuple(signature(v) if isinstance(v,types.CodeType) else v for v in code.co_consts),code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars)
checked=[]
for path in sorted((ROOT/'carvision').glob('*.py')):
    name='carvision' if path.stem=='__init__' else 'carvision.'+path.stem
    if name not in reader.toc:continue
    frozen=reader.extract(name);expected=compile(path.read_text(encoding='utf-8'),str(path),'exec')
    assert signature(frozen)==signature(expected),'Frozen source mismatch: '+name
    checked.append(name)
assert 'carvision.make_expert' in checked
entry=CArchiveReader(str(ROOT/'outputs/build-v23/dist/CarVision/CarVision.exe'))
assert signature(marshal.loads(entry.extract('launch')))==signature(compile((ROOT/'launch.py').read_text(encoding='utf-8'),str(ROOT/'launch.py'),'exec'))
report=dict(passed=True,modules=checked,entry_script_matches=True)
(ROOT/'outputs/frozen-v23-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
