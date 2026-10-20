"""Verify frozen application logic corresponds to the source files delivered alongside it."""
import json,types
from pathlib import Path
from PyInstaller.loader.pyimod01_archive import ZlibArchiveReader
root=Path(__file__).resolve().parents[1];reader=ZlibArchiveReader(str(root/'build/CarVision/PYZ-00.pyz'))
def signature(code):
    return (code.co_code,tuple(signature(v) if isinstance(v,types.CodeType) else v for v in code.co_consts),code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars)
checked=[]
for module in ['core','gui','training','detector_training','reporting','colour_model','overlays','vehicle_model','segmentation','scene','masked_colour','visual_parts']:
    source=root/'carvision'/f'{module}.py';frozen=reader.extract('carvision.'+module);expected=compile(source.read_text(encoding='utf-8'),str(source),'exec')
    assert signature(frozen)==signature(expected),'Frozen code differs from source: '+module
    checked.append(module)
(root/'outputs/frozen-v17-check.json').write_text(json.dumps(dict(passed=True,modules=checked),indent=2))
print('Frozen code matches current sources:',', '.join(checked))
