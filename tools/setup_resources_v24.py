"""Copy licensed general-purpose resources to the isolated experiment only."""
import hashlib,importlib.metadata,json,shutil
from pathlib import Path
import rapidocr_onnxruntime
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'models/multimodal_v24';ocr=out/'ocr';ocr.mkdir(parents=True,exist_ok=True)
package=Path(rapidocr_onnxruntime.__file__).parent;credits=[]
for source in (package/'models').glob('*.onnx'):
    destination=ocr/source.name;shutil.copy2(source,destination)
    credits.append(dict(file=source.name,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),source_package='rapidocr_onnxruntime',package_version=importlib.metadata.version('rapidocr_onnxruntime'),upstream='PaddleOCR general text detector/recognizer',source='https://github.com/RapidAI/RapidOCR',license='Apache-2.0',car_classifier=False))
(ocr/'SOURCE_LICENSES.json').write_text(json.dumps(credits,indent=2),encoding='utf-8')
dist=importlib.metadata.distribution('rapidocr_onnxruntime')
for f in dist.files:
    if f.name.lower() in ('license','license.txt','license.md'):
        shutil.copy2(dist.locate_file(f),ocr/'RAPIDOCR_LICENSE.txt')
print('General OCR resources',len(credits),flush=True)
