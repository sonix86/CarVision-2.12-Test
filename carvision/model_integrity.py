"""Bind joint feature heads to the exact paired encoder checkpoint they trained with."""
import hashlib,re
from pathlib import Path

def verify_clip_pair(checkpoint,path):
    expected=checkpoint.get('clip_checkpoint_sha256')
    if checkpoint.get('kind')!='joint' or expected is None:return None
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Invalid paired CLIP checkpoint fingerprint')
    path=Path(path)
    if not path.is_file():raise ValueError('Отсутствует парная модель semantic_make.pt. Восстановите всю папку models.')
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    actual=digest.hexdigest()
    if actual!=expected:raise ValueError('Модели dino_make.pt и semantic_make.pt из разных наборов. Восстановите всю папку models из одной версии.')
    return actual
