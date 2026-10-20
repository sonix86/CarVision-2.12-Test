"""Load only isolated synthetic checkpoints and verify that inference cannot call the network."""
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import Pipeline,ROOT

with patch.object(socket.socket,'connect',side_effect=AssertionError('Inference network call forbidden')):
    pipeline=Pipeline(ROOT/'outputs/smoke-models',mode='detail')
    result=pipeline.predict(ROOT/'outputs/smoke-data/0.png')
assert result['classifier_trained'] and result['detector_views']==2
assert all(0<=r['box'][0]<r['box'][2]<=96 and 0<=r['box'][1]<r['box'][3]<=96 for r in result['cars'])
print(json.dumps(dict(engineering_check='PASS; isolated synthetic weights ONLY',views=result['detector_views'],network='blocked',real_world_accuracy='not measured')))
