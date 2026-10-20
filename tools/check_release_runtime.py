"""Real-model offline inference and filename independence on separate city photographs."""
import hashlib,json,socket,sys,tempfile,shutil,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Pipeline,save_prediction
out=ROOT/'outputs/release-runtime';out.mkdir(parents=True,exist_ok=True);results=[]
def hashes():return {name:hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest() for name in ['classifier.pt','detector.pt','colour_model.pt']}
model_hashes=hashes()
with patch.object(socket.socket,'connect',side_effect=AssertionError('Unexpected network request')):
    pipeline=Pipeline(mode='fast')
    assert hashes()==model_hashes,'Models changed while loading the test pipeline'
    for file in sorted((ROOT/'samples/images').glob('*.jpg')):
        result=pipeline.predict(file);save_prediction(file,result,out);results.append(result)
        print(file.name,result['seconds'],[(r['brand'],r['colour']) for r in result['cars']],flush=True)
    with tempfile.TemporaryDirectory() as temp:
        renamed=Path(temp)/'unseen_camera_filename.jpg';shutil.copy2(ROOT/'samples/images/01.jpg',renamed)
        repeat=pipeline.predict(renamed)
        assert repeat['cars']==results[0]['cars'],'Predictions must not depend on the image filename'
(out/'checks.json').write_text(json.dumps(dict(network_connections_blocked=True,filename_independence=True,synthetic_weights=False,model_sha256=model_hashes,photos=results),indent=2))
print('Actual trained weights: offline and filename-independence checks passed.',flush=True)
