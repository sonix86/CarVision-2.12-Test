"""Real-model offline inference and filename independence on separate city photographs."""
import hashlib,json,socket,sys,tempfile,shutil,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Pipeline,save_prediction
out=ROOT/'outputs/release-runtime-v19';out.mkdir(parents=True,exist_ok=True);results=[]
def hashes():return {name:hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest() for name in ['classifier.pt','detector.pt','colour_model.pt','vehicle_model.pt','segmenter.pt','masked_colour.pt','visual_references.pt']}
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
        cached=pipeline.predict(renamed)
        assert cached['cars']==repeat['cars'],'Cached masks must preserve the full prediction'
        repeated_seconds=cached['seconds']
    source=ROOT/'samples/images/01.jpg';region=max(results[0]['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]))['box']
    single=pipeline.predict(source,region=region,single=True)
    assert len(single['cars'])==1 and single.get('segmented') and single['cars'][0].get('contours')
    cropped_scene=pipeline.predict(source,region=region)
    assert all(region[0]<=r['box'][0]<=r['box'][2]<=region[2] and region[1]<=r['box'][1]<=r['box'][3]<=region[3] for r in cropped_scene['cars'])
    save_prediction(source,single,out)
    spark=results[-1]['cars'][0];prompt=pipeline.predict(ROOT/'samples/images/06.jpg',region=spark['box'],single=True)
    assert len(prompt['cars'])==1 and prompt['cars'][0]['colour']==spark['colour'],'Single-car region must preserve the full-context paint evidence'
    assert all(r.get('segmentation') and r.get('contours') for photo in results for r in photo['cars'])
(out/'checks.json').write_text(json.dumps(dict(network_connections_blocked=True,filename_independence=True,actual_single_car_roi_passed=True,actual_scene_roi_passed=True,synthetic_weights=False,model_sha256=model_hashes,photos=results,repeated_seconds=repeated_seconds,region_checks=dict(single=single,scene=cropped_scene)),indent=2))
print('Actual trained weights: offline and filename-independence checks passed.',flush=True)

