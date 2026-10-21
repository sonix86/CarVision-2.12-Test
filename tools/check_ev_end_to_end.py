"""Final heldout whole-photo diagnostic AFTER selection; never tunes parameters."""
import json,socket,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Pipeline
dataset=json.loads((ROOT/'models/ev_dataset.json').read_text(encoding='utf-8'));selected=json.loads((ROOT/'models/release_v19_comparison.json').read_text(encoding='utf-8'));out=ROOT/'outputs/ev-end-to-end-v19';out.mkdir(exist_ok=True);results=[]
with patch.object(socket.socket,'connect',side_effect=AssertionError('Inference attempted network access')):
 pipeline=Pipeline(mode='fast')
 for i,row in enumerate([r for r in dataset['new_rows'] if r['split']=='test'],1):
  path=ROOT/'datasets/ev_commons_v19'/row['source']['image'];prediction=pipeline.predict(path);main=max(prediction['cars'],key=lambda r:r['mask_pixels']) if prediction['cars'] else None
  pair=(row['brand'],row['model']);top1=bool(main and (main.get('model_candidate_brand'),main.get('model_candidate'))==pair);top3=bool(main and any((r['brand'],r['model'])==pair for r in main.get('model_alternatives',[])))
  results.append(dict(source=row['source']['source'],ground_truth=pair,largest_visible_instance=main,detected_instances=len(prediction['cars']),top1_candidate_correct=top1,top3_candidate_correct=top3,seconds=prediction['seconds']))
  print(i,'/',27,'top1',sum(r['top1_candidate_correct'] for r in results),'top3',sum(r['top3_candidate_correct'] for r in results),'seconds',prediction['seconds'],flush=True)
confirmed=[r for r in results if r['largest_visible_instance'] and r['largest_visible_instance'].get('model','unknown')!='unknown']
summary=dict(network_blocked=True,model_selection_completed_before_this_check=True,model_sha256=selected['model_sha256'],photos=len(results),top1_count=sum(r['top1_candidate_correct'] for r in results),top3_count=sum(r['top3_candidate_correct'] for r in results),no_detection_count=sum(r['detected_instances']==0 for r in results),confirmed_model_count=len(confirmed),confirmed_correct_count=sum((r['largest_visible_instance']['brand'],r['largest_visible_instance']['model'])==tuple(r['ground_truth']) for r in confirmed),seconds_max=max(r['seconds'] for r in results),scope='27new heldout full photographs; largest visible detected instance compared to dominant-car source label. Candidate correctness, not calibrated confirmed-name accuracy. No all-instance mask ground truth, no scene AP claim. No model/settings changes based on this result.',results=results)
(out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');(ROOT/'models/ev_end_to_end.json').write_text(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),encoding='utf-8');print('FINAL',json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
