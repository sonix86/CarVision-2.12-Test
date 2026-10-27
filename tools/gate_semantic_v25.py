"""Review fixed supervised validation before any test scoring or promotion."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'models/semantic_v25'
report=json.loads((out/'training.json').read_text(encoding='utf-8'));p=report['policy']['validation_release']
checks=dict(confirmed_views=report['accepted_validation_views']>=p['confirmed_views_min'],confirmed_precision=(report['accepted_validation_precision'] or 0)>=p['confirmed_precision_min'],priority_makes=all(report['priority_enabled'].values()))
gate=dict(passed=all(checks.values()),checks=checks,priority_enabled=report['priority_enabled'],candidate_sha256=report['candidate_sha256'],test_used=False)
(out/'validation_gate.json').write_text(json.dumps(gate,indent=2),encoding='utf-8');print(json.dumps(gate),flush=True)
if not gate['passed']:
    (out/'REJECTED_DO_NOT_INSTALL.txt').write_text('The finite two-fit semantic iteration failed the fixed validation gate. Do not promote, lower thresholds, repeat training or score the test to search for a better epoch. See validation_gate.json and training.json. Existing released weights unchanged.',encoding='utf-8')
