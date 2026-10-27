"""Compute frozen features for a preparation snapshot without fitting/using answers."""
import hashlib,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.make_scope_features_v23 import build
torch.set_num_threads(3)
source=ROOT/'models/ev_patch_v22/vehicle_model.pt';sha=hashlib.sha256(source.read_bytes()).hexdigest()
data=torch.load(ROOT/'cache/make_scope_v23.pt',weights_only=True)
build(data['rows'],torch.load(source,weights_only=True),sha,ROOT/'cache/make_scope_v23_features.pt')
print('COMPLETE frozen feature cache; no make head fitted',flush=True)
