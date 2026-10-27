"""Verify generic image-encoder parity and generate initialization, never test scores."""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
import numpy as np,torch,open_clip
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import GeneralVisionEncoder,image_tensor
from PIL import Image
torch.set_num_threads(2);torch.manual_seed(250026)
source=ROOT/'cache/semantic_v25';out=ROOT/'models/semantic_v25';out.mkdir(exist_ok=True)
meta=json.loads((source/'SOURCE_LICENSES.json').read_text(encoding='utf-8'));expected=next(r['sha256'] for r in meta['records'] if r['file']=='open_clip_model.safetensors')
weights=source/'open_clip_model.safetensors';assert hashlib.sha256(weights.read_bytes()).hexdigest()==expected
reference=open_clip.create_model('ViT-B-32',pretrained=None).eval();reference.load_state_dict(load_file(str(weights)))
state={k.removeprefix('visual.'):v for k,v in reference.state_dict().items() if k.startswith('visual.')};encoder=GeneralVisionEncoder().eval();encoder.load_state_dict(state)
x=torch.stack([image_tensor(Image.fromarray(np.random.default_rng(25).integers(0,256,(123,241,3),dtype=np.uint8))),image_tensor(Image.new('RGB',(160,240),'white'))])
with torch.inference_mode():expected_vectors=reference.encode_image(x);actual=encoder(x)
error=float((expected_vectors-actual).abs().max());assert error<1e-4,('Generic encoder parity failure',error)
torch.save(dict(state=state,source_sha256=expected),source/'image_encoder.pt')
data=torch.load(ROOT/'cache/multimodal_v24_data.pt',weights_only=True);brands=data['brands'];tokenizer=open_clip.get_tokenizer('ViT-B-32')
prompts=[f'a photograph of a {b} car.' for b in brands]+[f'a vehicle manufactured by {b}.' for b in brands]+[f'the body and badge of a {b} automobile.' for b in brands]
with torch.inference_mode():z=reference.encode_text(tokenizer(prompts));z=torch.nn.functional.normalize(z,dim=1).reshape(3,len(brands),512).mean(0);z=torch.nn.functional.normalize(z,dim=1)
torch.save(dict(brands=brands,prototypes=z,prompts=prompts,source_sha256=expected),source/'initialization.pt')
report=dict(generic_backbone_only=True,automotive_classifier_downloaded=False,reference_max_absolute_error=error,dataset_sha256=hashlib.sha256((ROOT/'cache/multimodal_v24_data.pt').read_bytes()).hexdigest(),source=meta,initialization_only='Text embeddings initialize our supervised head; untrained zero-shot predictions never become the application answer.',dependencies={k:importlib.metadata.version(k) for k in ['open_clip_torch','timm','ftfy','regex','safetensors','huggingface_hub']})
(out/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('GENERAL ENCODER VERIFIED',error,flush=True)
