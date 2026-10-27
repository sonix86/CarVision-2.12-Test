"""Plain-PyTorch DINOv2-S/14 general image features (no automotive head).

Architecture follows Meta's Apache-2.0 DINOv2 implementation. Generic weights
and source attribution are recorded separately; only our own trained decision
heads may turn these features into project predictions.
"""
import math
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision.transforms.functional import to_tensor, normalize
from .segmentation import isolate

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.qkv=torch.nn.Linear(384,1152); self.proj=torch.nn.Linear(384,384)
    def forward(self,x):
        b,n,c=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,6,64).permute(2,0,3,1,4).unbind(0)
        return self.proj(torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,c))


class LayerScale(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.gamma=torch.nn.Parameter(torch.ones(384))
    def forward(self,x): return x*self.gamma


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=torch.nn.Linear(384,1536); self.fc2=torch.nn.Linear(1536,384)
    def forward(self,x): return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class DinoBlock(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.norm1=torch.nn.LayerNorm(384,eps=1e-6); self.attn=Attention(); self.ls1=LayerScale()
        self.norm2=torch.nn.LayerNorm(384,eps=1e-6); self.mlp=Mlp(); self.ls2=LayerScale()
    def forward(self,x):
        x=x+self.ls1(self.attn(self.norm1(x)))
        return x+self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.proj=torch.nn.Conv2d(3,384,14,stride=14)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)


class DinoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.patch_embed=PatchEmbed(); self.cls_token=torch.nn.Parameter(torch.zeros(1,1,384))
        self.pos_embed=torch.nn.Parameter(torch.zeros(1,1370,384)); self.mask_token=torch.nn.Parameter(torch.zeros(1,384))
        self.blocks=torch.nn.Sequential(*[DinoBlock() for _ in range(12)]); self.norm=torch.nn.LayerNorm(384,eps=1e-6)
    def forward(self,image):
        _,_,h,w=image.shape; x=self.patch_embed(image)
        x=torch.cat([self.cls_token.expand(x.shape[0],-1,-1),x],1)
        patches=self.pos_embed[:,1:].reshape(1,37,37,384).permute(0,3,1,2)
        patches=torch.nn.functional.interpolate(patches,scale_factor=((h//14+.1)/37,(w//14+.1)/37),mode='bicubic',align_corners=False)
        pos=torch.cat([self.pos_embed[:,:1],patches.permute(0,2,3,1).reshape(1,-1,384)],1)
        return self.norm(self.blocks(x+pos))


def dino_input(image,mask,kind='whole'):
    image=image.convert('RGB').copy(); mask=np.asarray(mask)
    if kind=='phone':
        image.thumbnail((96,96));mask=np.asarray(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST))
        image=ImageEnhance.Brightness(image).enhance(.55).filter(ImageFilter.GaussianBlur(.35))
    image=isolate(image,mask)
    canvas=Image.new('RGB',(224,224),tuple(round(v*255) for v in MEAN)); image=ImageOps.contain(image,(224,224),Image.Resampling.BICUBIC)
    canvas.paste(image,((224-image.width)//2,(224-image.height)//2))
    return normalize(to_tensor(canvas),MEAN,STD)


def descriptor(tokens):
    cls=torch.nn.functional.normalize(tokens[:,0],dim=-1)
    patches=torch.nn.functional.normalize(tokens[:,1:].mean(1),dim=-1)
    return torch.nn.functional.normalize(torch.cat([cls*.7,patches*.3],dim=-1),dim=-1)
