"""Restore the exact frozen release using only Python's standard library."""
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class SplitReader:
 def __init__(self,paths):
  self.files=[p.open('rb') for p in paths];self.sizes=[p.stat().st_size for p in paths];self.pos=0;self.length=sum(self.sizes)
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
  if pos<0:raise ValueError('negative seek')
  self.pos=pos;return pos
 def read(self,n=-1):
  remain=max(0,self.length-self.pos) if n<0 else min(n,max(0,self.length-self.pos));out=[];start=0
  for f,size in zip(self.files,self.sizes):
   if remain and self.pos<start+size:
    local=self.pos-start;f.seek(local);b=f.read(min(remain,size-local));out.append(b);self.pos+=len(b);remain-=len(b)
   start+=size
  return b''.join(out)
 def close(self):
  for f in self.files:f.close()
def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe_target(root,name):
 p=(root/name).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path: '+name)
 return p
def restore_member(z,info,name):
 target=safe_target(ROOT,name)
 if info.is_dir():return
 if target.exists():
  h=hashlib.sha256()
  with z.open(info) as source:
   for b in iter(lambda:source.read(8*1024*1024),b''):h.update(b)
  if sha_file(target)!=h.hexdigest():raise RuntimeError('Refusing to overwrite a different existing file: '+str(target))
  return
 target.parent.mkdir(parents=True,exist_ok=True)
 with z.open(info) as source,target.open('xb') as dest:
  for b in iter(lambda:source.read(8*1024*1024),b''):dest.write(b)
def main():
 p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,default=ROOT/'.release-downloads');p.add_argument('--download',action='store_true');p.add_argument('--training',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 manifest=json.loads((ROOT/'release_assets.json').read_text('utf-8'));extra=([manifest['training']]+([manifest['local_photos']] if manifest.get('local_photos') else [])) if a.training else [];items=manifest['parts']+extra;a.downloads.mkdir(parents=True,exist_ok=True)
 for item in items:
  dest=a.downloads/item['name']
  if not dest.exists() and a.download:
   url=manifest['repository']+'/releases/download/'+manifest['release_tag']+'/'+item['name'];temp=dest.with_suffix(dest.suffix+'.download')
   print('DOWNLOAD',item['name'],flush=True)
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CarVisionRestore/2.12'}),timeout=120) as source,temp.open('wb') as out:
    for b in iter(lambda:source.read(8*1024*1024),b''):out.write(b)
   temp.replace(dest)
  if not dest.exists():raise FileNotFoundError('Download from '+manifest['repository']+'/releases/tag/'+manifest['release_tag']+': '+item['name'])
  if dest.stat().st_size!=item['bytes'] or sha_file(dest)!=item['sha256']:raise ValueError('Release file failed SHA256/size verification: '+item['name'])
  print('VERIFIED',item['name'],flush=True)
 stream=SplitReader([a.downloads/r['name'] for r in manifest['parts']]);h=hashlib.sha256()
 for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 if stream.length!=manifest['original_zip']['bytes'] or h.hexdigest()!=manifest['original_zip']['sha256']:raise ValueError('Reconstructed original ZIP mismatch')
 stream.seek(0)
 with zipfile.ZipFile(stream) as z:
  for info in z.infolist():
   parts=Path(info.filename).parts
   if len(parts)>2 and parts[0]=='CarVision' and parts[1] in ('models','data','datasets','samples'):
    name=Path(*parts[1:]).as_posix();safe_target(ROOT,name)
    if not a.verify_only:restore_member(z,info,name)
 stream.close()
 if a.training:
  for item in extra:
   with zipfile.ZipFile(a.downloads/item['name']) as z:
    for info in z.infolist():
     safe_target(ROOT,info.filename)
     if not a.verify_only:restore_member(z,info,info.filename)
 print('PASS - frozen release verified'+(' and restored' if not a.verify_only else ' (no extraction)'),flush=True)
if __name__=='__main__':main()
