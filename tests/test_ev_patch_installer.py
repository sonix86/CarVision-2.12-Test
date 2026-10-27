"""The recipient installer must preserve user files and reject corrupt/stale inputs."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
import unittest,tempfile

ROOT=Path(__file__).resolve().parents[1]
def check_recipient_patch(tmp_path,failure):
    patch=tmp_path/'patch';target=tmp_path/'CarVision';patch.mkdir();target.mkdir()
    (target/'models').mkdir();(patch/'payload/models').mkdir(parents=True)
    exe=target/'CarVision.exe';exe.write_bytes(b'engineering fixture, not executable')
    old=target/'models/vehicle_model.pt';old.write_bytes(b'old model')
    user=target/'user-photo.txt';user.write_text('preserve me')
    payload=patch/'payload/models/vehicle_model.pt';payload.write_bytes(b'new model')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(exe_sha256=sha(exe),files=[dict(path='models/vehicle_model.pt',sha256=sha(payload),previous_sha256=sha(old))])
    (patch/'PATCH_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
    shutil.copy2(ROOT/'handoff/APPLY_EV_PATCH.ps1',patch)
    if failure=='payload':payload.write_bytes(b'corrupted')
    if failure=='baseline':old.write_bytes(b'owner-trained model')
    original=old.read_bytes()
    shell=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    run=subprocess.run([str(shell),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(patch/'APPLY_EV_PATCH.ps1'),'-Target',str(target)],capture_output=True,timeout=60)
    assert user.read_text()=='preserve me'
    if failure:
        assert run.returncode!=0
        assert old.read_bytes()==original
        assert not (target/'patch-backups').exists()
    else:
        assert run.returncode==0,run.stdout.decode(errors='replace')+run.stderr.decode(errors='replace')
        assert old.read_bytes()==b'new model'
        backups=list((target/'patch-backups').glob('*/models/vehicle_model.pt'))
        assert len(backups)==1 and backups[0].read_bytes()==b'old model'

@unittest.skipIf(os.name!='nt','Windows recipient installer')
class RecipientPatchTests(unittest.TestCase):
    def test_install_and_preserve(self):
        with tempfile.TemporaryDirectory() as d:check_recipient_patch(Path(d),None)
    def test_reject_corrupt_payload(self):
        with tempfile.TemporaryDirectory() as d:check_recipient_patch(Path(d),'payload')
    def test_reject_changed_owner_model(self):
        with tempfile.TemporaryDirectory() as d:check_recipient_patch(Path(d),'baseline')
