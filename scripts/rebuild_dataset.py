"""Rebuild images from supplied original archive and the committed assets."""
import argparse,zipfile,subprocess,sys,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('original_archive',type=Path);p.add_argument('--output',type=Path,default=Path('datasets/rebuilt'));p.add_argument('--v4-weights',type=Path);a=p.parse_args()
source=a.output.parent/'original'
with zipfile.ZipFile(a.original_archive) as z:
    for e in z.infolist():assert (source/e.filename).resolve().is_relative_to(source.resolve())
    z.extractall(source)
manifest=next(source.rglob('manifest.json'))
subprocess.run([sys.executable,str(root/'scripts/build_dataset.py'),'--source',str(manifest.parent),'--assets',str(root/'training-assets'),'--output',str(a.output)],check=True)
subprocess.run([sys.executable,str(root/'scripts/validate_dataset.py'),str(a.output)],check=True)

if a.v4_weights:
    (a.output/'weights').mkdir(exist_ok=True)
    shutil.copy2(a.v4_weights,a.output/'weights/boxes-yolo11n-v4.pt')
