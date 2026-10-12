"""Build a checked repository ZIP from committed git contents."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
files=subprocess.check_output(['git','-C',str(a.root),'ls-files','-z']).decode().split('\0')
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name in files:
  if name:z.write(a.root/name,'box-detector-v5/'+name)
with zipfile.ZipFile(a.output) as z:assert z.testzip() is None
print(json.dumps({'file':str(a.output.resolve()),'bytes':a.output.stat().st_size,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))
