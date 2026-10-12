import argparse,hashlib,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,default=Path('datasets'));a=p.parse_args()
expected='101c8748f0e77405ad9c67217ec5271bfd6272cac81846277bd16573b647a3a4'
assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==expected,'Dataset hash mismatch'
with zipfile.ZipFile(a.archive) as z:
    for entry in z.infolist():
        dst=(a.output/entry.filename).resolve()
        assert dst.is_relative_to(a.output.resolve()),'Unsafe archive path'
    z.extractall(a.output)
print(a.output.resolve())
