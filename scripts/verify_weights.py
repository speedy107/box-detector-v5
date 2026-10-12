import argparse,json,hashlib
from pathlib import Path
import torch
from ultralytics import YOLO
p=argparse.ArgumentParser();p.add_argument('--v4',type=Path,required=True);p.add_argument('--v5',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x=YOLO(str(a.v4)).model.float().state_dict();y=YOLO(str(a.v5)).model.float().state_dict()
changed=[n for n in x if x[n].is_floating_point() and not torch.equal(x[n],y[n])];assert changed,'Weights did not change'
a.output.write_text(json.dumps({'changed_floating_point_tensors':len(changed),'v4_sha256':hashlib.sha256(a.v4.read_bytes()).hexdigest(),'v5_sha256':hashlib.sha256(a.v5.read_bytes()).hexdigest()},indent=2))
