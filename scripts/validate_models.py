import argparse,json
from pathlib import Path
import torch,yaml
from ultralytics import YOLO
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--v4',type=Path,required=True);p.add_argument('--v5',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(8)
config=yaml.safe_load((a.dataset/'data.yaml').read_text());config['path']=str(a.dataset.resolve())
resolved=a.output.parent/'validation-data.yaml';resolved.write_text(yaml.safe_dump(config))
result={}
for name,weights in [('v4',a.v4),('v5',a.v5)]:
    m=YOLO(str(weights)).val(data=str(resolved.resolve()),imgsz=320,rect=False,batch=16,device='cpu',workers=0,plots=False,verbose=False,project=str(a.output.parent),name=name)
    result[name]={'precision':float(m.box.mp),'recall':float(m.box.mr),'map50':float(m.box.map50),'map50_95':float(m.box.map),'per_class_map50_95':m.box.maps.tolist()}
a.output.write_text(json.dumps(result,indent=2));print(result)
