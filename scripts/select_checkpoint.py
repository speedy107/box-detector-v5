"""Select a saved checkpoint after a completed/recovered run."""
import argparse,json,shutil
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer
from evaluate_models import evaluate,sampled
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(8);rows=json.loads((a.dataset/'manifest.json').read_text());old,extra=sampled(rows)
raw=json.loads((a.output/'training/checkpoint-regressions.json').read_text());records={}
for r in raw:records.setdefault(r['epoch'],r)
(a.output/'training/checkpoint-regressions.json').write_text(json.dumps(list(records.values()),indent=2))
passing=[r for r in records.values() if r['old_pass']==r['old_total'] and r['validation_map50_95']>=.8595]
assert passing,'No candidate passed the known cases and validation floor'
passing.sort(key=lambda r:(r['expanded_pass'],r['validation_map50_95']),reverse=True)
selected=passing[0];weights=a.run/'weights'/f"epoch{selected['epoch']-1}.pt"
rep=evaluate(YOLO(str(weights)),a.dataset,old+extra)
assert rep['summary']['0.35']['passed']==selected['old_pass']+selected['expanded_pass'],'Regression check disagrees with training log'
strip_optimizer(weights,s=a.output/'models/boxes-yolo11n-v5.pt')
(a.output/'training/selection.json').write_text(json.dumps(selected,indent=2));print(selected)
