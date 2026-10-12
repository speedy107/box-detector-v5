import json, argparse
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1])
    return inter/(area(a)+area(b)-inter+1e-9)

def score(gt,det,threshold=.35):
    ds=[b for b in det if b['confidence']>=threshold]
    tp=0
    if gt and ds:
        bonus=min(len(gt),len(ds))+1
        costs=[]
        for g in gt:
            row=[]
            for b in ds:
                v=iou(g['xyxy'],b['xyxy']) if g['class_id']==b['class_id'] else 0
                row.append(-(bonus+v) if v>=.5 else 0)
            costs.append(row)
        rows,cols=linear_sum_assignment(costs)
        tp=sum(costs[r][c]<0 for r,c in zip(rows,cols))
    return {'tp':int(tp),'fp':len(ds)-int(tp),'fn':len(gt)-int(tp),'passed':tp==len(gt)==len(ds)}

def evaluate(model,dataset,rows):
    outcomes=[]
    for start in range(0,len(rows),16):
        batch=rows[start:start+16]
        preds=model.predict([str(dataset/r['image']) for r in batch],imgsz=320,rect=False,device='cpu',conf=.25,iou=.45,verbose=False)
        for row,p in zip(batch,preds):
            det=[{'class_id':int(b.cls.item()),'confidence':float(b.conf.item()),'xyxy':b.xyxy[0].tolist()} for b in p.boxes]
            outcomes.append({'image':row['image'],'split':row['split'],'kind':row['kind'],'targets':row['objects'],'detections':det,'thresholds':{str(t):score(row['objects'],det,t) for t in [.25,.35,.5]}})
    summary={}
    for t in ['0.25','0.35','0.5']:
        s={k:sum(o['thresholds'][t][k] for o in outcomes) for k in ['tp','fp','fn','passed']}
        s.update(total=len(rows),precision=s['tp']/max(1,s['tp']+s['fp']),recall=s['tp']/max(1,s['tp']+s['fn']))
        summary[t]=s
    return {'summary':summary,'outcomes':outcomes}

def sampled(rows):
    old=[r for r in rows if r['split']=='val' or Path(r['image']).stem.startswith(('new_v3_','new_v4_'))]
    kinds=sorted({r['kind'] for r in rows if r['kind']!='original_real'})
    extra=[]
    for k in kinds:
        candidates=[r for r in rows if r['kind']==k]
        extra.extend(candidates[::max(1,len(candidates)//30)][:30])
    return old,extra

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--v4',type=Path,required=True);ap.add_argument('--v5',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(8)
    rows=json.loads((args.dataset/'manifest.json').read_text());result={}
    for name,weights in [('v4',args.v4),('v5',args.v5)]:
        result[name]=evaluate(YOLO(str(weights)),args.dataset,rows)
        print(name,result[name]['summary'],flush=True)
    args.output.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
