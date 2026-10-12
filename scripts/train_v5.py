import os, argparse, json, shutil, datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'training'/'ultralytics-config'))
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('MKL_NUM_THREADS','8')
import torch,yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.torch_utils import unwrap_model
from evaluate_models import evaluate,sampled
class SquareTrainer(DetectionTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        stride=max(int(unwrap_model(self.model).stride.max() if self.model else 0),32)
        return build_yolo_dataset(self.args,img_path,batch,self.data,mode=mode,rect=False,stride=stride)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--weights',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'training'/'runs');ap.add_argument('--stop-after',type=int,default=9);ap.add_argument('--resume',type=Path);ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=8);args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    ds=args.dataset.resolve();config=yaml.safe_load((ds/'data_balanced.yaml').read_text());config['path']=str(ds)
    resolved=ROOT/'training'/'resolved-data.yaml';resolved.write_text(yaml.safe_dump(config))
    rows=json.loads((ds/'manifest.json').read_text());old,extra=sampled(rows);history=[];rank=(-1,-1);selected=None
    def callback(trainer):
        nonlocal rank,selected
        epoch=trainer.epoch+1
        if history and epoch<=history[-1]['epoch']:return
        probe=YOLO(str(trainer.last));rep=evaluate(probe,ds,old+extra)
        passes=[r['thresholds']['0.35']['passed'] for r in rep['outcomes']]
        a=sum(passes[:len(old)]);b=sum(passes[len(old):]);m=float(trainer.metrics.get('metrics/mAP50-95(B)',0))
        record={'epoch':epoch,'old_pass':a,'old_total':len(old),'expanded_pass':b,'expanded_total':len(extra),'validation_map50_95':m}
        history.append(record)
        if a==len(old) and m>=.8595 and (b,m)>rank:
            rank=(b,m);selected=record;shutil.copy2(trainer.last,ROOT/'training'/'selected-v5.pt')
            (ROOT/'training'/'selection.json').write_text(json.dumps(selected,indent=2))
        (ROOT/'training'/'checkpoint-regressions.json').write_text(json.dumps(history,indent=2))
        print('V5 REGRESSION',record,'SELECTED',selected,flush=True)
        if epoch>=args.stop_after:trainer.stop=True
    model=YOLO(str(args.resume or args.weights or ds/'weights'/'boxes-yolo11n-v4.pt'));model.add_callback('on_fit_epoch_end',callback)
    model.train(trainer=SquareTrainer,resume=bool(args.resume),data=str(resolved),epochs=60,patience=12,save_period=1,imgsz=320,batch=16,device=args.device,workers=0,project=str(args.output.resolve()),name='v5-hybrid',exist_ok=True,seed=107,optimizer='AdamW',lr0=.0002,lrf=.01,weight_decay=.0005,warmup_epochs=1,warmup_bias_lr=.001,rect=False,hsv_h=0,hsv_s=.1,hsv_v=.15,degrees=0,translate=.03,scale=.15,shear=0,perspective=0,fliplr=0,flipud=0,mosaic=.05,close_mosaic=10,mixup=0,copy_paste=0,amp=False,cache='ram',plots=True,save=True)
    assert selected,'No suitable V5 checkpoint found'
    from ultralytics.utils.torch_utils import strip_optimizer
    strip_optimizer(ROOT/'training'/'selected-v5.pt',s=ROOT/'models'/'boxes-yolo11n-v5.pt')
    summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epochs_completed':history[-1]['epoch'],'selection':selected,'dataset_train_images':1390,'validation_images':23,'balanced_training_positions':2502,'runtime':'CPU','note':'Recovered by retraining after the previous ephemeral workspace was lost.'}
    (ROOT/'training'/'training-summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
