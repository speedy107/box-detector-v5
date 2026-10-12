import argparse
from pathlib import Path
from ultralytics import YOLO
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--weights',default=str(Path(__file__).resolve().parents[1]/'models/boxes-yolo11n-v5.pt'));p.add_argument('--device',default='cpu');p.add_argument('--conf',type=float,default=.35);a=p.parse_args()
# Iterate the stream so inference and saved output actually happen.
for result in YOLO(a.weights).predict(source=int(a.source) if a.source.isdigit() else a.source,imgsz=320,rect=False,conf=a.conf,iou=.45,device=a.device,save=True,stream=True):
    print(result.path,len(result.boxes))
