import argparse,json,hashlib
from pathlib import Path
import torch,numpy as np,onnx,onnxruntime as ort
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2
p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(8)
model=YOLO(str(a.weights));exported=Path(model.export(format='onnx',imgsz=320,batch=1,dynamic=False,half=False,simplify=False,opset=17,device='cpu',nms=False))
onnx.checker.check_model(str(exported))
o=ort.SessionOptions();o.intra_op_num_threads=8
session=ort.InferenceSession(str(exported),sess_options=o,providers=['CPUExecutionProvider'])
model.model.eval();coord=[];scores=[]
for image in sorted((a.dataset/'images/val').glob('*.jpg')):
    im=cv2.imread(str(image));im=LetterBox((320,320),auto=False)(image=im)
    x=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    with torch.no_grad():native=model.model(torch.from_numpy(x))[0].numpy()
    out=session.run(None,{'images':x})[0]
    coord.append(float(np.max(np.abs(native[:,:4]-out[:,:4]))));scores.append(float(np.max(np.abs(native[:,4:]-out[:,4:]))))
assert max(coord)<.05 and max(scores)<1e-4,(max(coord),max(scores))
a.output.write_text(json.dumps({'checked':True,'validation_images':len(coord),'input_shape':[1,3,320,320],'output_shape':list(out.shape),'max_coordinate_error':max(coord),'max_class_score_error':max(scores),'sha256':hashlib.sha256(exported.read_bytes()).hexdigest()},indent=2))
print(a.output.read_text())
