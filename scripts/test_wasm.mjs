import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {letterbox,rgbaToTensor,decode} from '../detector.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(process.env.V5_TEST_NODE_MODULES?path.join(process.env.V5_TEST_NODE_MODULES,'../package.json'):path.join(root,'package.json'));
const ort=require('onnxruntime-web'),sharp=require('sharp');
ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
const ds=path.resolve(process.argv[2]||'datasets/blocks_dataset_10x');
const all=JSON.parse(fs.readFileSync(path.join(ds,'manifest.json')));
const rows=all.filter(r=>r.split==='val'||/^new_v[34]_/.test(path.basename(r.image)));
const session=await ort.InferenceSession.create(fs.readFileSync(path.join(root,'models/boxes-yolo11n-v5.onnx')),{executionProviders:['wasm']});
function iou(a,b){const area=v=>Math.max(0,v[2]-v[0])*Math.max(0,v[3]-v[1]);const n=area([Math.max(a[0],b[0]),Math.max(a[1],b[1]),Math.min(a[2],b[2]),Math.min(a[3],b[3])]);return n/(area(a)+area(b)-n+1e-9);}
const outcomes=[];
for(const row of rows){
  const image=path.join(ds,row.image),meta=await sharp(image).metadata(),g=letterbox(meta.width,meta.height);
  const data=await sharp(image).resize(g.resizedWidth,g.resizedHeight,{kernel:'linear'}).extend({left:g.left,top:g.top,right:320-g.resizedWidth-g.left,bottom:320-g.resizedHeight-g.top,background:{r:114,g:114,b:114,alpha:1}}).ensureAlpha().raw().toBuffer();
  const result=await session.run({images:new ort.Tensor('float32',rgbaToTensor(data),[1,3,320,320])});
  const detections=decode(result.output0,g,.35,.45);const available=new Set(detections.map((_,i)=>i));
  let matched=0;
  for(const target of row.objects){let best=-1,overlap=.5;for(const index of available){const d=detections[index];const v=iou(target.xyxy,[d.x1,d.y1,d.x2,d.y2]);if(d.classId===target.class_id&&v>=overlap){best=index;overlap=v;}}if(best>=0){matched++;available.delete(best);}}
  outcomes.push({image:row.image,expected:row.objects.length,detected:detections.length,passed:matched===row.objects.length&&available.size===0});
}
const result={runtime:'onnxruntime-web 1.20.1 WASM in Node; repository decoder; not a browser UI test',confidence:.35,total:rows.length,passed:outcomes.filter(o=>o.passed).length,outcomes};
fs.writeFileSync(path.join(root,'evaluation/wasm-results.json'),JSON.stringify(result,null,2));console.log(result.total,result.passed);if(result.passed!==result.total)process.exitCode=1;
