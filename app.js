import { INPUT_SIZE, CLASSES, COLORS, letterbox, rgbaToTensor, decode } from './detector.js';
const $=id=>document.getElementById(id);
const video=$('video'),photo=$('photo'),overlay=$('overlay');
const prep=document.createElement('canvas');prep.width=prep.height=INPUT_SIZE;
const prepContext=prep.getContext('2d',{willReadFrequently:true}),context=overlay.getContext('2d');
let sessionPromise=null,stream=null,running=false,generation=0,inferenceBusy=false,currentPhoto=null,photoBusy=false;
const status=(text,error=false)=>{$('status').textContent=text;$('status').classList.toggle('error',error);};
const threshold=()=>Number($('confidence').value)/100;
async function loadModel(){
 if(sessionPromise)return sessionPromise;
 sessionPromise=(async()=>{
  if(!globalThis.ort)throw new Error('The inference runtime did not load. Reload the page.');
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths=new URL('./vendor/',location.href).href;
  status('Downloading the model…');
  const response=await fetch(new URL('./models/boxes-yolo11n-v5.onnx?v=5',location.href));
  if(!response.ok)throw new Error(`Model download failed (${response.status}). Try reloading.`);
  const bytes=await response.arrayBuffer();status('Preparing the model on your device…');
  return await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
 })().catch(error=>{sessionPromise=null;throw error;});return sessionPromise;
}
function setView(source,width,height,label){
 $('viewer').style.aspectRatio=`${width} / ${height}`;
 $('viewer').style.maxWidth=`min(100%, ${58*width/height}vh)`;
 $('viewer').style.marginInline='auto';
 overlay.width=width;overlay.height=height;overlay.hidden=false;
 $('placeholder').hidden=true;video.hidden=source!=='video';photo.hidden=source!=='photo';
 $('viewBadge').hidden=false;$('viewBadge').textContent=label;$('saveFrame').disabled=false;
}
function paint(boxes,width,height,elapsed){
 context.clearRect(0,0,width,height);
 const font=Math.max(14,Math.round(width*.025));context.font=`600 ${font}px system-ui`;context.lineWidth=Math.max(2,width*.004);
 for(const box of boxes){
  const color=COLORS[box.classId];context.strokeStyle=color;context.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);
  const text=`${CLASSES[box.classId]} ${Math.round(box.score*100)}%`,textWidth=context.measureText(text).width+14;
  const labelX=Math.min(box.x1,Math.max(0,width-textWidth)),labelY=Math.max(0,box.y1-font-12);
  context.fillStyle=color;context.fillRect(labelX,labelY,textWidth,font+12);context.fillStyle='#091321';context.fillText(text,labelX+7,labelY+font+2);
 }
 const counts=[0,0];boxes.forEach(b=>counts[b.classId]++);
 $('detections').textContent=boxes.length?counts.map((n,i)=>n?`${n} ${CLASSES[i].toLowerCase()}${n>1?'s':''}`:'').filter(Boolean).join(' · '):'No box detected';
 $('timing').textContent=`${Math.round(elapsed)} ms / detection`;
}
async function infer(source,width,height){
 const model=await loadModel(),geometry=letterbox(width,height);
 prepContext.fillStyle='rgb(114,114,114)';prepContext.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
 prepContext.drawImage(source,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);
 const pixels=prepContext.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data;
 const tensor=new ort.Tensor('float32',rgbaToTensor(pixels),[1,3,INPUT_SIZE,INPUT_SIZE]);let outputs;
 const start=performance.now();
 try{outputs=await model.run({[model.inputNames[0]]:tensor});return {boxes:decode(outputs[model.outputNames[0]],geometry,threshold()),elapsed:performance.now()-start};}
 finally{tensor.dispose();if(outputs)Object.values(outputs).forEach(output=>output.dispose());}
}
function stopCamera(message='Camera stopped.'){
 running=false;generation++;if(stream)stream.getTracks().forEach(track=>track.stop());
 stream=null;video.srcObject=null;$('start').disabled=false;$('stop').disabled=true;
 if(!currentPhoto){video.hidden=true;overlay.hidden=true;$('placeholder').hidden=false;$('viewBadge').hidden=true;$('detections').textContent='No feed yet';$('timing').textContent='';$('saveFrame').disabled=true;}
 status(message);
}
async function cameraLoop(token){
 if(!running||token!==generation)return;
 if(inferenceBusy||photoBusy||video.readyState<2){setTimeout(()=>cameraLoop(token),100);return;}
 inferenceBusy=true;
 try{
  const width=video.videoWidth,height=video.videoHeight,result=await infer(video,width,height);
  if(running&&token===generation){if(overlay.width!==width||overlay.height!==height)setView('video',width,height,'LIVE');paint(result.boxes,width,height,result.elapsed);status('Live detection · processing on this device');}
 }catch(error){if(token===generation){stopCamera();status(`Detection failed: ${error.message}`,true);}}
 finally{inferenceBusy=false;}
 if(running&&token===generation)setTimeout(()=>cameraLoop(token),100);
}
$('start').addEventListener('click',async()=>{
 if(running)return;
 if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){status('Camera access requires HTTPS. Open the published GitHub Pages link in Chrome.',true);return;}
 currentPhoto=null;photo.hidden=true;running=true;const token=++generation;$('start').disabled=true;$('stop').disabled=false;
 try{
  status('Allow camera access when Chrome asks.');
  const camera=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}}});
  if(!running||token!==generation){camera.getTracks().forEach(t=>t.stop());return;}
  stream=camera;video.srcObject=camera;video.hidden=false;await video.play();
  if(!running||token!==generation)return;
  setView('video',video.videoWidth,video.videoHeight,'LIVE');await loadModel();
  if(running&&token===generation)cameraLoop(token);
 }catch(error){
  if(token!==generation)return;stopCamera();
  const errors={NotAllowedError:'Camera access was denied. Allow camera access in Chrome site settings and try again.',NotFoundError:'No camera was found. You can still choose a photo.',NotReadableError:'The camera is busy. Close other camera apps and try again.'};
  status(errors[error.name]||`Could not start: ${error.message}`,true);
 }
});
$('stop').addEventListener('click',()=>stopCamera());
async function analyzePhoto(){
 if(!currentPhoto||running||photoBusy)return;photoBusy=true;
 try{
  while(inferenceBusy)await new Promise(resolve=>setTimeout(resolve,30));
  const selected=currentPhoto,result=await infer(photo,photo.width,photo.height);
  if(selected===currentPhoto&&!running){paint(result.boxes,photo.width,photo.height,result.elapsed);status('Photo detection complete · image stays on this device');}
 }catch(error){status(`Could not analyze this image: ${error.message}`,true);}
 finally{photoBusy=false;}
}
$('file').addEventListener('change',async event=>{
 const file=event.target.files[0];if(!file)return;stopCamera('Opening your photo…');
 try{
  const bitmap=await createImageBitmap(file),scale=Math.min(1,1600/Math.max(bitmap.width,bitmap.height));
  photo.width=Math.round(bitmap.width*scale);photo.height=Math.round(bitmap.height*scale);
  photo.getContext('2d').drawImage(bitmap,0,0,photo.width,photo.height);bitmap.close();currentPhoto=file;
  setView('photo',photo.width,photo.height,'PHOTO');await analyzePhoto();
 }catch(error){status(`Could not open the image: ${error.message}`,true);}event.target.value='';
});
$('confidence').addEventListener('input',()=>{$('confidenceValue').textContent=`${$('confidence').value}%`;});
$('confidence').addEventListener('change',()=>{if(currentPhoto)analyzePhoto();});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stopCamera('Camera paused while the page was hidden. Tap Start camera to resume.');});
window.addEventListener('pagehide',()=>{if(stream)stream.getTracks().forEach(track=>track.stop());});

$('saveFrame').addEventListener('click',()=>{
 const source=running&&video.readyState>=2?video:currentPhoto?photo:null;
 if(!source)return;
 const width=source===video?video.videoWidth:photo.width,height=source===video?video.videoHeight:photo.height;
 const frame=document.createElement('canvas');frame.width=width;frame.height=height;
 frame.getContext('2d').drawImage(source,0,0,width,height);
 frame.toBlob(blob=>{
  if(!blob){status('Could not save this frame. Please try again.',true);return;}
  const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;
  link.download=`box-example-${Date.now()}.jpg`;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),30000);
 },'image/jpeg',.92);
});
