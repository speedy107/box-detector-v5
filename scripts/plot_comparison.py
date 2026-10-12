import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.report.read_text());maps={n:{o['image']:o for o in v['outcomes']} for n,v in r.items()}
colors={0:(60,130,255),1:(235,65,80)}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
selected=[]
for kind in ['small_block','single_block','multi_block','occlusion','stack','synthetic_negative']:
    candidates=[o for o in r['v5']['outcomes'] if o['kind']==kind and o['thresholds']['0.35']['passed'] and not maps['v4'][o['image']]['thresholds']['0.35']['passed']]
    if candidates:selected.append(candidates[0]['image'])
if not selected:selected=list(maps['v5'])[:6]
def tile(name,image,width=600,height=400):
    o=maps[name][image];src=Image.open(a.dataset/image).convert('RGB');scale=min(width/src.width,(height-40)/src.height);src=src.resize((round(src.width*scale),round(src.height*scale)));out=Image.new('RGB',(width,height),(248,248,246));left=(width-src.width)//2;top=40;out.paste(src,(left,top));d=ImageDraw.Draw(out)
    s=o['thresholds']['0.35'];d.text((8,8),f"{name.upper()} · {Path(image).stem} · FP {s['fp']} / FN {s['fn']}",font=small,fill=(30,30,30))
    for g in o['targets']:
        b=[g['xyxy'][i]*scale+([left,top][i%2]) for i in range(4)];d.rectangle(b,outline=(255,255,255),width=2)
    for b in o['detections']:
        if b['confidence']<.35:continue
        xy=[b['xyxy'][i]*scale+([left,top][i%2]) for i in range(4)];c=colors[b['class_id']];d.rectangle(xy,outline=c,width=3);label=('blue' if b['class_id']==0 else 'red')+f" {b['confidence']:.2f}";d.text((xy[0]+3,xy[1]+3),label,font=small,fill=c,stroke_width=1,stroke_fill=(255,255,255))
    return out
sheet=Image.new('RGB',(1200,400*len(selected)+48),(248,248,246));d=ImageDraw.Draw(sheet);d.text((12,12),'V4 versus V5 · expanded TRAINING examples · white = labels; colour = predictions',font=font,fill=(30,30,30))
for i,image in enumerate(selected):
    for j,n in enumerate(['v4','v5']):sheet.paste(tile(n,image),(j*600,48+i*400))
sheet.save(a.output/'prediction-comparison.jpg',quality=90)
errors=[o for o in r['v5']['outcomes'] if not o['thresholds']['0.35']['passed']]
(a.output/'remaining-errors.json').write_text(json.dumps(errors,indent=2))
if errors:
    subset=errors[:8];sheet=Image.new('RGB',(1200,400*((len(subset)+1)//2)+48),(248,248,246));ImageDraw.Draw(sheet).text((12,12),f'V5 remaining errors · {len(errors)} images · preview of first {len(subset)}',font=font,fill=(30,30,30))
    for i,o in enumerate(subset):sheet.paste(tile('v5',o['image']),(600*(i%2),48+400*(i//2)))
    sheet.save(a.output/'remaining-errors.jpg',quality=90)
print('Plotted examples',len(selected),'remaining errors',len(errors))
