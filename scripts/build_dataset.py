#!/usr/bin/env python3
"""Deterministic YOLO dataset expansion with real silhouettes and generated backgrounds.

Dataset processing: masks and numerical label geometry are kept together. GenAI
supplies eight background assets; it never supplies automatic target annotations.
Requires numpy, Pillow and PyYAML. See README.md for limits and reproduction.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import yaml

SEED = 10720261011
SIZE = (640, 480)
CLASSES = ['blue_box', 'red_box']
COLORS = [(30, 120, 255), (255, 40, 60)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_labels(path, width, height):
    result = []
    for line in Path(path).read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        result.append({'class_id': int(c), 'xyxy': [
            (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]})
    return result


def write_labels(path, objects, width, height):
    lines = []
    for obj in objects:
        x1, y1, x2, y2 = obj['xyxy']
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        lines.append(f"{obj['class_id']} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    Path(path).write_text('\n'.join(lines) + ('\n' if lines else ''))


def photometric(im, rng, mild=False):
    """Only changes appearance, leaving geometry/labels untouched and hue classes intact."""
    low, high = (.85, 1.12) if mild else (.72, 1.20)
    params = {'brightness': rng.uniform(low, high),
              'contrast': rng.uniform(.88, 1.13),
              'saturation': rng.uniform(.82, 1.12),
              'gamma': rng.uniform(.88, 1.16),
              'temperature_gain': rng.uniform(-.025, .025),
              'blur_radius': rng.choice([0, 0, .25, .4] if mild else [0, 0, .3, .6, .9]),
              'noise_std': rng.uniform(0, 1.8)}
    im = ImageEnhance.Brightness(im).enhance(params['brightness'])
    im = ImageEnhance.Contrast(im).enhance(params['contrast'])
    im = ImageEnhance.Color(im).enhance(params['saturation'])
    arr = np.array(im, dtype=np.float32)/255
    arr = np.power(np.clip(arr, 0, 1), params['gamma'])
    t = params['temperature_gain']
    arr *= np.array([1+t, 1, 1-t], dtype=np.float32)
    noise_seed = rng.randrange(2**32)
    params['noise_seed'] = noise_seed
    arr = arr*255 + np.random.default_rng(noise_seed).normal(0, params['noise_std'], arr.shape)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if params['blur_radius']:
        im = im.filter(ImageFilter.GaussianBlur(params['blur_radius']))
    return im, params


def affine_variant(im, objects, rng):
    """Warp full image and all label corners with the same affine matrix."""
    im = ImageOps.exif_transpose(im).convert('RGB')
    original_size = im.size
    im.thumbnail((768, 768), Image.Resampling.LANCZOS)
    sx, sy = im.width/original_size[0], im.height/original_size[1]
    boxes = [{**o, 'xyxy': [o['xyxy'][0]*sx, o['xyxy'][1]*sy,
                           o['xyxy'][2]*sx, o['xyxy'][3]*sy]} for o in objects]
    w, h = im.size
    for _ in range(200):
        angle = math.radians(rng.uniform(-4, 4))
        scale = rng.uniform(.88, 1.02)
        shear = rng.uniform(-.025, .025)
        co, si = math.cos(angle), math.sin(angle)
        a = scale*np.array([[co, -si+shear], [si, co]])
        center = np.array([w/2, h/2])
        shift = center-a@center + np.array([rng.uniform(-.02, .02)*w, rng.uniform(-.02, .02)*h])
        matrix = np.eye(3)
        matrix[:2, :2] = a
        matrix[:2, 2] = shift
        new = []
        valid = True
        for o in boxes:
            x1, y1, x2, y2 = o['xyxy']
            corners = np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]])@matrix.T
            mn, mx = corners[:,:2].min(0), corners[:,:2].max(0)
            unclipped = np.prod(mx-mn)
            mn = np.maximum(mn, 0)
            mx = np.minimum(mx, [w,h])
            clipped = np.prod(np.maximum(mx-mn, 0))
            if min(mx-mn) < 12 or clipped/max(unclipped, 1) < .88:
                valid = False
                break
            new.append({'class_id': o['class_id'], 'xyxy': [*mn.tolist(), *mx.tolist()]})
        if valid:
            inverse = np.linalg.inv(matrix)
            warped = im.transform((w,h), Image.Transform.AFFINE,
                tuple(inverse[:2].flatten()), Image.Resampling.BICUBIC, fillcolor=(114,114,114))
            warped, pp = photometric(warped, rng)
            return warped, new, {'source_resize': [sx,sy], 'affine_matrix': matrix.tolist(), 'photometric': pp}
    raise RuntimeError('Could not find safe full-image affine transform')


def load_background(path, rng):
    im = Image.open(path).convert('RGB')
    # Small center-crop/zoom variants keep the originally reviewed central floor usable.
    crop_factor = rng.uniform(.88, 1.0)
    cw, ch = int(im.width*crop_factor), int(im.height*crop_factor)
    x = rng.randrange(im.width-cw+1)
    y = rng.randrange(im.height-ch+1)
    crop = [x,y,x+cw,y+ch]
    im = ImageOps.fit(im.crop(tuple(crop)), SIZE, method=Image.Resampling.LANCZOS)
    return im, crop


def sprite(asset, width, rng):
    im = Image.open(asset['path']).convert('RGBA')
    height = max(4, round(im.height*width/im.width))
    im = im.resize((width,height), Image.Resampling.LANCZOS)
    rgb = Image.merge('RGB', im.split()[:3])
    brightness = rng.uniform(.80, 1.12)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    im = Image.merge('RGBA', (*rgb.split(), im.getchannel('A')))
    return im, brightness


def compose(bg, placements, rng):
    """Return image and visible-mask envelopes; all nonempty targets must be labelled."""
    w,h = bg.size
    prepared=[]
    for asset, width, x, y in placements:
        im, brightness = sprite(asset,width,rng)
        if x < 0 or y < 0 or x+im.width > w or y+im.height > h:
            return None
        layer = Image.new('RGBA',(w,h))
        layer.alpha_composite(im,(int(x),int(y)))
        prepared.append((asset, layer, int(x), int(y), brightness))
    # Approximate contact shadows. These are compositing cues, not a calibrated renderer.
    shadow = Image.new('RGBA',(w,h))
    sd=ImageDraw.Draw(shadow)
    for i,(asset,layer,x,y,bright) in enumerate(prepared):
        bounds=layer.getchannel('A').getbbox()
        x1,y1,x2,y2=bounds
        if i == 0 or not any(x1 < p[1].getchannel('A').getbbox()[2] and x2 > p[1].getchannel('A').getbbox()[0] and y2 < p[1].getchannel('A').getbbox()[3] for p in prepared[:i]):
            sd.ellipse((x1+3,y2-4,x2+6,min(h,y2+max(3,(x2-x1)*.06))),fill=(0,0,0,55))
    shadow=shadow.filter(ImageFilter.GaussianBlur(5))
    output=bg.convert('RGBA')
    output.alpha_composite(shadow)
    for _,layer,_,_,_ in prepared:output.alpha_composite(layer)
    visible_masks=[]
    objs=[]
    for i,(asset,layer,x,y,brightness) in enumerate(prepared):
        full=np.array(layer.getchannel('A'),dtype=np.float32)/255
        visible=full.copy()
        for _,front,_,_,_ in prepared[i+1:]:
            visible*=1-np.array(front.getchannel('A'),dtype=np.float32)/255
        binary=visible>.5
        ys,xs=np.where(binary)
        if len(xs)<30 or binary.sum()/max((full>.5).sum(),1) < .35:
            return None
        box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        if min(box[2]-box[0],box[3]-box[1]) < 10:return None
        visible_masks.append(binary)
        objs.append({'class_id':asset['class_id'],'xyxy':box,
            'cutout_source':asset['source'],'visible_fraction':float(binary.sum()/max((full>.5).sum(),1)),
            'placement':[x,y],'sprite_width':placements[i][1],
            'sprite_brightness':brightness,'full_mask_xyxy':list(layer.getchannel('A').getbbox())})
    output,pp=photometric(output.convert('RGB'),rng,mild=any(min(o['xyxy'][2]-o['xyxy'][0],o['xyxy'][3]-o['xyxy'][1]) < 55 for o in objs))
    return output,objs,visible_masks,pp


def composite_sample(kind,index,bg,assets,rng):
    w,h=bg.size
    groups={c:[a for a in assets if a['class_id']==c] for c in [0,1]}
    for attempt in range(500):
        placements=[]
        if kind in ['small_block','single_block']:
            c=index%2
            width=rng.randint(26,86) if kind=='small_block' else rng.randint(88,230)
            a=rng.choice(groups[c]);sh=round(Image.open(a['path']).height*width/Image.open(a['path']).width)
            floor_y=rng.randint(290,360) if kind=='small_block' else rng.randint(max(sh+160,365),470)
            x=rng.randint(max(20,int(w*.25)),min(w-width-20,int(w*.72)))
            placements=[(a,width,x,floor_y-sh)]
        elif kind in ['multi_block','occlusion']:
            count=2 if index%3 else 3
            width=rng.randint(46,120)
            classes=([(index//4)%2]*count) if index%4==0 else [(index+i)%2 for i in range(count)]
            floor_y=rng.randint(345,466)
            start=rng.randint(130,230)
            gap=.90 if kind=='occlusion' else 1.10
            for i,c in enumerate(classes):
                a=rng.choice(groups[c]);sw=round(width*rng.uniform(.93,1.07));sh=round(Image.open(a['path']).height*sw/Image.open(a['path']).width)
                x=round(start+i*width*gap);y=floor_y-sh+rng.randint(-8,8)
                placements.append((a,sw,x,y))
        elif kind=='stack':
            count=3 if index%3==0 else 2
            width=rng.randint(55,140) if count==3 else rng.randint(62,170)
            classes=([(index//4)%2]*count) if index%4==0 else [(index+i)%2 for i in range(count)]
            floor_y=rng.randint(380,466)
            x=rng.randint(170,440-width)
            prev_y=None;prev_h=None
            for i,c in enumerate(classes):
                # Similar low-angle views avoid absurdly different perspective within a stack.
                candidates=[a for a in groups[c] if a['source'] not in ['red_clip_04_frame_0004','blue_clip_03_frame_0005']]
                a=rng.choice(candidates);sw=round(width*rng.uniform(.96,1.04));sh=round(Image.open(a['path']).height*sw/Image.open(a['path']).width)
                y=floor_y-sh if i==0 else round(prev_y-sh+prev_h*.22)
                placements.append((a,sw,x+rng.randint(-3,3),y));prev_y=y;prev_h=sh
        else:raise ValueError(kind)
        result=compose(bg,placements,rng)
        if result:
            image,objects,masks,params=result
            return image,objects,masks,params
    raise RuntimeError(f'Cannot create scene: {kind} {index}')


def generate_preview(root,records):
    choices=[]
    for kind in ['small_block','single_block','multi_block','occlusion','stack','synthetic_negative','real_positive_aug','real_negative_aug']:
        matches=[r for r in records if r['kind']==kind]
        choices.extend(matches[:3])
    width=1400;tile_w=350;tile_h=300
    sheet=Image.new('RGB',(width,80+math.ceil(len(choices)/4)*tile_h),(245,245,245))
    draw=ImageDraw.Draw(sheet)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
    except OSError:font=ImageFont.load_default();title=font
    draw.text((20,14),'BLOCK DATASET — 10× HYBRID EXPANSION',font=title,fill=(20,20,20))
    draw.text((20,46),'Blue/red outlines = target labels. Chairs, bags and clothing stay unlabelled.',font=font,fill=(50,50,50))
    for i,r in enumerate(choices):
        im=Image.open(root/r['image']).convert('RGB')
        # Inspect at native scale first; preview draws only the saved label file.
        d=ImageDraw.Draw(im)
        for o in parse_labels(root/r['label'],im.width,im.height):
            d.rectangle(tuple(o['xyxy']),outline=COLORS[o['class_id']],width=max(2,im.width//170))
        im.thumbnail((334,250),Image.Resampling.LANCZOS)
        x=(i%4)*tile_w+8;y=80+(i//4)*tile_h
        sheet.paste(im,(x+(334-im.width)//2,y))
        draw.text((x,y+257),r['kind'].replace('_',' '),font=font,fill=(20,20,20))
        draw.text((x,y+277),Path(r['image']).stem,font=font,fill=(90,90,90))
    sheet.save(root/'dataset_preview.jpg',quality=94)


def build(source,asset_dir,output):
    rng=random.Random(SEED)
    output.mkdir(parents=True,exist_ok=True)
    for split in ['train','val']:
        for folder in ['images','labels']:(output/folder/split).mkdir(parents=True,exist_ok=True)
    mask_dir=output/'qa'/'visible_masks';mask_dir.mkdir(parents=True,exist_ok=True)
    records=[];original=[]
    source_rows=json.loads((source/'manifest.json').read_text())
    for r in source_rows:
        name=r['export_name'];split=r['split']
        img=source/'images'/(name+'.jpg');label=source/'labels'/(name+'.txt')
        target_img=output/'images'/split/img.name;target_lab=output/'labels'/split/label.name
        shutil.copy2(img,target_img);shutil.copy2(label,target_lab)
        w,h=Image.open(img).size
        original.append({'name':name,'split':split,'positive':bool(label.read_text().strip())})
        records.append({'image':str(target_img.relative_to(output)),'label':str(target_lab.relative_to(output)),
            'split':split,'kind':'original_real','source_name':name,'source_group':r.get('source_group',r.get('image',r.get('source_image','unknown'))),
            'image_sha256':sha(target_img),'label_sha256':sha(target_lab),
            'objects':parse_labels(label,w,h),'original_provenance':r})
    train=[r for r in original if r['split']=='train']
    positives=[r for r in train if r['positive']];negatives=[r for r in train if not r['positive']]
    assert len(train)==139 and len(positives)==64 and len(negatives)==75
    bg_paths=sorted((asset_dir/'backgrounds').glob('*.png'))
    assert len(bg_paths)==8
    polygons=json.loads((asset_dir/'cutout_polygons.json').read_text())
    assets=[{'source':r['source'],'class_id':r['class_id'],'path':asset_dir/'cutouts'/(r['source']+'.png')} for r in polygons]
    assert all(a['source'] in {r['name'] for r in train} for a in assets)
    schedule=[('real_positive_aug',240),('real_negative_aug',211),('small_block',120),
              ('single_block',120),('multi_block',100),('occlusion',80),('stack',180),('synthetic_negative',200)]
    generated=0
    for kind,count in schedule:
        for index in range(count):
            name=f'{kind}_{index+1:04d}'
            record={'split':'train','kind':kind,'seed':SEED,'index':index}
            masks=[]
            if kind.startswith('real_'):
                pool=positives if kind=='real_positive_aug' else negatives
                r=pool[index%len(pool)]
                im=Image.open(source/'images'/(r['name']+'.jpg')).convert('RGB')
                objects=parse_labels(source/'labels'/(r['name']+'.txt'),im.width,im.height)
                im,objects,params=affine_variant(im,objects,rng)
                record.update(source_name=r['name'],source_split='train',transform=params)
            else:
                # Randomize independently of class/index to avoid background-colour shortcuts.
                bg_path=bg_paths[rng.randrange(len(bg_paths))]
                bg,crop=load_background(bg_path,rng)
                record.update(background=bg_path.name,background_crop=crop,background_origin='image_gen')
                if kind=='synthetic_negative':
                    # Reviewed target-free backgrounds; this path never pastes any block.
                    im,params=photometric(bg,rng);objects=[]
                else:
                    im,objects,masks,params=composite_sample(kind,index,bg,assets,rng)
                record['photometric']=params
                record['annotation_policy']='visible_alpha_mask_envelope' if objects else 'empty_no_target_blocks'
            image_path=output/'images'/'train'/(name+'.jpg')
            label_path=output/'labels'/'train'/(name+'.txt')
            quality=rng.randint(76,94)
            im.save(image_path,quality=quality,subsampling=0)
            write_labels(label_path,objects,im.width,im.height)
            record.update(image=str(image_path.relative_to(output)),label=str(label_path.relative_to(output)),
                          jpeg_quality=quality,objects=objects,image_sha256=sha(image_path),label_sha256=sha(label_path))
            if masks:
                # Every composite has an independent saved instance-index raster for geometry QA.
                instance=np.zeros((im.height,im.width),dtype=np.uint8)
                for i,m in enumerate(masks):instance[m]=i+1
                mask_path=mask_dir/(name+'.png');Image.fromarray(instance).save(mask_path)
                record['instance_mask']=str(mask_path.relative_to(output))
            records.append(record);generated+=1
            if generated%100==0:print(f'Created {generated}/1251 extra images',flush=True)
    assert generated==1251
    train_records=[r for r in records if r['split']=='train']
    val_records=[r for r in records if r['split']=='val']
    def image_list(rs):return ''.join('./'+r['image']+'\n' for r in rs)
    real=[r for r in train_records if r['kind']=='original_real']
    conventional=[r for r in train_records if r['kind']=='original_real' or r['kind'].startswith('real_')]
    (output/'train.txt').write_text(image_list(train_records))
    (output/'val.txt').write_text(image_list(val_records))
    (output/'real_train.txt').write_text(image_list(real))
    (output/'conventional_train.txt').write_text(image_list(conventional))
    # Repeat paths, not image files: exactly 50% original examples in an epoch.
    extras=[r for r in train_records if r['kind']!='original_real']
    balanced=real*9+extras;rng.shuffle(balanced)
    (output/'balanced_train.txt').write_text(image_list(balanced))
    for name,train_path in [('data.yaml','images/train'),('data_balanced.yaml','balanced_train.txt'),
                            ('data_real.yaml','real_train.txt'),('data_conventional.yaml','conventional_train.txt')]:
        (output/name).write_text(yaml.safe_dump({'train':train_path,'val':'images/val','names':{0:'blue_box',1:'red_box'}},sort_keys=False))
    (output/'manifest.json').write_text(json.dumps(records,indent=2))
    summary={'training_images':len(train_records),'validation_images':len(val_records),
             'original_training_images':139,'extra_training_images':1251,'expansion_factor':10,
             'generated_background_assets':8,'real_block_cutout_assets':8,
             'training_category_counts':dict(Counter(r['kind'] for r in train_records)),
             'training_positive_images':sum(bool(r['objects']) for r in train_records),
             'training_background_images':sum(not bool(r['objects']) for r in train_records),
             'training_instances_per_class':dict(Counter(CLASSES[o['class_id']] for r in train_records for o in r['objects'])),
             'validation_unchanged':True,'balanced_training_samples':len(balanced),
             'balanced_original_sample_fraction':.5,'new_real_test_set_available':False,
             'model_retrained':False,'accuracy_improvement_measured':False,'seed':SEED}
    (output/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
    generate_preview(output,records)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True,help='Extracted original yolo11-boxes-v4-dataset directory')
    parser.add_argument('--assets',type=Path,default=Path(__file__).parent/'assets')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.assets,args.output)
