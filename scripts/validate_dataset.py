#!/usr/bin/env python3
"""Validate actual image bytes, YOLO labels, instance masks and source isolation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels(path,w,h):
    rows=[]
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        assert len(fields)==5,(path,line)
        c,x,y,bw,bh=map(float,fields)
        assert c in [0,1] and c==int(c),(path,c)
        assert np.isfinite([x,y,bw,bh]).all()
        assert 0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1
        box=np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        assert box[0]>=-.02 and box[1]>=-.02 and box[2]<=w+.02 and box[3]<=h+.02,(path,box)
        rows.append((int(c),box))
    return rows


def validate(root,source=None):
    records=json.loads((root/'manifest.json').read_text())
    original_train={r['source_name'] for r in records if r['split']=='train' and r['kind']=='original_real'}
    hashes={'train':set(),'val':set()};mask_checked=0;val_checked=0;sizes=[]
    counts=Counter();class_counts=Counter();errors=[];background_classes={};background_negatives=Counter()
    for r in records:
        try:
            p=root/r['image'];lp=root/r['label']
            with Image.open(p) as im:
                im.load();w,h=im.size
                assert im.mode=='RGB',p
            assert sha(p)==r['image_sha256']
            assert sha(lp)==r['label_sha256']
            hashes[r['split']].add(sha(p))
            labs=labels(lp,w,h)
            assert len(labs)==len(r['objects'])
            counts[(r['split'],r['kind'])]+=1
            for (c,box),obj in zip(labs,r['objects']):
                assert c==obj['class_id']
                assert np.max(np.abs(box-np.array(obj['xyxy'])))<.02
                if r['split']=='train':class_counts[c]+=1
                if r.get('background'):background_classes.setdefault(r['background'],set()).add(c)
                if r['kind']=='small_block':sizes.append(float(min(box[2]-box[0],box[3]-box[1])*320/max(w,h)))
            if r['kind'].startswith('real_'):assert r['source_name'] in original_train
            if r['kind']=='synthetic_negative':background_negatives[r['background']]+=1
            if r.get('instance_mask'):
                mask=np.array(Image.open(root/r['instance_mask']))
                assert mask.shape==(h,w)
                assert set(np.unique(mask))==set(range(len(labs)+1))
                for idx,((c,box),obj) in enumerate(zip(labs,r['objects']),1):
                    yy,xx=np.where(mask==idx)
                    assert len(xx)>0
                    actual=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
                    assert np.max(np.abs(actual-box))<.02,(p,idx,actual,box)
                    assert obj['cutout_source'] in original_train
                    assert obj['visible_fraction']>=.35
                mask_checked+=1
            if r['split']=='val':
                assert r['kind']=='original_real'
                if source:
                    assert sha(p)==sha(source/'images'/p.name)
                    assert sha(lp)==sha(source/'labels'/lp.name)
                val_checked+=1
        except Exception as exc:
            errors.append({'image':r['image'],'error':str(exc)})
    assert len(records)==1413
    assert len([r for r in records if r['split']=='train'])==1390
    assert val_checked==23
    assert mask_checked==600
    assert not hashes['train']&hashes['val'],'Exact image leakage between splits'
    assert len(background_classes)==8 and all(v=={0,1} for v in background_classes.values())
    assert len(background_negatives)==8
    train_images=set((root/'images'/'train').glob('*.jpg'))
    val_images=set((root/'images'/'val').glob('*.jpg'))
    assert len(train_images)==1390 and len(val_images)==23
    for split in ['train','val']:
        image_stems={p.stem for p in (root/'images'/split).glob('*.jpg')}
        label_stems={p.stem for p in (root/'labels'/split).glob('*.txt')}
        assert image_stems==label_stems
    balanced=(root/'balanced_train.txt').read_text().splitlines()
    assert len(balanced)==2502
    assert sum(Path(s).stem in original_train for s in balanced)==1251
    for f in ['train.txt','val.txt','real_train.txt','conventional_train.txt','balanced_train.txt']:
        assert all((root/line).is_file() for line in (root/f).read_text().splitlines())
    report={'passed':not errors,'image_label_pairs_checked':len(records),
            'composite_instance_masks_checked':mask_checked,'original_validation_pairs_checked':val_checked,
            'byte_identical_validation_vs_source':bool(source),'exact_train_validation_image_overlap':0,
            'training_instances_per_class':{'blue_box':class_counts[0],'red_box':class_counts[1]},
            'small_block_min_side_at_320px_input':{'min':min(sizes),'max':max(sizes),'median':float(np.median(sizes))},
            'all_composite_sources_training_only':True,'balanced_sampling_original_fraction':.5,
            'every_generated_background_has_both_target_classes_and_negative_examples':True,
            'errors':errors,'model_accuracy_test_performed':False,
            'visual_review':'All background assets and eight cutouts reviewed. Stratified composite contact sheet reviewed; not every image manually reviewed.'}
    (root/'qa_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--source',type=Path)
    a=p.parse_args();validate(a.dataset,a.source)
