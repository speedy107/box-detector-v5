import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('--selected-epoch',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[{k.strip():v for k,v in row.items()} for row in csv.DictReader(a.csv.open())]
x=[float(r['epoch']) for r in rows];fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,keys,title in [(axes[0,0],['train/box_loss','val/box_loss'],'Box loss'),(axes[0,1],['train/cls_loss','val/cls_loss'],'Class loss'),(axes[1,0],['metrics/mAP50(B)','metrics/mAP50-95(B)'],'Validation mAP'),(axes[1,1],['metrics/precision(B)','metrics/recall(B)'],'Validation precision and recall')]:
 for k in keys:ax.plot(x,[float(r[k]) for r in rows],marker='o',label=k)
 ax.axvline(a.selected_epoch,color='#555',linestyle='--',label='Selected epoch');ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('V5 training — 23-image validation; eight labelled blocks',fontsize=13);fig.savefig(a.output,dpi=160);plt.close(fig)
