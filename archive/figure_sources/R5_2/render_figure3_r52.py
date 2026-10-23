"""Plot preserved interval and run-level aggregates. No inference or statistical estimation.
Each panel is a separate Matplotlib figure; PIL combines their raster exports.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'Submission_Files'/'Figures'
OUT.mkdir(parents=True,exist_ok=True)
ORDER=[('SIIM','B2'),('SIIM','B6'),('Derm7pt','B2'),('Derm7pt','B6')]
YPOS={key:3-i for i,key in enumerate(ORDER)}
for panel,title,xlabel,interval_label in [
 ('A','A  Clinical-field availability','AP difference: available − structured-missing input','Mean; original 97.5% interval'),
 ('B','B  Clinical-record correspondence','AP difference: matched − reassigned records','Mean; SIIM 97.5%, Derm7pt 95% interval')]:
 pts=pd.read_csv(ROOT/f'Figure_3{panel}_per_run.csv')
 ints=pd.read_csv(ROOT/f'Figure_3{panel}_intervals.csv')
 fig=plt.figure(figsize=(7.2,3.70));ax=fig.add_subplot(111)
 for seed,marker,offset in [(42,'o',-.15),(2024,'s',0),(3407,'^',.15)]:
  p=pts[pts.seed==seed]
  ax.plot(p.difference,[YPOS[(z.cohort,z.model)]+offset for z in p.itertuples()],
          marker=marker,linestyle='none',markersize=5.0,label=f'Seed {seed}')
 xs=[];ys=[];los=[];his=[]
 for key in ORDER:
  r=ints[(ints.cohort==key[0])&(ints.model==key[1])].iloc[0]
  xs.append(r['mean']);ys.append(YPOS[key]);los.append(r['mean']-r.lower);his.append(r.upper-r['mean'])
 ax.errorbar(xs,ys,xerr=[los,his],fmt='D',capsize=3.5,markersize=6,linewidth=1.3,label=interval_label)
 ax.axvline(0,linewidth=.8,linestyle='--',alpha=.55)
 ax.set_title(title,loc='left',fontsize=12,pad=9,fontweight='bold')
 ax.set_yticks([3,2,1,0],['SIIM · B2','SIIM · B6','Derm7pt · B2','Derm7pt · B6'],fontsize=9.5)
 ax.set_xlim(-.012,.019);ax.set_ylim(-.6,3.6)
 ax.set_xticks([-.010,-.005,0,.005,.010,.015])
 ax.tick_params(axis='x',labelsize=9)
 ax.set_xlabel(xlabel,fontsize=9.5,labelpad=6)
 ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.26),ncol=2,frameon=False,fontsize=8.2,columnspacing=1.7)
 fig.subplots_adjust(left=.16,right=.975,top=.86,bottom=.30)
 for ext in ['png','pdf','svg']:
  fig.savefig(ROOT/f'Figure_3{panel}_R5_2.{ext}',dpi=300)
 plt.close(fig)
with Image.open(ROOT/'Figure_3A_R5_2.png') as a, Image.open(ROOT/'Figure_3B_R5_2.png') as b:
 a=a.convert('RGB');b=b.convert('RGB')
 canvas=Image.new('RGB',(max(a.width,b.width),a.height+b.height),(255,255,255))
 canvas.paste(a,(0,0));canvas.paste(b,(0,a.height))
 canvas.save(OUT/'Figure_3_R5_2.png',dpi=(300,300))
 canvas.save(OUT/'Figure_3_R5_2.tif',compression='tiff_lzw',dpi=(300,300))
 canvas.save(OUT/'Figure_3_R5_2.pdf','PDF',resolution=300.0)
print('Figure 3A/B prepared from existing coordinates and real seed differences; no new performance or interval calculations.')
