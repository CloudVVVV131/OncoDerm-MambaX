"""Render existing aggregate results; no model inference or statistical estimation."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'Figures';OUT.mkdir(exist_ok=True)

def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.15)
    with Image.open(OUT/f'{name}.png') as im:
        im.convert('RGB').save(OUT/f'{name}.tif',compression='tiff_lzw',dpi=(300,300))
    plt.close(fig)

points=pd.read_csv(ROOT/'Figure_3_per_run.csv');ci=pd.read_csv(ROOT/'Figure_3_intervals.csv')
fig,ax=plt.subplots(figsize=(7.2,4.35))
order=[('SIIM','B2'),('SIIM','B6'),('Derm7pt','B2'),('Derm7pt','B6')]
ypos={k:3-i for i,k in enumerate(order)}
for seed,marker,offset in [(42,'o',-.15),(2024,'s',0),(3407,'^',.15)]:
    p=points[points.seed==seed]
    ax.plot(p.difference,[ypos[(r.cohort,r.model)]+offset for r in p.itertuples()],linestyle='none',marker=marker,markersize=5.5,label=f'Seed {seed}')
x=[];y=[];lo=[];hi=[]
for key in order:
    z=ci[(ci.cohort==key[0])&(ci.model==key[1])].iloc[0]
    x.append(z['mean']);y.append(ypos[key]);lo.append(z['mean']-z.lower);hi.append(z.upper-z['mean'])
ax.errorbar(x,y,xerr=[lo,hi],fmt='D',capsize=4,markersize=7,linewidth=1.5,label='Mean and 97.5% interval')
ax.axvline(0,linewidth=.9,linestyle='--',alpha=.5)
ax.set_yticks(range(4),['Derm7pt · B6','Derm7pt · B2','SIIM · B6','SIIM · B2'])
ax.set_ylim(-.65,3.75);ax.set_xlabel('AP difference: available clinical fields − structured missing input')
ax.set_xlim(-.012,.019);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,fontsize=9)
fig.tight_layout();save(fig,'Figure_3_R5_1')

op=pd.read_csv(ROOT/'Figure_4_operating_points.csv')
fig,ax=plt.subplots(figsize=(7.2,5.0))
for r in op.itertuples():
    ax.errorbar(r.specificity,r.sensitivity,xerr=r.specificity_sd,yerr=r.sensitivity_sd,
                fmt='o' if r.policy==90 else '^',markersize=6,capsize=3,linewidth=1.1,
                label=f'{r.cohort} · {r.model} · {r.policy}% target')
ax.set_xlabel('External specificity');ax.set_ylabel('External melanoma sensitivity')
ax.set_xlim(.36,1);ax.set_ylim(.26,1);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=8.7)
fig.tight_layout();save(fig,'Figure_4_R5_1')
print('Prepared Figure 3 from actual seed differences and saved intervals; Figure 4 from saved means and sample SDs.')
