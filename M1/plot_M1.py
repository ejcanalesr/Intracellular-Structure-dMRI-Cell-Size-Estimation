"""M1 permeability figure from every saved parameter estimate."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import plot_style as PS
PS.apply_style()
DATA=Path(__file__).resolve().parent/'data'
RESULTS=Path(__file__).resolve().parent/'results'
FIGURES=ROOT/'figures'

KAPPAS=[0.,20.,50.,100.,200.,500.,float('inf')]
METHODS=['Strang_free_D','GPA_fixed_D','GPA_free_D']
LABELS={'Strang_free_D':'M0-Strang','GPA_fixed_D':'M0-GPA','GPA_free_D':'M0-GPA'}

def plot_results(frame=None):
    if frame is None:
        file=RESULTS/'fits.csv'
        frame=pd.read_csv(file if file.exists() else DATA/'fits.csv',float_precision='round_trip')
    noisy=frame[(frame.snr==50)&frame.estimator.isin(METHODS)]
    lo=min(0.,noisy.e_R_pct.min());hi=max(0.,noisy.e_R_pct.max());pad=max(1.,.1*(hi-lo))
    fig,axes=plt.subplots(2,2,figsize=(7.6,6.2))
    for col,protocol in enumerate(['P1','P2']):
        radii=sorted(noisy[noisy.protocol==protocol].Rc.unique())
        for row,R in enumerate(radii):
            ax=axes[row,col];subset=noisy[(noisy.protocol==protocol)&(noisy.Rc==R)]
            methods=['Strang_free_D','GPA_fixed_D' if protocol=='P1' else 'GPA_free_D']
            for i,k in enumerate(KAPPAS):
                for j,method in enumerate(methods):
                    g=subset[(subset.estimator==method)&(subset.kappa_um_s==k)].sort_values('rep')
                    y=g.e_R_pct.to_numpy()
                    if len(y)!=50:raise ValueError('Expected every one of the50 realizations')
                    x=i+(j-.5)*2*PS.BOX_OFFSET;model=LABELS[method];color=PS.MODEL_COLORS[model]
                    ax.scatter(x+np.linspace(-.035,.035,len(y)),y,s=8,c='.4',alpha=.24,edgecolors='none')
                    b=ax.boxplot([y],positions=[x],widths=PS.BOX_WIDTH,patch_artist=True,manage_ticks=False,
                        medianprops=dict(color=color,lw=PS.MEDIAN_LINEWIDTH),
                        flierprops=dict(marker='.',markersize=2.4,markeredgecolor=color,alpha=.65))
                    b['boxes'][0].set(facecolor=color,alpha=PS.BOX_FACE_ALPHA,edgecolor=color)
                    for key in ['whiskers','caps']:
                        for line in b[key]:line.set(color=color,lw=PS.WHISKER_LINEWIDTH)
                    ax.plot(x,y.mean(),'D',ms=4.2,color=color)
            ax.axhline(0,color='.35',ls=':',lw=1);ax.grid(axis='y',alpha=PS.GRID_ALPHA)
            ax.set(xlim=(-.55,len(KAPPAS)-.45),ylim=(lo-pad,hi+pad),xticks=range(len(KAPPAS)),
                xticklabels=[r'$\infty$' if np.isinf(k) else f'{k:g}' for k in KAPPAS],
                xlabel=PS.AXIS_LABELS['permeability'],ylabel=PS.AXIS_LABELS['radius_error_percent'])
            ax.set_title(PS.PROTOCOL_LABELS[protocol],fontweight='bold',pad=21)
            c=subset.iloc[0]
            ax.text(.5,1.015,rf'$R_c={R:g}$ $\mu$m; $D_c={c.Dc:g}$ $\mu$m$^2$/ms; $R_n/R_c={c.eta:g}$',
                ha='center',va='bottom',transform=ax.transAxes,fontsize=8)
            PS.panel_label(ax,2*row+col)
    fig.legend(handles=[Line2D([],[],color=PS.MODEL_COLORS[m],lw=2,label=m) for m in ['M0-Strang','M0-GPA']],
        ncol=2,loc='upper center',frameon=False)
    fig.text(.5,.061,r'$D_n=D_c$; boxes: median, quartiles and 1.5 IQR; diamonds: means.',ha='center',fontsize=7.3)
    fig.text(.5,.025,'P1: M0-Strang fits R,D; M0-GPA fixes D=2. P2: both fit R,D. S0 is profiled.',ha='center',fontsize=7.3)
    fig.subplots_adjust(left=.10,right=.985,top=.83,bottom=.18,hspace=.52,wspace=.28)
    PS.save_pdf(fig,FIGURES/'Fig06_M1_NuclearPermeability');plt.close(fig)

if __name__=='__main__':plot_results()
