"""M0 manuscript figures, drawn from numerical results (never image crops)."""
from pathlib import Path
import sys
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
REPETITIONS=50

# Editable manuscript plot settings.
EXPERIMENT={'protocols': ['P1', 'P2'], 'repetitions': 50, 'fixed_radii': [3, 5, 7, 9, 11, 13, 15], 'joint_radii': [3, 5, 7], 'diffusivities': [2.0, 0.6], 'R_bounds': [2.0, 20.0], 'D_bounds': [0.3, 3.0]}
P={'protocol_labels': {'P1': 'P1: SANDI Connectome 2.0', 'P2': 'P2: POLAR-CEXI'}, 'radius_limits': [2, 16], 'radius_ticks': [2, 4, 6, 8, 10, 12, 14, 16], 'boxplot_offset_um': 0.08, 'boxplot_width_um': 0.32, 'figure1_noisy_diffusivities': [2.0, 0.6], 'joint_radius_halfwidth_um': 2.0, 'joint_full_range_supplement': 'figureS3', 'snr_display_order': [20, 50, 100, 0], 'table1_bold_criterion': 'Minimum RMSE at finite SNR; minimum absolute bias at infinite SNR; ties at two displayed decimal places are both bold'}

PROTOCOL=PS.PROTOCOL_LABELS
MODEL=PS.MODEL_LABELS
COLOR=PS.MODEL_COLORS

def protocol_key():return '; '.join(PROTOCOL[p] for p in EXPERIMENT['protocols'])
def save(fig,name):PS.save_pdf(fig,FIGURES/name);plt.close(fig)
def radius_boxes(ax,frame,name,D,snr,xlabel=True):
    c=EXPERIMENT;n=c.get('fit_repetitions',c['repetitions']);offset=P['boxplot_offset_um']
    for model,shift in [('M0-Strang',-offset),('M0-GPA',offset)]:
        data=[]
        for R in c['fixed_radii']:
            g=frame[(frame.protocol==name)&(frame.D==D)&(frame.snr==snr)&(frame.R==R)&(frame.model==model)&(frame.realization<n)]
            if len(g)!=n:raise ValueError('Incomplete radius boxplot population')
            data.append(g.Rhat.to_numpy())
        box=ax.boxplot(data,positions=np.asarray(c['fixed_radii'])+shift,widths=P['boxplot_width_um'],patch_artist=True,manage_ticks=False,
            showfliers=True,medianprops=dict(color=COLOR[model],lw=1.15),
            flierprops=dict(marker='.',markersize=2.4,markeredgecolor=COLOR[model],alpha=.65))
        for patch in box['boxes']:patch.set(facecolor=COLOR[model],alpha=.20,edgecolor=COLOR[model])
        for key in ['whiskers','caps']:
            for line in box[key]:line.set(color=COLOR[model],lw=.7)
    ax.plot(P['radius_limits'],P['radius_limits'],color='.35',ls=':',lw=1,zorder=0)
    ax.set(xlim=P['radius_limits'],ylim=P['radius_limits'],xticks=P['radius_ticks'],yticks=P['radius_ticks'],
           xlabel=r'True $R$ ($\mu$m)' if xlabel else '',ylabel=r'Estimated $R$ ($\mu$m)')
    ax.grid(alpha=.13)


def figure1(fixed,summary,errors):
    c=EXPERIMENT;fig,axes=plt.subplots(4,2,figsize=(7.6,9.8))
    positive=errors.max_abs_error[errors.max_abs_error>0]
    error_range=(10**np.floor(np.log10(positive.min())),10**np.ceil(np.log10(positive.max())))
    for col,name in enumerate(c['protocols']):
        axes[0,col].set_title(PROTOCOL[name],fontweight='bold',fontsize=9.3,pad=10)
        for model in COLOR:
            for D in c['diffusivities']:
                g=errors[(errors.protocol==name)&(errors.model==model)&(errors.D==D)].sort_values('R')
                axes[0,col].plot(g.R,g.max_abs_error,ls='-' if D==2 else '--',marker='o' if D==2 else 's',color=COLOR[model],ms=3.2,lw=1.15)
        if (errors.max_abs_error==0).any():axes[0,col].set_yscale('symlog',linthresh=1e-15)
        else:axes[0,col].set_yscale('log');axes[0,col].set_ylim(error_range)
        axes[0,col].set(xlim=P['radius_limits'],xticks=P['radius_ticks'],ylabel=r'Max. signal error ($S_0$)');axes[0,col].grid(axis='y',alpha=.16)
        for model in ['M0-Strang','M0-GPA']:
            for D in c['diffusivities']:
                g=summary[(summary.protocol==name)&(summary.model==model)&(summary.D==D)&(summary.snr==0)].sort_values('R')
                axes[1,col].plot(g.R,g.mean_R,ls='-' if D==2 else '--',marker='o' if D==2 else 's',color=COLOR[model],ms=3.3,lw=1.15)
        axes[1,col].plot(P['radius_limits'],P['radius_limits'],color='.35',ls=':',lw=1)
        axes[1,col].set(xlim=P['radius_limits'],ylim=P['radius_limits'],xticks=P['radius_ticks'],yticks=P['radius_ticks'],ylabel=r'Estimated $R$ ($\mu$m)');axes[1,col].grid(alpha=.13)
        axes[1,col].text(.03,.94,'Noiseless: both D',transform=axes[1,col].transAxes,va='top',fontsize=8)
        for row,D in enumerate(P['figure1_noisy_diffusivities'],2):
            radius_boxes(axes[row,col],fixed,name,D,50,xlabel=row==3)
            axes[row,col].text(.03,.94,rf'SNR 50: $D={D:g}$',transform=axes[row,col].transAxes,va='top',fontsize=8)
        for row in range(4):PS.panel_label(axes[row,col],2*row+col)
    # Matplotlib fills legend columns in order: three model rows left, two D
    # rows right, exactly matching the requested three-row/two-column layout.
    handles=[Line2D([],[],color=COLOR[m],lw=2,label=MODEL[m]) for m in COLOR]
    handles += [Line2D([],[],color='.3',ls='-',marker='o',ms=4,label=r'$D=2$ $\mu$m$^2$/ms'),Line2D([],[],color='.3',ls='--',marker='s',ms=4,label=r'$D=0.6$ $\mu$m$^2$/ms')]
    fig.legend(handles=handles,ncol=2,loc='upper center',frameon=False,fontsize=8.3,bbox_to_anchor=(.5,1),columnspacing=2.5)
    fig.subplots_adjust(left=.10,right=.985,bottom=.055,top=.885,hspace=.24,wspace=.25)
    save(fig,'Fig04_M0_FixedDiffusivity')


def joint_scatter(data,full_range=False):
    c=EXPERIMENT;n=c['repetitions'];halfwidth=P['joint_radius_halfwidth_um'];fig,axes=plt.subplots(2,3,figsize=(8.,5.8));means=[];axis_rows=[]
    colors={'P1':'#2878A2','P2':'#D57824'};markers={'P1':'o','P2':'^'};xlimits={};ylimits={}
    for R in c['joint_radii']:
        for D in c['diffusivities']:
            lo,hi=R-halfwidth,R+halfwidth
            g=data[data.R==R] if full_range else data[(data.R==R)&(data.D==D)]
            groups=['protocol','D'] if full_range else ['protocol']
            minimum=float(g.Rhat.min());maximum=float(g.Rhat.max())
            for _,part in g.groupby(groups):
                minimum=min(minimum,part.Rhat.mean()-part.Rhat.std(ddof=1));maximum=max(maximum,part.Rhat.mean()+part.Rhat.std(ddof=1))
            if minimum<lo:lo=minimum-.15
            if maximum>hi:hi=maximum+.15
            xlimits[R,D]=(lo,hi)
    for D in c['diffusivities']:
        g=data[data.D==D];lo=min(g.Dhat.min(),D-.3);hi=max(g.Dhat.max(),D+.3)
        for _,part in g.groupby(['protocol','R']):lo=min(lo,part.Dhat.mean()-part.Dhat.std(ddof=1));hi=max(hi,part.Dhat.mean()+part.Dhat.std(ddof=1))
        ylimits[D]=(min(0.,lo-.05) if lo<0 else max(0.,lo-.05),hi+.05)
    for row,D in enumerate(c['diffusivities']):
        for col,R in enumerate(c['joint_radii']):
            ax=axes[row,col];lo,hi=xlimits[R,D]
            ax.axvline(R,color='.40',ls=':',lw=.8,zorder=0);ax.axhline(D,color='.40',ls=':',lw=.8,zorder=0)
            plotted=0;initial_outside=0
            for protocol in c['protocols']:
                g=data[(data.protocol==protocol)&(data.R==R)&(data.D==D)]
                if len(g)!=n:raise ValueError('Incomplete joint scatter')
                points=ax.scatter(g.Rhat,g.Dhat,s=15,alpha=.30,color=colors[protocol],marker=markers[protocol],edgecolors='none',zorder=2)
                np.testing.assert_allclose(points.get_offsets(),g[['Rhat','Dhat']].to_numpy(),atol=0.,rtol=0.)
                assert g.Rhat.between(lo,hi).all() and g.Dhat.between(*ylimits[D]).all()
                initial_outside+=int(((g.Rhat<R-halfwidth)|(g.Rhat>R+halfwidth)).sum());plotted+=len(g)
                rmean,dmean=g.Rhat.mean(),g.Dhat.mean();rsd,dsd=g.Rhat.std(ddof=1),g.Dhat.std(ddof=1)
                ax.errorbar(rmean,dmean,xerr=rsd,yerr=dsd,fmt='D',ms=5,color=colors[protocol],capsize=3,lw=1.2,zorder=3)
                means.append(dict(protocol=protocol,R=R,D=D,n=n,mean_R=rmean,mean_D=dmean,sd_R=rsd,sd_D=dsd))
            ax.scatter([R],[D],s=95,marker='*',color='black',edgecolor='white',linewidth=.45,zorder=5)
            tick_step=max(1,int(np.ceil((hi-lo)/8)))
            ticks=np.arange(np.ceil(lo/tick_step)*tick_step,np.floor(hi/tick_step)*tick_step+.5*tick_step,tick_step)
            ax.set(xlim=xlimits[R,D],ylim=ylimits[D],xticks=ticks,xlabel=r'Estimated $R$ ($\mu$m)',ylabel=r'Estimated $D$ ($\mu$m$^2$/ms)')
            ax.set_title(rf'True $R={R:g},\ D={D:g}$',fontsize=10);ax.grid(alpha=.13)
            PS.panel_label(ax,row*3+col)
            axis_rows.append(dict(R=R,D=D,initial_left=R-halfwidth,initial_right=R+halfwidth,left=lo,right=hi,bottom=ylimits[D][0],top=ylimits[D][1],plotted_points=plotted,initial_window_outside=initial_outside,all_points_visible=True))
    handles=[Line2D([],[],ls='none',marker=markers[p],color=colors[p],label=PROTOCOL[p]) for p in c['protocols']]
    handles += [Line2D([],[],ls='none',marker='*',ms=9,color='black',label='Ground truth'),Line2D([],[],marker='D',color='.35',ms=4,label='Mean and SD')]
    fig.legend(handles=handles,ncol=2,loc='upper center',frameon=False,fontsize=8.5,columnspacing=2.)
    fig.subplots_adjust(left=.08,right=.985,top=.83,bottom=.10,hspace=.39,wspace=.36)
    save(fig,'Fig05_M0_JointRadiusDiffusivity')
    return 0


def plot_results(fixed=None,joint=None,errors=None):
    if fixed is None:
        file=RESULTS/'fits_fixed_D.csv'
        fixed=pd.read_csv(file if file.exists() else DATA/'fits_fixed_D.csv',float_precision='round_trip')
    if joint is None:
        file=RESULTS/'fits_free_D.csv'
        joint=pd.read_csv(file if file.exists() else DATA/'fits_free_D.csv',float_precision='round_trip')
    if errors is None:errors=pd.read_csv(DATA/'signal_errors.csv',float_precision='round_trip')
    summary=fixed.groupby(['protocol','model','D','R','snr'],as_index=False).agg(mean_R=('Rhat','mean'))
    figure1(fixed,summary,errors)
    joint_scatter(joint[joint.snr==50].copy())


if __name__=='__main__':plot_results()
