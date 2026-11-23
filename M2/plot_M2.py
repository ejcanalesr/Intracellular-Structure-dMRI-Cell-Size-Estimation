"""M2 manuscript figures, drawn from numerical results (never image crops)."""
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

def estimators(c):return ['Strang_free_D','Strang_fixed_D','GPA_fixed_D'] if c['protocol']=='P1' else ['Strang_free_D','GPA_free_D']
def model(e):return 'M0-GPA' if e.startswith('GPA') else 'M0-Strang'
def settings(e):
    return dict(color=PS.MODEL_COLORS[model(e)],linestyle='--' if e=='Strang_fixed_D' else '-',
        sample_marker='s' if e=='Strang_fixed_D' else '.',filled=e!='Strang_fixed_D')


def render(f,column,stem,ylabel):
    PS.apply_style();fig,axes=plt.subplots(3,2,figsize=(PS.FIGURE_WIDTH_INCHES,7.3),sharex=True)
    fig.subplots_adjust(left=.105,right=.985,bottom=.165,top=.94,hspace=.30,wspace=.27)
    npoints=nzero=0;limits={}
    for j,name in enumerate(PS.PROTOCOL_ORDER):
        extrema=[0.]
        for i,ratio in enumerate([.7,1.,1.3]):
            ax=axes[i,j];g=f[(f.protocol==name)&(f.ratio==ratio)]
            methods=estimators(dict(protocol=name))
            offsets=dict(zip(methods,[-.035,0.,.035] if name=='P1' else [-.02,.02]))
            for e in methods:
                sty=settings(e);part=g[g.estimator==e];means=[];sds=[];xs=[];zero=[]
                for k in range(3):
                    h=part[part.offset_index==k];noisy=h[h.snr==50].sort_values('rep');z=h[h.snr==0]
                    assert len(noisy)==50 and len(z)==1
                    v=noisy[column].to_numpy();x=k/3+offsets[e]
                    jitter=.009*(((noisy.rep.to_numpy()*17)%50)/49-.5)
                    kwargs=dict(color=sty['color']) if sty['filled'] else dict(facecolors='none',edgecolors=sty['color'])
                    ax.scatter(x+jitter,v,s=9,marker=sty['sample_marker'],alpha=.28,linewidths=.45,zorder=2,**kwargs)
                    mu=v.mean();sd=v.std(ddof=1);true=float(z.iloc[0][column])
                    means.append(mu);sds.append(sd);xs.append(x);zero.append(true)
                    extrema.extend([*v,mu-sd,mu+sd,true]);npoints+=len(v);nzero+=1
                ax.errorbar(xs,means,yerr=sds,color=sty['color'],ls=sty['linestyle'],marker=PS.MEAN_MARKER,
                    mfc=sty['color'] if sty['filled'] else 'white',mec=sty['color'],ms=4.2,lw=1.15,elinewidth=.85,capsize=2.5,zorder=4)
                ax.plot(xs,zero,ls='none',marker='o',ms=4.2,mfc='white',mec=sty['color'],mew=1.,zorder=5)
            ax.axhline(0,color='.47',ls=PS.TRUTH_LINESTYLE,lw=.75,zorder=0)
            ax.grid(axis='y',alpha=PS.GRID_ALPHA);ax.set_xlim(-.09,.755)
            ax.set_xticks([0,1/3,2/3],['0','1/3','2/3'])
            panel=rf'$D_n/D_c={ratio:g}$'
            ax.set_title((PS.PROTOCOL_LABELS[name]+'\n' if i==0 else '')+panel,
                fontsize=PS.TITLE_FONT_SIZE,pad=7)
            PS.panel_label(ax,2*i+j)
            if j==0:ax.set_ylabel(ylabel)
            if i==2:ax.set_xlabel(r'Nuclear eccentricity $\alpha=d/(R_c-R_n)$')
        lo,hi=min(extrema),max(extrema);margin=.07*(hi-lo) if hi>lo else .1
        limits[name]=[lo-margin,hi+margin]
        for ax in axes[:,j]:ax.set_ylim(*limits[name])
    orange=PS.MODEL_COLORS['M0-Strang'];purple=PS.MODEL_COLORS['M0-GPA']
    handles=[Line2D([],[],color=orange,lw=1.4,label='M0-Strang: free D'),
        Line2D([],[],color=purple,lw=1.4,label='M0-GPA: fixed D in P1, free D in P2'),
        Line2D([],[],color=orange,lw=1.4,ls='--',label='M0-Strang: fixed D in P1'),
        Line2D([],[],color='.25',marker=PS.MEAN_MARKER,ls='none',ms=4,label='Mean; bars: sample SD (50 fits)'),
        Line2D([],[],color='.25',marker='o',mfc='white',ls='none',ms=4,label='Noiseless fit'),
        Line2D([],[],color='.45',marker='.',ls='none',ms=4,label='All 50 individual realizations')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.014),ncol=2,frameon=False,
        fontsize=7.45,handlelength=2.,columnspacing=1.5,labelspacing=.65)
    PS.save_pdf(fig,FIGURES/stem);plt.close(fig)
    assert npoints==2250 and nzero==45
    return dict(stem=stem,noisy_points=npoints,noiseless_points=nzero,column_y_limits=limits,
        all_points_and_SD_endpoints_visible=True,columns_have_different_scales=True)


def plot_results(frame=None):
    if frame is None:
        file=RESULTS/'fits.csv'
        frame=pd.read_csv(file if file.exists() else DATA/'fits.csv',float_precision='round_trip')
    render(frame,'delta_e_R_pp','Fig07_M2_NuclearDisplacement','Additional radius error (pp)')



# Mesh panels: original tetrahedron/plane intersection and boundary triangles.
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
NUCLEUS='#21856C';CYTO='#C9D2DA';OUTER='#53677A'
def cut_polygons(p,t,r,normal=1):
    xyz=p[t];vals=xyz[:,:,normal]
    selected=np.flatnonzero((vals.min(axis=1)<=1e-12)&(vals.max(axis=1)>=-1e-12))
    dims=[i for i in range(3) if i!=normal]
    edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    polys=[];reg=[]
    for k in selected:
        a=xyz[k];v=vals[k]
        # Keep vertices already on the plane; dropping them creates blank slivers.
        verts=[a[i,dims] for i in np.flatnonzero(np.abs(v)<=1e-12)]
        for i,j in edges:
            if v[i]*v[j]<0:
                point=a[i]+v[i]/(v[i]-v[j])*(a[j]-a[i]);verts.append(point[dims])
        if len(verts)<3:continue
        v=np.unique(np.round(verts,12),axis=0)
        if len(v)<3:continue
        mid=v.mean(axis=0);ix=np.argsort(np.arctan2(v[:,1]-mid[1],v[:,0]-mid[0]))
        polys.append(v[ix]);reg.append(r[k])
    return polys,np.array(reg)


def planar(ax,d,Rc,Rn,horiz='x',mesh=False,zoom=False):
    if not mesh:
        ax.add_patch(Circle((0,0),Rc,fc=CYTO,alpha=.30,ec='none'))
        ax.add_patch(Circle((0,d),Rn,fc=NUCLEUS,alpha=.12,ec='none'))
    ax.add_patch(Circle((0,0),Rc,fc='none',ec=OUTER,lw=1.0,ls='--' if mesh else '-'))
    ax.add_patch(Circle((0,d),Rn,fc='none',ec=NUCLEUS,lw=1.1,ls='--' if mesh else '-'))
    ax.plot(0,0,'o',color='black',ms=3.5,fillstyle='none')
    ax.plot(0,d,'D',color=NUCLEUS,ms=3)
    if not zoom:
        ax.axvline(0,color='.4',lw=.6,ls=':')
        ax.set_xlim(-7.6,7.6);ax.set_ylim(-7.6,7.6)
        ax.set_xticks([-7,-3.5,0,3.5,7]);ax.set_yticks([-7,-3.5,0,3.5,7])
    else:
        ax.set_xlim(-1.7,1.7);ax.set_ylim(d+Rn-.55,Rc+.35)
    xx=.45 if zoom else .55
    ax.annotate('',xy=(xx,Rc),xytext=(xx,d+Rn),arrowprops=dict(arrowstyle='<->',lw=.9,color='black'))
    ax.text(xx+.14,(Rc+d+Rn)/2,rf'$g_{{\min}}={Rc-Rn-d:.3f}$',fontsize=7.3,ha='left',va='center',bbox=dict(fc='white',ec='none',alpha=.82,pad=.6))
    ax.set_aspect('equal');ax.set_xlabel(rf'${horiz}$ ($\mu$m)');ax.set_ylabel(r'$z$ ($\mu$m)')
    ax.grid(alpha=.12)


def mesh_3d(ax,p,f,markers):
    # Genuine volume-mesh boundary triangles, cut away front outer half.
    for tag,col,alpha in [(2,OUTER,.11),(1,NUCLEUS,.24)]:
        tri=p[f[markers==tag]]
        if tag==2:tri=tri[tri.mean(axis=1)[:,1]>=0]
        artist=Poly3DCollection(tri,facecolors=col,edgecolors=col,linewidths=.12,alpha=alpha)
        ax.add_collection3d(artist)


def axes_3d(ax):
    ax.set(xlim=(-7.5,7.5),ylim=(-7.5,7.5),zlim=(-7.5,7.8))
    ax.set_box_aspect((1,1,1));ax.view_init(elev=14,azim=-67)
    ax.set_xlabel(r'$x$ ($\mu$m)',labelpad=-2);ax.set_ylabel(r'$y$ ($\mu$m)',labelpad=-2);ax.set_zlabel(r'$z$ ($\mu$m)',labelpad=-2)
    ax.set_xticks([-7,0,7]);ax.set_yticks([-7,0,7]);ax.set_zticks([-7,0,7]);ax.tick_params(labelsize=7,pad=0)


def plot_mesh():
    Rc=7.;Rn=4.2
    with np.load(DATA/'meshes.npz',allow_pickle=False) as mesh:
        fig=plt.figure(figsize=(7.6,10.1))
        gs=fig.add_gridspec(3,2,left=.09,right=.96,bottom=.07,top=.94,hspace=.36,wspace=.31)
        for col,(prefix,alpha) in enumerate([('a1_',1/3),('a2_',2/3)]):
            p=mesh[prefix+'points'];t=mesh[prefix+'tetrahedra'];r=mesh[prefix+'regions']
            f=mesh[prefix+'triangles'];b=mesh[prefix+'boundaries'];d=alpha*(Rc-Rn)
            ax=fig.add_subplot(gs[0,col],projection='3d',computed_zorder=False);mesh_3d(ax,p,f,b);axes_3d(ax)
            ax.scatter([0],[0],[0],marker='o',color='black',s=12,depthshade=False)
            ax.scatter([0],[0],[d],marker='D',color=NUCLEUS,edgecolor='white',linewidth=.7,s=18,depthshade=False)
            ax.plot([0,0],[0,0],[-Rc,Rc],ls=':',color='.25',lw=.7)
            ax.set_title(rf'$\alpha={alpha:.3f}$; $d={d:.3f}$ $\mu$m',pad=3)
            PS.panel_label(ax,col)
            polygons,regions=cut_polygons(p,t,r)
            for row,zoom in [(1,False),(2,True)]:
                ax=fig.add_subplot(gs[row,col])
                selected=[i for i,v in enumerate(polygons) if not zoom or (v[:,1].max()>d+Rn-.6 and abs(v[:,0]).min()<1.9)]
                colors=[NUCLEUS if regions[i]==1 else CYTO for i in selected]
                ax.add_collection(PolyCollection([polygons[i] for i in selected],facecolors=colors,edgecolors='.35',linewidths=.22,alpha=.40))
                planar(ax,d,Rc,Rn,'x',mesh=True,zoom=zoom)
                title='Minimum-gap detail' if zoom else 'Tetrahedra cut at y = 0'
                ax.set_title(title,fontsize=9)
                PS.panel_label(ax,2*row+col)
        handles=[Line2D([],[],color=OUTER,lw=1.3,label='Outer cell surface'),
            Line2D([],[],color=NUCLEUS,lw=1.5,label='Nuclear surface'),
            Line2D([],[],marker='o',ls='',color='black',label='Cell center'),
            Line2D([],[],marker='D',ls='',color=NUCLEUS,label='Nuclear center')]
        fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.008),fontsize=7.2)
        fig.suptitle(r'M2: displaced nucleus; $R_c=7$, $R_n=4.2$ $\mu$m',fontsize=11,y=.985)
        PS.save_pdf(fig,FIGURES/'Fig02_DisplacedNucleusMesh');plt.close(fig)

if __name__=='__main__':
    plot_mesh();plot_results()
