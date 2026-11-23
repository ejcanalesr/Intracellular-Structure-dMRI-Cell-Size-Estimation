"""M3 manuscript figures, drawn from numerical results (never image crops)."""
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

LABEL={'Strang_free_D':'M0-Strang: free D','Strang_fixed_D':'M0-Strang: fixed D',
       'GPA_fixed_D':'M0-GPA: fixed D','GPA_free_D':'M0-GPA: free D'}
def estimators(c):return ['Strang_free_D','Strang_fixed_D','GPA_fixed_D'] if c['protocol']=='P1' else ['Strang_free_D','GPA_free_D']
def model(e):return 'M0-GPA' if e.startswith('GPA') else 'M0-Strang'
def joint(e):return e.endswith('free_D')
def appearance(e):return dict(color=PS.MODEL_COLORS[model(e)],linestyle='--' if e=='Strang_fixed_D' else '-',marker='s' if e=='Strang_fixed_D' else 'o')


def plot_population(frame,column,stem,preliminary=False):
    fig,axes=plt.subplots(2,2,figsize=(7.6,6.1),squeeze=False);fig.subplots_adjust(left=.105,right=.97,bottom=.10,top=.93,wspace=.33,hspace=.38)
    for j,p in enumerate(['P1','P2']):
        for i,k in enumerate(['0','Inf']):
            ax=axes[i,j];subset=frame[(frame.protocol==p)&((frame.kappa_org_um_s.astype(str)==k)|(frame.N==0))]
            for ie,e in enumerate(estimators({'protocol':p})):
                q=subset[(subset.estimator==e)&(subset.snr==0)].sort_values('N');a=appearance(e)
                ax.plot(100*q.cell_fraction,q[column],label=LABEL[e],ms=3,mfc='white',**a)
                for _,z in q.iterrows():
                    n=subset[(subset.configuration==z.configuration)&(subset.estimator==e)&(subset.snr==50)]
                    if len(n):
                        x=100*z.cell_fraction+(ie-(len(estimators({'protocol':p}))-1)/2)*.018
                        jitter=np.linspace(-.008,.008,len(n));ax.scatter(x+jitter,n[column],s=4,color=a['color'],alpha=.28,zorder=2)
                        ax.errorbar(x,n[column].mean(),yerr=n[column].std(ddof=1),fmt='D',ms=3,color=a['color'],capsize=2,lw=.7,zorder=5)
            ax.axhline(0,color='.45',ls=':',lw=.7);ax.grid(alpha=PS.GRID_ALPHA)
            ax.set(xlabel='Organelle water (% of cell)',ylabel='Total radius error (%)' if column=='e_R_pct' else 'Additional radius error (pp)',title=f"{PS.PROTOCOL_LABELS[p]}; "+('sealed' if k=='0' else 'transparent'))
            PS.panel_label(ax,i*2+j)
            ax.tick_params(labelsize=8);ax.legend(frameon=False,fontsize=6.7)
    if preliminary:fig.suptitle('Stage A: noiseless endpoint fits; finite permeability and noise pending',fontsize=9)
    PS.save_pdf(fig,FIGURES/stem);plt.close(fig)


def permeability(frame):
    fig,axes=plt.subplots(2,2,figsize=(7.6,6.1));fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.94,wspace=.32,hspace=.34)
    for j,p in enumerate(['P1','P2']):
        q=frame[(frame.protocol==p)&(frame.N==(20 if p=='P1' else 60))]
        ks=[k for k in ['0','20','200','Inf'] if k in q.kappa_org_um_s.values]
        for i,col in enumerate(['e_R_pct','D_hat']):
            ax=axes[i,j]
            for ie,e in enumerate(estimators({'protocol':p})):
                if i==1 and not joint(e):continue
                a=appearance(e);z=q[(q.estimator==e)&(q.snr==0)].set_index('kappa_org_um_s').loc[ks]
                if p=='P2':a['linestyle']='none'
                ax.plot(range(len(ks)),z[col],mfc='white',ms=4,label=LABEL[e],**a)
                for x,k in enumerate(ks):
                    n=q[(q.estimator==e)&(q.snr==50)&(q.kappa_org_um_s==k)]
                    offset=(ie-(len(estimators({'protocol':p}))-1)/2)*.06
                    ax.scatter(x+offset+np.linspace(-.025,.025,len(n)),n[col],color=a['color'],alpha=.3,s=5)
                    ax.errorbar(x+offset,n[col].mean(),yerr=n[col].std(ddof=1),fmt='D',ms=3,color=a['color'],capsize=2,lw=.7)
            ax.set_xticks(range(len(ks)),[r'$\infty$' if k=='Inf' else k for k in ks]);ax.set_xlabel(r'$\kappa_{\rm org}$ ($\mu$m/s)')
            ax.set_ylabel('Total radius error (%)' if i==0 else PS.AXIS_LABELS['estimated_D']);ax.axhline(0 if i==0 else (2 if p=='P1' else .6),color='.45',ls=':',lw=.7)
            ax.set_title(PS.PROTOCOL_LABELS[p]);ax.grid(alpha=PS.GRID_ALPHA);ax.legend(frameon=False,fontsize=6.9)
            PS.panel_label(ax,i*2+j)
            if p=='P1' and i==1:ax.legend(frameon=False,fontsize=6.9,loc='upper left',bbox_to_anchor=(.02,.86))
            if p=='P2':ax.text(.5,.48,'Finite permeability not validated',transform=ax.transAxes,ha='center',fontsize=7,color='.35')
    PS.save_pdf(fig,FIGURES/'Fig09_M3_InclusionPermeability');plt.close(fig)


def plot_results(frame=None):
    if frame is None:
        file=RESULTS/'fits.csv'
        frame=pd.read_csv(file if file.exists() else DATA/'fits.csv',dtype={'kappa_org_um_s':str},float_precision='round_trip')
    plot_population(frame,'e_R_pct','Fig08_M3_InclusionVolumeFraction')
    permeability(frame)



# Mesh panels: original geometric intersections, not saved image panels.
import json
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
NUC='#21856C';OUT='#53677A';CYTO='#D5DDE4'
FIRST=['#D57824','#81539D','#BD5360','#3C8FA6','#B39837']
EDGES=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
def colors(N):
    return FIRST[:N] if N<=5 else FIRST+[matplotlib.colors.to_hex(plt.get_cmap('hsv')((k*.61803398875)%1)) for k in range(N-5)]


def contour(ax,c,A,U,origin,color,fill=True):
    c=np.asarray(c)-origin;B=np.linalg.inv(A@A.T);E=U[:,:2];H=E.T@B@E;l=E.T@B@c;m=np.linalg.solve(H,l);rho=1-c@B@c+l@m
    if rho<=0:return
    val,V=np.linalg.eigh(H);theta=np.linspace(0,2*np.pi,200);xy=m[:,None]+V@(np.sqrt(rho/val)[:,None]*np.array([np.cos(theta),np.sin(theta)]))
    if fill:ax.fill(*xy,color=color,alpha=.15)
    ax.plot(*xy,color=color,lw=.9);return m


def contours(ax,layout,U,origin,col,fill=True,labels=False):
    contour(ax,np.zeros(3),layout['Rc_um']*np.eye(3),U,origin,OUT,fill)
    contour(ax,np.zeros(3),layout['Rn_um']*np.eye(3),U,origin,NUC,fill)
    for o,c in zip(layout['organelles'],col):
        m=contour(ax,np.array(o['center_um']),np.array(o['A_um']),U,origin,c,fill)
        if labels and m is not None:ax.text(*m,str(o['index']),ha='center',va='center',fontsize=6,color='black')


def format3d(ax,R):
    ax.set(xlim=(-R,R),ylim=(-R,R),zlim=(-R,R));ax.set_box_aspect((1,1,1));ax.view_init(24,-48)
    for a in ['x','y','z']:
        getattr(ax,'set_'+a+'ticks')([-R,0,R]);getattr(ax,'set_'+a+'label')(a+r' ($\mu$m)',labelpad=-2)
    ax.tick_params(labelsize=6.5,pad=0)


def cut(p,t,r,U,origin,span=None):
    xyz=((p-origin)@U)[t];v=xyz[:,:,2];choose=(v.min(axis=1)<=1e-12)&(v.max(axis=1)>=-1e-12)
    if span is not None:
        choose &= (xyz[:,:,0].min(axis=1)<span)&(xyz[:,:,0].max(axis=1)>-span)&(xyz[:,:,1].min(axis=1)<span)&(xyz[:,:,1].max(axis=1)>-span)
    polygons=[];regions=[]
    for i in np.flatnonzero(choose):
        verts=[xyz[i,j,:2]+v[i,j]/(v[i,j]-v[i,k])*(xyz[i,k,:2]-xyz[i,j,:2]) for j,k in EDGES if v[i,j]*v[i,k]<0]
        verts += [xyz[i,j,:2] for j in np.flatnonzero(np.abs(v[i])<=1e-12)]
        vv=np.unique(np.round(verts,11),axis=0)
        if len(vv)<3:continue
        mid=vv.mean(axis=0);ix=np.argsort(np.arctan2(vv[:,1]-mid[1],vv[:,0]-mid[0]));polygons.append(vv[ix]);regions.append(r[i])
    return polygons,regions


def draw_mesh(ax,mesh,layout,U,origin,col,span=None):
    polygons,regions=cut(mesh['points'],mesh['tetrahedra'],mesh['regions'],U,origin,span)
    cc=[NUC if k==1 else CYTO if k==2 else col[k-3] for k in regions]
    ax.add_collection(PolyCollection(polygons,facecolors=cc,edgecolors='.4',linewidths=.17,alpha=.42))
    contours(ax,layout,U,origin,col,False);ax.set_aspect('equal')
    R=layout['Rc_um']*1.04 if span is None else span;ax.set(xlim=(-R,R),ylim=(-R,R));ax.grid(alpha=.1)


def gap_basis(p0,p1,long):
    v=p1-p0;v/=np.linalg.norm(v);s=long-v*(v@long)
    if np.linalg.norm(s)<1e-8:s=np.cross(v,[1,0,0])
    s/=np.linalg.norm(s);return np.stack([s,v,np.cross(s,v)],axis=1)


def plot_mesh():
    with np.load(DATA/'meshes.npz',allow_pickle=False) as saved:
        fig=plt.figure(figsize=(10.5,10.2))
        gs=fig.add_gridspec(3,3,left=.07,right=.945,bottom=.065,top=.95,wspace=.40,hspace=.37)
        for col,N in enumerate([5,20,60]):
            layout=json.loads(str(saved[f'N{N}_layout_json']));palette=colors(N)
            p=saved[f'N{N}_points'];f=saved[f'N{N}_triangles'];b=saved[f'N{N}_boundaries']
            ax=fig.add_subplot(gs[0,col],projection='3d')
            # Every plotted polygon is a saved boundary triangle; outer cell hidden.
            ax.add_collection3d(Poly3DCollection(p[f[b==1]],facecolors=NUC,edgecolors='none',alpha=.08))
            for o,c in zip(layout['organelles'],palette):
                tri=p[f[b==o['index']+2]]
                ax.add_collection3d(Poly3DCollection(tri,facecolors=c,edgecolors=c,linewidths=.035,alpha=.5))
                ax.text(*o['center_um'],str(o['index']),fontsize=6 if N<=20 else 4.5,color='black',
                    zorder=1000,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.1))
            format3d(ax,layout['Rc_um'])
            ax.set_title(f'N={N}: numbered boundary mesh',fontsize=9,pad=0)
            PS.panel_label(ax,col)
        layout=json.loads(str(saved['N20_layout_json']));palette=colors(20)
        mesh={k:saved['N20_'+k] for k in ['points','tetrahedra','regions']};origin=np.zeros(3)
        planes=[(np.eye(3)[:,[1,2,0]],'x = 0','y','z'),(np.eye(3)[:,[0,2,1]],'y = 0','x','z'),(np.eye(3),'z = 0','x','y')]
        for col,(U,title,x,y) in enumerate(planes):
            ax=fig.add_subplot(gs[1,col]);draw_mesh(ax,mesh,layout,U,origin,palette)
            ax.set(xlabel=x+r' ($\mu$m)',ylabel=y+r' ($\mu$m)',title=f'N=20: {title}')
            PS.panel_label(ax,3+col)
        obj=layout['organelles'];R=layout['Rc_um']
        outer=min(obj,key=lambda o:o['outer_gap_um']);far=np.array(outer['radial']['far']);po=R*far/np.linalg.norm(far)
        inner=min(obj,key=lambda o:o['nuclear_gap_um']);near=np.array(inner['radial']['near']);pn=layout['Rn_um']*near/np.linalg.norm(near)
        pair=json.loads(str(saved['N20_pair_json']))
        gaps=[(far,po,np.array(outer['orientation'])[:,0],'Outer membrane'),
            (pn,near,np.array(inner['orientation'])[:,0],'Nuclear membrane'),
            (np.array(pair['x']),np.array(pair['y']),np.array(obj[pair['i']-1]['orientation'])[:,0],f"Inclusions {pair['i']}/{pair['j']}")]
        for col,(p0,p1,long,title) in enumerate(gaps):
            ax=fig.add_subplot(gs[2,col]);U=gap_basis(p0,p1,long);mid=(p0+p1)/2;g=np.linalg.norm(p1-p0)
            draw_mesh(ax,mesh,layout,U,mid,palette,max(1.2,min(3*g,2.)))
            ax.annotate('',xy=(0,g/2),xytext=(0,-g/2),arrowprops=dict(arrowstyle='<->',color='black',lw=1))
            ax.set(xlabel=r'In-plane $s$ ($\mu$m)',ylabel=r'In-plane $t$ ($\mu$m)',
                title=f'{title}: {g:.4f} '+r'$\mu$m')
            PS.panel_label(ax,6+col)
        fig.suptitle('M3: inclusion populations, N=20 tetrahedral sections and minimum gaps',fontsize=11,y=.985)
        PS.save_pdf(fig,FIGURES/'Fig03_OrganelleInclusionMesh');plt.close(fig)

if __name__=='__main__':
    plot_mesh();plot_results()
