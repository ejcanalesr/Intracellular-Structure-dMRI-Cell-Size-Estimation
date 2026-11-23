"""Optional original SpinDoctor FEM route. Not used by saved-data fitting/plots.

Set SPINDOCTOR_ROOT to the pinned checkout in README.md. MATLAB and the
SpinDoctor Linux TetGen binary are required. Execute one FEM case at a time.
The relocated MATLAB route has not been rerun during code consolidation.
"""
from pathlib import Path
import os
import json
import hashlib
import shutil
import subprocess
import numpy as np
from scipy.spatial import ConvexHull
from scipy.io import savemat,loadmat
import signals

def fibonacci(n):
    k=np.arange(n);z=1-2*(k+.5)/n;phi=k*np.pi*(3-np.sqrt(5))
    return np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]


def plc(layout,mesh_params,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);objects=layout['organelles'];surfaces=[]
    for radius,n,c,A,tag in [(layout['Rn_um'],mesh_params['nuclear_points'],np.zeros(3),None,1),
                            (layout['Rc_um'],mesh_params['outer_points'],np.zeros(3),None,2)]+[
            (1.,mesh_params['organelle_points'],np.array(o['center_um']),np.array(o['A_um']),o['index']+2) for o in objects]:
        xyz=fibonacci(n);xyz=xyz*radius+c if A is None else xyz@A.T+c
        surfaces.append((xyz,ConvexHull(xyz).simplices,tag))
    vertices=[];facets=[];tags=[];offset=0
    for xyz,f,tag in surfaces:
        vertices.append(xyz);facets.append(f+offset+1);tags.extend([tag]*len(f));offset+=len(xyz)
    seed=np.array([0.,0.,-(layout['Rc_um']+layout['Rn_um'])/2])
    rng=np.random.default_rng(97812)
    while any(np.linalg.norm(np.linalg.solve(np.array(o['A_um']),seed-np.array(o['center_um'])))<1.1 for o in objects):
        seed=rng.normal(size=3);seed*=(layout['Rc_um']+layout['Rn_um'])/2/np.linalg.norm(seed)
    regions=np.array([np.zeros(3),seed]+[o['center_um'] for o in objects])
    savemat(folder/'surfaces.mat',dict(points=np.concatenate(vertices).T,facets=np.concatenate(facets).T,facetmarkers=np.array(tags),regions=regions.T))

def read_directional(path):
    """Last retained spectral budget from the original MAT-v7 signal output."""
    d=loadmat(path)
    a=d['signals']
    a=a.ravel()[-1] if a.dtype==object else a[-1]
    return np.asarray(a),d


def spherical_mean(signal,job):
    if 'quadrature' in job:
        # The manuscript primary signal used GL12, not a direction-count mean.
        rule=[q for q in job['quadrature'] if q['order']==12]
        return signal[:,[q['direction_index'] for q in rule]]@np.array([q['weight'] for q in rule])
    rule=next(q for q in job['rules'] if q['degree']==11 and not q['rotated'])
    return signal[:,rule['indices']]@np.asarray(rule['weights'])


def regenerate_signal(descriptor,folder,matlab='matlab'):
    """Regenerate ONE saved case, writing new outputs only to folder.

    descriptor is data['fem_jobs'][case] from fitting.load_experiment().
    Returned array contains the DW spherical-mean signal, normalized by S0.
    """
    desc=descriptor;work=Path(folder).resolve();c=desc['condition']
    if desc['solver']=='analytical_M1':
        return signals.m1_signal(c['Rc'],c['Rn'],c['Dc'],c['Dc'],c['kappa_n_um_s'],
                                 signals.published_protocol(c['protocol'])['protocol'])
    spin=Path(os.environ.get('SPINDOCTOR_ROOT',''))
    if not (spin/'src').is_dir():raise RuntimeError('Set SPINDOCTOR_ROOT; see README.md')
    if shutil.which(matlab) is None:raise RuntimeError('MATLAB executable not found')
    for sub in ['configs/jobs','cache','data/forward','meshes','logs']:(work/sub).mkdir(parents=True,exist_ok=True)
    campaign='M2' if desc['solver']=='run08' else 'M3'
    mesh_name='mesh' if campaign=='M2' else desc['mesh_name']
    for original in desc['jobs']:
        job=dict(original);job['mesh_file']=str(work/'meshes'/mesh_name/'spindoctor_mesh.mat')
        job['source_identity']=original.get('identity')
        job['identity']=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
        (work/'configs/jobs'/f'{job["tag"]}.json').write_text(json.dumps(job,indent=2)+'\n')
    if campaign=='M2':
        (work/'configs/geometry.json').write_text(json.dumps(desc['geometry'],indent=2)+'\n')
        mesh_command="signals_fem('mesh_M2','');"
    else:
        plc(desc['layout'],desc['geometry'],work/'meshes'/mesh_name)
        (work/'configs'/f'{mesh_name}_mesh.json').write_text(json.dumps(desc['geometry'],indent=2)+'\n')
        mesh_command=f"signals_fem('mesh_M3','{mesh_name}');"
    j=desc['jobs'][0];tag=j['tag'];solver=desc['solver']
    commands=mesh_command+f"signals_fem('{solver}','{tag}');"
    if desc.get('local_completion'):commands+=f"signals_fem('local','{j['level']}|{j['N']}');"
    matlab_folder=str(Path(__file__).resolve().parent/'fem').replace("'","''")
    commands=f"addpath('{matlab_folder}');"+commands
    subprocess.run([matlab,'-batch',commands],env=dict(os.environ,INSIDE_SPHERE_FEM_WORK=str(work)),check=True)
    rawpath=work/('data/forward' if solver in ['run08','run_coupled'] else 'data')/f'{tag}.mat'
    directional,raw=read_directional(rawpath)
    if desc.get('local_completion'):
        local=loadmat(work/'data'/f'organelle_local_N{j["N"]}_{j["level"]}.mat')
        own=np.real(local['local_signals']).reshape(j['N'],3,len(j['b_s_mm2'])+1,len(j['directions']))
        weights=raw['volumes'].ravel();weights=weights/weights.sum()
        components=np.real(raw['components'].ravel()[-1])
        local_total=np.einsum('i,ibsd->bsd',weights[2:],own)
        directional=np.sum(components[:2],axis=0)+local_total[-1]
    return np.real(spherical_mean(directional,j))[1:]
