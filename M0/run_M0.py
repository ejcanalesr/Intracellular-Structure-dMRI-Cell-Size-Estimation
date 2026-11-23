"""M0: saved results, optional same-data refitting, or physical regeneration."""
from pathlib import Path
import os
# One BLAS thread per worker; change N_WORKERS rather than nesting BLAS pools.
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import signals
import fitting
from plot_M0 import plot_results

# EDIT THESE SETTINGS. Default: load accepted results, calculate metrics, plot.
MODE='saved'                 # 'saved', 'refit', or 'regenerate'
N_WORKERS=8                  # Used only for explicit refitting; use1 for serial.
REPETITIONS=50
DATA=Path(__file__).resolve().parent/'data'
RESULTS=Path(__file__).resolve().parent/'results'
REGENERATE_CASES=[0]          # Expensive physical generation is opt-in, case by case.
MATLAB='matlab'               # Used only by M2/M3 physical regeneration.


def refit(data):
    fixed=[];joint=[];screens={}
    # The physical cases and ALL observations come from the definitive NPZ.
    with Pool(N_WORKERS) as pool:
        for i,c in enumerate(data['conditions']):
            protocol=signals.published_protocol(c['protocol'])['protocol']
            estimators=['Strang_fixed_D','GPA_fixed_D']
            if c['R'] in [3,5,7]:estimators+=['Strang_free_D']
            for estimator in estimators:
                method='M0-GPA' if estimator.startswith('GPA') else 'M0-Strang'
                D_fixed=None if estimator.endswith('free_D') else c['D']
                key=(c['protocol'],method,D_fixed)
                if key not in screens:screens[key]=fitting.make_screen(protocol,method,D_fixed)
                tasks=[];identities=[]
                for snr in [0,20,50,100]:
                    for rep in range(1 if snr==0 else REPETITIONS):
                        means,counts,S0_init=fitting.observations(data,i,snr,rep)
                        tasks.append(dict(means=means,counts=counts,protocol=protocol,method=method,
                            D_fixed=D_fixed,S0_initial=S0_init,screen=screens[key]))
                        identities.append((snr,rep))
                fits=pool.map(fitting.fit_task,tasks)
                for result,(snr,rep) in zip(fits,identities):
                    row=fitting.fit_row(result,c,str(data['case_ids'][i]),method,snr,rep)
                    row['scenario']='02_free_D' if D_fixed is None else '01_fixed_D'
                    (joint if D_fixed is None else fixed).append(row)
                print(c,estimator,flush=True)
    rename={'R_hat':'Rhat','D_hat':'Dhat','S0_hat':'S0hat','rep':'realization',
            'R_bound':'boundary_R','D_bound':'boundary_D','estimator':'model'}
    return pd.DataFrame(fixed).rename(columns=rename),pd.DataFrame(joint).rename(columns=rename)


def regenerate(data):
    rows=[]
    for i in REGENERATE_CASES:
        c=data['conditions'][i].copy()  # Edit R/D here to explore a NEW case.
        p=signals.published_protocol(c['protocol'])['protocol']
        rows.append(dict(condition=c,
            reference=signals.m0_signal(c['R'],c['D'],p,radial_modes=30,lmax=30),
            direct150=signals.m0_signal(c['R'],c['D'],p),
            strang=signals.m0_strang_signal(c['R'],c['D'],p),
            gpa=signals.gpa_m0_signal(c['R'],c['D'],p,terms=128)))
    # Separate new file, never a replacement of the manuscript data.
    import json
    payload={str(i)+'_'+key:value for i,row in enumerate(rows) for key,value in row.items() if key!='condition'}
    payload['conditions_json']=json.dumps([row['condition'] for row in rows])
    np.savez_compressed(RESULTS/'signals_regenerated.npz',**payload)


def main():
    RESULTS.mkdir(exist_ok=True);data=fitting.load_experiment(DATA/'experiment.npz')
    if MODE=='regenerate':regenerate(data);return
    if MODE=='saved':
        fixed=pd.read_csv(DATA/'fits_fixed_D.csv',float_precision='round_trip')
        joint=pd.read_csv(DATA/'fits_free_D.csv',float_precision='round_trip')
    elif MODE=='refit':
        fixed,joint=refit(data)
        fixed.to_csv(RESULTS/'fits_fixed_D.csv',index=False)
        joint.to_csv(RESULTS/'fits_free_D.csv',index=False)
    else:raise ValueError('MODE must be saved, refit or regenerate')
    fitting.metrics(fixed).to_csv(RESULTS/'summary_fixed_D.csv',index=False)
    fitting.metrics(joint).to_csv(RESULTS/'summary_free_D.csv',index=False)
    plot_results(fixed,joint)
    print('M0: metrics in',RESULTS,'; figures in',ROOT/'figures')

if __name__=='__main__':main()
