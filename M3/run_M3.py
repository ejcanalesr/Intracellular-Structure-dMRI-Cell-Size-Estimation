"""M3: saved results, optional same-data refitting, or physical regeneration."""
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
from plot_M3 import plot_results

# EDIT THESE SETTINGS. Default: load accepted results, calculate metrics, plot.
MODE='saved'                 # 'saved', 'refit', or 'regenerate'
N_WORKERS=8                  # Used only for explicit refitting; use1 for serial.
REPETITIONS=50
DATA=Path(__file__).resolve().parent/'data'
RESULTS=Path(__file__).resolve().parent/'results'
REGENERATE_CASES=[0]          # Expensive physical generation is opt-in, case by case.
MATLAB='matlab'               # Used only by M2/M3 physical regeneration.


def refit(data):
    rows=[];screens={}
    with Pool(N_WORKERS) as pool:
        for i,c in enumerate(data['conditions']):
            protocol=signals.published_protocol(c['protocol'])['protocol']
            estimators=['Strang_free_D','Strang_fixed_D','GPA_fixed_D'] if c['protocol']=='P1' else ['Strang_free_D','GPA_free_D']
            for estimator in estimators:
                method='M0-GPA' if estimator.startswith('GPA') else 'M0-Strang'
                D_fixed=None if estimator.endswith('free_D') else c['Dc']
                key=(c['protocol'],method,D_fixed)
                if key not in screens:screens[key]=fitting.make_screen(protocol,method,D_fixed)
                tasks=[];identities=[]
                for snr in [0,50]:
                    for rep in range(1 if snr==0 else REPETITIONS):
                        means,counts,S0_init=fitting.observations(data,i,snr,rep)
                        tasks.append(dict(means=means,counts=counts,protocol=protocol,method=method,
                            D_fixed=D_fixed,S0_initial=S0_init,screen=screens[key]))
                        identities.append((snr,rep))
                fits=pool.map(fitting.fit_task,tasks)
                for result,(snr,rep) in zip(fits,identities):
                    rows.append(fitting.fit_row(result,c,str(data['case_ids'][i]),estimator,snr,rep))
                print(c,estimator,flush=True)
    frame=pd.DataFrame(rows)
    frame['kappa_org_um_s']=frame.kappa_org_um_s.astype(str)
    frame['cell_fraction']=np.where(frame.protocol=='P1',.06*(1-(4.2/7)**3)*frame.N/20,.038*frame.N/60)
    keys=['protocol','estimator','snr','rep']
    base=frame[frame.N==0][keys+['e_R_pct']].rename(columns={'e_R_pct':'base_error'})
    frame=frame.merge(base,on=keys,validate='many_to_one')
    frame['delta_e_R_pp']=frame.e_R_pct-frame.base_error;frame=frame.drop(columns='base_error')
    return frame


def regenerate(data):
    output={}
    for i in REGENERATE_CASES:
        c=data['conditions'][i].copy()
        # These descriptors retain the original mesh, eigenmode and quadrature
        # choices. Inspect them before an expensive external calculation.
        from signals_fem import regenerate_signal
        signal=regenerate_signal(data['fem_jobs'][i],RESULTS/'fem'/str(data['case_ids'][i]),MATLAB)
        n=len(signal);output[str(data['case_ids'][i])]=signal
        print(c,'max difference from saved signal:',np.max(np.abs(signal-data['signals'][i,:n])))
    np.savez_compressed(RESULTS/'signals_regenerated.npz',**output)


def main():
    RESULTS.mkdir(exist_ok=True);data=fitting.load_experiment(DATA/'experiment.npz')
    if MODE=='regenerate':regenerate(data);return
    if MODE=='saved':
        frame=pd.read_csv(DATA/'fits.csv',float_precision='round_trip',dtype={'kappa_org_um_s':str})
    elif MODE=='refit':
        frame=refit(data);frame.to_csv(RESULTS/'fits.csv',index=False)
    else:raise ValueError('MODE must be saved, refit or regenerate')
    fitting.metrics(frame).to_csv(RESULTS/'summary.csv',index=False)
    plot_results(frame)
    from plot_M3 import plot_mesh
    plot_mesh()
    print('M3: metrics in',RESULTS,'; figures in',ROOT/'figures')

if __name__=='__main__':main()
