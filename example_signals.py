"""Standalone guide: generate and plot normalized finite-PGSE signals.

Place beside signals.py in the extracted repository; run python example_signals.py.
M0/M1 are calculated afresh. M2/M3 load accepted FEM signals by default.
FEM_MODE='compute' explicitly calls the original MATLAB/SpinDoctor route, which
can take minutes or longer and was not rerun for this example. Set
SPINDOCTOR_ROOT as described in README.md. No fitting or manuscript files change.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='1'
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import signals
import plot_style as PS


# EDIT THE PHYSICAL PARAMETERS AND ACQUISITION HERE.
R_UM=7.0
D_UM2_MS=2.0
NUCLEAR_RADIUS_UM=4.2
NUCLEAR_D_UM2_MS=2.0
NUCLEAR_KAPPA_UM_S=200.0
PROTOCOL_NAME='P1'                  # Published P1 or P2.

# None uses the published shells/timings. For another protocol, specify b and
# the two timings. Saved FEM values cannot be reused at different acquisitions.
CUSTOM_B_S_MM2=None                # Example: [500,1000,2000,4000,6000].
CUSTOM_DELTA_MS=6.0
CUSTOM_BIG_DELTA_MS=13.0

INCLUDE_FEM=True
FEM_MODE='saved'                   # 'saved' or explicit expensive 'compute'.
M2_CASE='P1_r10_a2'                 # Eccentric nucleus, alpha=2/3, Dn=Dc.
M3_CASE='P1_N5_k0'                  # Concentric nucleus +5 sealed inclusions.
# For a matched P2 example: R=5, Dc=Dn=0.6, Rn=4,
# M2_CASE='P2_r10_a2', M3_CASE='P2_N10_k0' (same nuclear kappa=200).
MATLAB='matlab'

M0_RADIAL_MODES=10                 # 10*(14+1)=150 modes.
M0_LMAX=14
M1_RADIAL_MODES=24                 # 24*(24+1)=600 modes.
M1_LMAX=24
# These spatial budgets are the manuscript choices, not a convergence guarantee
# for arbitrary new radii, diffusivities, permeabilities or gradient waveforms.
SAVE_SEPARATE_FIGURES=False
ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'example_output'        # Separate from every manuscript output.
COLORS={'M0':PS.MODEL_COLORS['M0-expm-truncated'],
        'M1':'#21856C','M2':'#BD5360','M3':'#3C8FA6'}


def fem_signal(model,case_id,protocol):
    """Read one accepted geometry; optionally regenerate it at these timings.

    Case IDs define the FEM geometry/physics. They do not rescale when R or D
    changes above: choose matching cases, or set INCLUDE_FEM=False for arbitrary
    M0/M1 parameters. The returned provenance makes loaded/computed explicit.
    """
    with np.load(ROOT/model/'data/experiment.npz',allow_pickle=False) as data:
        index=list(data['case_ids']).index(case_id)
        condition=json.loads(str(data['conditions_json']))[index]
        descriptor=json.loads(str(data['fem_jobs_json']))[index]
        n=5 if condition['protocol']=='P1' else 8
        saved=data['signals'][index,:n].copy()
    kappa=condition['kappa_um_s'] if model=='M2' else condition['kappa_n_um_s']
    actual=[condition['Rc'],condition['Dc'],condition['Rn'],
            condition['ratio']*condition['Dc'],float(kappa)]
    requested=[R_UM,D_UM2_MS,NUCLEAR_RADIUS_UM,NUCLEAR_D_UM2_MS,NUCLEAR_KAPPA_UM_S]
    if not np.allclose(actual,requested,rtol=1e-12,atol=1e-12):
        raise ValueError(f'{model} case {case_id} has different physical parameters: {condition}')
    job=descriptor['jobs'][0]
    same_protocol=all(np.asarray(job[key]).shape==protocol[key].shape and
        np.allclose(job[key],protocol[key],rtol=1e-12,atol=1e-10)
        for key in ['b_s_mm2','delta_ms','Delta_ms'])
    if FEM_MODE=='saved':
        if not same_protocol:
            raise ValueError(f'{case_id}: saved FEM does not match the requested acquisitions. '
                             'Choose matching cases, disable FEM, or use FEM_MODE="compute".')
        result=saved
        source='loaded accepted FEM spherical mean'
    elif FEM_MODE=='compute':
        # Only the acquisition is changed; geometry, compartments, retained
        # modes and spherical quadrature stay as in this accepted case.
        for job in descriptor['jobs']:
            for key in ['b_s_mm2','delta_ms','Delta_ms']:
                job[key]=protocol[key].tolist()
        from signals_fem import regenerate_signal
        print(f'Computing {model} ({case_id}) with MATLAB/SpinDoctor...',flush=True)
        # Different acquisitions get separate work folders, so an old MATLAB
        # result cannot be mistaken for a calculation at new b/delta/Delta.
        waveform={key:protocol[key].tolist() for key in ['b_s_mm2','delta_ms','Delta_ms']}
        key=hashlib.sha256(json.dumps(waveform,sort_keys=True).encode()).hexdigest()[:12]
        result=regenerate_signal(descriptor,OUTPUT/'fem'/f'{case_id}_{key}',MATLAB)
        source='new MATLAB/SpinDoctor FEM calculation; inspect convergence before new scientific use'
        if same_protocol:print('  max difference from accepted signal:',np.max(np.abs(result-saved)),flush=True)
    else:
        raise ValueError('FEM_MODE must be saved or compute')
    if model=='M2':description=rf'M2: $\alpha={condition["alpha"]:.3f}$'
    else:description=rf'M3: $N={condition["N"]}$, $\kappa_{{org}}={condition["kappa_org_um_s"]}$'
    return np.r_[1.,result],dict(case=case_id,condition=condition,source=source,
        label=description+(' (saved FEM)' if FEM_MODE=='saved' else ' (computed FEM)'),
        saved_descriptor=descriptor)


def plot_signals(protocol,curves,metadata,filename):
    PS.apply_style()
    fig,ax=plt.subplots(figsize=(7.6,4.4))
    for model,values in curves.items():
        ax.plot(protocol['b_s_mm2'],values,'o-',ms=4,lw=1.3,color=COLORS[model],label=metadata[model]['label'])
    ax.set(xlabel=r'$b$ (s/mm$^2$)',ylabel=r'Normalized signal $S/S_0$',ylim=(0,1.03))
    ax.grid(alpha=PS.GRID_ALPHA)
    ax.legend(frameon=False,fontsize=8)
    title='Custom rectangular PGSE' if CUSTOM_B_S_MM2 is not None else PS.PROTOCOL_LABELS[PROTOCOL_NAME]
    ax.set_title(title+'\n'+rf'$R={R_UM:g}$ $\mu$m, $D_c={D_UM2_MS:g}$ $\mu$m$^2$/ms',
        fontsize=PS.TITLE_FONT_SIZE)
    fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.88)
    PS.save_pdf(fig,filename)
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    protocol=signals.published_protocol(PROTOCOL_NAME)['protocol']
    if CUSTOM_B_S_MM2 is not None:
        protocol=signals.pgse_protocol(CUSTOM_B_S_MM2,CUSTOM_DELTA_MS,CUSTOM_BIG_DELTA_MS)
    if INCLUDE_FEM and np.any(protocol['b_s_mm2']==0):
        raise ValueError('Specify DW acquisitions only; the script adds the normalized b=0 point.')
    # Prepend the mathematical normalization point. This is not a noisy b=0 observation.
    p=signals.pgse_protocol(np.r_[0.,protocol['b_s_mm2']],
        np.r_[protocol['delta_ms'][0],protocol['delta_ms']],
        np.r_[protocol['Delta_ms'][0],protocol['Delta_ms']])

    # These are the actual public forward calls, with explicit physical inputs.
    curves={}
    curves['M0']=signals.m0_signal(R_UM,D_UM2_MS,p,radial_modes=M0_RADIAL_MODES,lmax=M0_LMAX)
    curves['M1']=signals.m1_signal(R_UM,NUCLEAR_RADIUS_UM,NUCLEAR_D_UM2_MS,D_UM2_MS,
        NUCLEAR_KAPPA_UM_S,p,radial_modes=M1_RADIAL_MODES,lmax=M1_LMAX)
    # Other M0 choices, if desired: signals.m0_strang_signal(R_UM,D_UM2_MS,p)
    # or signals.gpa_m0_signal(R_UM,D_UM2_MS,p,terms=128). GPA is a physical approximation.
    metadata={
        'M0':dict(source='calculated real expm',label='M0: real expm',
                  radial_modes=M0_RADIAL_MODES,lmax=M0_LMAX),
        'M1':dict(source='calculated real expm',label='M1: concentric nucleus',
                  radial_modes=M1_RADIAL_MODES,lmax=M1_LMAX)}
    if INCLUDE_FEM:
        curves['M2'],metadata['M2']=fem_signal('M2',M2_CASE,protocol)
        curves['M3'],metadata['M3']=fem_signal('M3',M3_CASE,protocol)

    # Save both the numerical signals and the actual b/delta/Delta/G arrays.
    parameters=dict(R_um=R_UM,Dc_um2_ms=D_UM2_MS,Rn_um=NUCLEAR_RADIUS_UM,
        Dn_um2_ms=NUCLEAR_D_UM2_MS,kappa_n_um_s=NUCLEAR_KAPPA_UM_S,
        protocol_name=PROTOCOL_NAME,custom_protocol=CUSTOM_B_S_MM2 is not None,
        gamma_rad_ms_um_T_m=signals.GAMMA_RAD_MS_UM_T_M,
        signal='S/S0; no noise or relaxation contrast; b0 is a normalization point',
        plotting='lines connect computed/saved acquisitions; no FEM interpolation')
    name='signals_custom' if CUSTOM_B_S_MM2 is not None else 'signals_'+PROTOCOL_NAME
    np.savez_compressed(OUTPUT/(name+'.npz'),**curves,**p,
        metadata_json=json.dumps(dict(parameters=parameters,models=metadata)))
    columns=['b_s_mm2','delta_ms','Delta_ms','G_T_m']
    table=np.column_stack([p[key] for key in columns]+list(curves.values()))
    np.savetxt(OUTPUT/(name+'.csv'),table,delimiter=',',header=','.join(columns+list(curves)),comments='')
    plot_signals(p,curves,metadata,OUTPUT/name)
    if SAVE_SEPARATE_FIGURES:
        for model,values in curves.items():plot_signals(p,{model:values},metadata,OUTPUT/(name+'_'+model))
    for model in curves:print(model,':',metadata[model]['source'])
    print('delta (ms):',protocol['delta_ms'],'; Delta (ms):',protocol['Delta_ms'])
    print('Required G (mT/m):',1000*protocol['G_T_m'])
    print('PDF, CSV and NPZ saved in',OUTPUT)


if __name__=='__main__':main()
