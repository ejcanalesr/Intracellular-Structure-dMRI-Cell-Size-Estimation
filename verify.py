"""Small integration checks against bundled definitive data; not a campaign.

Run: python verify.py. No source data are modified. This checks code relocation
and a few complete fits, not continuum FEM/spectral convergence afresh.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import signals as s
import fitting as f

ROOT=Path(__file__).resolve().parent


def check_observations():
    """Every saved noisy magnitude is reproduced using its original seed."""
    expected_cases={'M0':28,'M1':28,'M2':18,'M3':18}
    for model in expected_cases:
        data=f.load_experiment(ROOT/model/'data/experiment.npz')
        assert len(data['conditions'])==expected_cases[model]
        for i,c in enumerate(data['conditions']):
            n=5 if c['protocol']=='P1' else 8
            p=s.published_protocol(c['protocol']);counts=np.r_[p['counts'],n]
            np.testing.assert_array_equal(data['counts'][i,:n+1],counts)
            for j,snr in enumerate(data['snrs']):
                regenerated=f.generate_observations(data['signals'][i,:n],c['protocol'],c,int(snr),campaign=model)
                for key in ['dwi','b0','means','S0_initial']:
                    a=data[key][i,j]
                    if key in ['means','b0']:a=a[...,:n+(key=='means')]
                    np.testing.assert_array_equal(regenerated[key],a)
        print(model,': saved observations and seeds match for every case')


def check_signals():
    data=f.load_experiment(ROOT/'M0/data/experiment.npz')
    for name,R,D in [('P1',7.,2.),('P2',3.,.6),('P2',15.,2.)]:
        i=next(i for i,c in enumerate(data['conditions']) if c==dict(protocol=name,R=R,D=D))
        p=s.published_protocol(name)['protocol'];n=len(p['b_s_mm2'])
        np.testing.assert_allclose(s.m0_signal(R,D,p),data['direct150'][i,:n],atol=3e-13,rtol=0)
        np.testing.assert_allclose(s.gpa_m0_signal(R,D,p,terms=128),data['gpa128'][i,:n],atol=3e-13,rtol=0)
        settings=dict(calibration_format=1,gamma_rad_s_T=s.GAMMA_RAD_S_T,model='M0',resolution=[10,14],quadrature=None,
            parameter_region={'R':[2.,20.],'D':[.3,3.]},protocol=s.protocol_record(p),subdivisions=[64]*n)
        np.testing.assert_allclose(s.m0_strang_signal(R,D,p),s.m0_signal(R,D,p,backend='strang',calibration=settings),atol=3e-13,rtol=0)
    data=f.load_experiment(ROOT/'M1/data/experiment.npz')
    for k in [0.,200.,float('inf')]:
        i=next(i for i,c in enumerate(data['conditions']) if c['protocol']=='P1' and c['Rc']==7 and float(c['kappa_um_s'])==k)
        c=data['conditions'][i];p=s.published_protocol('P1')['protocol']
        np.testing.assert_allclose(s.m1_signal(7.,4.2,2.,2.,k,p),data['signals'][i,:5],atol=3e-12,rtol=0)
    # Transparent equal-D and zero-radius endpoints, at matched spatial budgets.
    p=s.published_protocol('P1')['protocol'];m0=s.m0_signal(7.,2.,p,radial_modes=12,lmax=14)
    np.testing.assert_allclose(s.m1_signal(7.,0.,1.,2.,20.,p,12,14),m0,atol=3e-13,rtol=0)
    np.testing.assert_allclose(s.m1_signal(7.,4.2,2.,2.,np.inf,p,12,14),m0,atol=2e-10,rtol=0)
    for k,nzero in [(0.,2),(.2,1),(np.inf,1)]:
        # spectrum_info is the INTERNAL interface: k is um/ms here.
        info=s.spectrum_info('M1',dict(R=7.,a=4.2,Dn=2.,Dc=2.,kappa=k),radial_modes=12,lmax=14)
        assert info['zero_modes']==nzero
        assert info['generator_equilibrium_norm']<1e-12
        np.testing.assert_allclose(sum(info['zero_mode_weights']),1.,atol=1e-13)
        if k==0:np.testing.assert_allclose(sorted(info['zero_mode_weights']),[.6**3,1-.6**3],atol=1e-12)
    zero=s.pgse_protocol([0.,0.],[6.,10.],[13.,30.])
    np.testing.assert_array_equal(s.m0_signal(7.,2.,zero),[1.,1.])
    p=s.published_protocol('P2')['protocol']
    np.testing.assert_allclose(1000*(s.GAMMA*p['G_T_m']*p['delta_ms'])**2*(p['Delta_ms']-p['delta_ms']/3),p['b_s_mm2'],atol=2e-12)
    print('Signals: real expm, Numba Strang, GPA, M1 endpoints and zero modes match')


def check_fits():
    """Four complete same-data fits, including fixed/free D and GPA/Strang."""
    for model in ['M0','M1','M2','M3']:
        data=f.load_experiment(ROOT/model/'data/experiment.npz')
        csv='fits_fixed_D.csv' if model=='M0' else 'fits.csv'
        frame=pd.read_csv(ROOT/model/'data'/csv,float_precision='round_trip')
        if model=='M0':
            expected=frame[(frame.protocol=='P1')&(frame.R==7)&(frame.D==2)&(frame.snr==50)&(frame.realization==0)&(frame.model=='M0-Strang')].iloc[0]
            i=next(i for i,c in enumerate(data['conditions']) if c==dict(protocol='P1',R=7.,D=2.))
            method='M0-Strang';D_fixed=2.;names=['Rhat','Dhat','S0hat']
        else:
            if model=='M1':mask=(frame.protocol=='P1')&(frame.Rc==7)&(frame.kappa_um_s==0)&(frame.estimator=='GPA_fixed_D')
            elif model=='M2':mask=(frame.protocol=='P2')&(frame.ratio>1.2)&(frame.offset_index==2)&(frame.estimator=='Strang_free_D')
            else:mask=(frame.protocol=='P1')&(frame.N==20)&(frame.kappa_org_um_s.astype(str)=='20')&(frame.estimator=='Strang_free_D')
            expected=frame[mask&(frame.snr==50)&(frame.rep==0)].iloc[0]
            i=list(data['case_ids']).index(expected.configuration)
            method='M0-GPA' if expected.estimator.startswith('GPA') else 'M0-Strang'
            D_fixed=None if expected.estimator.endswith('free_D') else float(expected.Dc)
            names=['R_hat','D_hat','S0_hat']
        c=data['conditions'][i];p=s.published_protocol(c['protocol'])['protocol']
        means,counts,S0_init=f.observations(data,i,50,0)
        result=f.fit_radius(means,counts,p,method,D_fixed,S0_init)
        parameters=result['parameters'];D=parameters.get('D',D_fixed)
        found=[parameters['R'],D,parameters['S0']];saved=expected[names].to_numpy(dtype=float)
        np.testing.assert_allclose(found,saved,atol=2e-6,rtol=0)
        np.testing.assert_allclose(result['objective'],expected.objective,atol=1e-10,rtol=0)
        normalized=np.asarray(result['normalized_prediction'])
        assert abs(np.dot(counts*normalized,parameters['S0']*normalized-means))<3e-12
        assert result['success']==expected.success
        print(model,': selected fit matches; max parameter difference',float(np.max(np.abs(np.array(found)-saved))))


def check_meshes():
    for model,prefixes in [('M2',['a1_','a2_']),('M3',['N5_','N20_','N60_'])]:
        with np.load(ROOT/model/'data/meshes.npz',allow_pickle=False) as mesh:
            for prefix in prefixes:
                p=mesh[prefix+'points'];tri=mesh[prefix+'triangles'];b=mesh[prefix+'boundaries']
                assert tri.min()>=0 and tri.max()<len(p) and len(b)==len(tri)
                if prefix+'tetrahedra' in mesh:
                    tet=mesh[prefix+'tetrahedra']
                    assert tet.min()>=0 and tet.max()<len(p) and len(mesh[prefix+'regions'])==len(tet)
    print('Meshes: valid saved node/triangle/tetrahedron indices; no image inputs')


if __name__=='__main__':
    check_observations();check_signals();check_fits();check_meshes()
    print('All consolidation checks passed. No full campaign or external FEM run.')
