"""Parameter fitting, Rician noise and descriptive metrics.

The profiled-S0 WLS, multistart optimizer, final refinement and uniform retry
are copied from the definitive campaign. No changes to the fitting mathematics.
"""
from functools import lru_cache
from time import perf_counter
import itertools
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize,least_squares
import signals
from pathlib import Path

R_BOUNDS=[2.,20.]
D_BOUNDS=[.3,3.]
R_STARTS=[2,3,5,7,9,11,13,15,17,20]
D_STARTS=[.3,.6,1.,2.,3.]
OPTIMIZER=dict(starts=3,separation=.15,maxiter=200,ftol=1e-12,gtol=1e-7,
    signal_derivative_step=1e-4,objective_scale=1e4,residual_polish=True,polish_max_nfev=200)


# Validated local optimizer and WLS helpers
def bounds_arrays(bounds):
    names=list(bounds);lo=np.array([bounds[k][0] for k in names],float)
    hi=np.array([bounds[k][1] for k in names],float)
    if np.any(lo<=0) or np.any(hi<=lo):raise ValueError('Strict positive ordered fitting bounds required')
    return names,lo,hi


def select_starts(grid, costs, number=3, separation=.25):
    """Data-ranked, spatially dispersed screening; ties use original grid order."""
    order=np.lexsort((np.arange(len(costs)),costs));chosen=[]
    for i in order:
        if not chosen or min(np.linalg.norm(grid[i]-grid[k]) for k in chosen)>=separation:
            chosen.append(int(i))
        if len(chosen)==number:break
    if len(chosen)<number:
        chosen.extend(int(i) for i in order if i not in chosen)
    return chosen[:number]


def local_minimize(objective, start, settings, gradient=None):
    return minimize(objective,np.asarray(start),method='L-BFGS-B',bounds=[(0.,1.)]*len(start),
                    jac=gradient,
                    options=dict(maxiter=settings.get('maxiter',150),maxls=30,
                                 ftol=settings.get('ftol',1e-12),gtol=settings.get('gtol',1e-7),
                                 eps=settings.get('eps',1e-6)))


# Arithmetic magnitude means and screening
def shell_means(observations,shell_index,number_shells=None):
    """Arithmetic mean of individual magnitudes; accepts one or many voxels."""
    y=np.asarray(observations,float);index=np.asarray(shell_index,int)
    n=int(index.max())+1 if number_shells is None else int(number_shells)
    if y.shape[-1]!=len(index) or np.any(index<0) or np.any(index>=n):raise ValueError('Invalid shell design')
    counts=np.bincount(index,minlength=n)
    if np.any(counts==0) or not np.isfinite(y).all():raise ValueError('Empty shell or nonfinite observation')
    return np.stack([np.mean(y[...,index==j],axis=-1) for j in range(n)],axis=-1)


def weighted_objective(prediction,means,counts):
    prediction,means,counts=map(lambda a:np.asarray(a,float),(prediction,means,counts))
    if prediction.shape!=means.shape or means.shape!=counts.shape or np.any(counts<=0):raise ValueError('One positive weight per acquisition required')
    return float(np.dot(counts,(prediction-means)**2))


def prepare_weighted_screen(forward,bounds,physical_grid):
    names,lo,hi=bounds_arrays(bounds);physical=np.asarray(physical_grid,float)
    if physical.ndim!=2 or physical.shape[1]!=len(names) or np.any(physical<lo) or np.any(physical>hi):raise ValueError('Candidate outside physical bounds')
    tic=perf_counter();signals=np.asarray([forward(theta) for theta in physical])
    return dict(names=names,bounds=bounds,physical_grid=physical,grid=(physical-lo)/(hi-lo),signals=signals,seconds=perf_counter()-tic,evaluations=len(physical))


# Exact analytical elimination of S0 and the final multistart fit
def profile_amplitude(normalized, means, counts):
    """Return the exact unconstrained scalar WLS amplitude.

    Do not subtract two large squares to evaluate the minimized objective:
    callers evaluate the residual using this amplitude instead. There is no
    initialization, clipping, prior, or numerical optimization of S0 here.
    """
    f, y, w = (np.asarray(x, float) for x in (normalized, means, counts))
    if f.ndim != 1 or f.shape != y.shape or f.shape != w.shape or not all(np.isfinite(x).all() for x in (f, y, w)) or np.any(w <= 0):
        raise ValueError('Finite vectors and positive multiplicities required')
    denominator = float(np.dot(w, f * f))
    if denominator <= 0:
        raise ValueError('Amplitude is undefined for an identically zero prediction')
    return float(np.dot(w * y, f) / denominator)


def fit_weighted_means_profiled_s0(forward, means, counts, bounds, *, S0_init, screen, settings=None):
    """Screened bounded multistart WLS in R, or R,D; eliminate S0 exactly.

    S0_init ranks physical starts exactly as in the bounded reference fitter.
    It does not enter the analytic amplitude solution or local objective.
    The physical grid and first three starts are unchanged. Joint fits also
    cover the best profiled screen point in each diffusivity stratum. One
    residual-based local refinement checks the best endpoint; it does not
    add another screen or use earlier estimates or generating parameters.
    The envelope theorem removes the derivative of the minimizing amplitude:
    dF/dtheta = 2*S0*sum(w*(S0*f-y)*df/dtheta).
    """
    settings = {} if settings is None else settings
    names, lo, hi = bounds_arrays(bounds)
    means, counts = np.asarray(means, float), np.asarray(counts, float)
    initial = float(S0_init)
    if 'S0' in names or not np.isfinite(initial) or initial <= 0:
        raise ValueError('Positive diagnostic S0_init required separately')
    if means.ndim != 1 or means.shape != counts.shape or not np.isfinite(means).all() or not np.isfinite(counts).all() or np.any(means < 0) or np.any(counts <= 0):
        raise ValueError('Invalid magnitude means or multiplicities')
    if screen['names'] != names or screen['bounds'] != bounds:
        raise ValueError('Mismatched physical screen')
    signals = np.asarray(screen['signals'], float)
    if signals.shape != (len(screen['grid']), len(means)) or not np.isfinite(signals).all():
        raise ValueError('Invalid screening predictions')
    tic = perf_counter()
    calls = 0
    scale = settings.get('objective_scale', 1e4) / counts.sum()
    known = {tuple(map(float, z)): f for z, f in zip(screen['grid'], signals)}

    @lru_cache(maxsize=1024)
    def normalized(key):
        nonlocal calls
        if key in known:
            return known[key]
        theta = np.clip(lo + np.asarray(key) * (hi - lo), lo, hi)
        calls += 1
        f = np.asarray(forward(theta), float)
        if f.shape != means.shape or not np.isfinite(f).all():
            raise ValueError('Invalid normalized forward prediction')
        return f

    def state(z):
        f = normalized(tuple(map(float, z)))
        return f, profile_amplitude(f, means, counts)

    def raw(z):
        f, amplitude = state(z)
        return weighted_objective(amplitude * f, means, counts)

    def objective(z):
        return scale * raw(z)

    def gradient(z):
        f, amplitude = state(z)
        score = 2 * counts * (amplitude * f - means) * amplitude
        h = settings.get('signal_derivative_step', 1e-4)
        out = []
        for j in range(len(names)):
            plus, minus = np.array(z), np.array(z)
            plus[j], minus[j] = min(1., z[j] + h), max(0., z[j] - h)
            derivative = (normalized(tuple(plus)) - normalized(tuple(minus))) / (plus[j] - minus[j])
            out.append(scale * np.dot(score, derivative))
        return np.asarray(out)

    amplitudes = np.array([profile_amplitude(f, means, counts) for f in signals])
    costs = np.sum(counts * (amplitudes[:, None] * signals - means) ** 2, axis=1)
    ranking_costs = np.sum(counts * (initial * signals - means) ** 2, axis=1)
    chosen = select_starts(screen['grid'], ranking_costs, settings.get('starts', 3), settings.get('separation', .15))
    # Keep the original starts, then cover the supplied coarse diffusivity
    # strata. The joint restricted-sphere cost can have separated basins that
    # three greedily dispersed points miss. This uses only current data and
    # the declared screen, never an earlier fitted shape or generating truth.
    if 'D' in names:
        column = names.index('D')
        physical_grid = np.asarray(screen['physical_grid'])
        for diffusivity in np.unique(physical_grid[:, column]):
            candidates = np.flatnonzero(physical_grid[:, column] == diffusivity)
            index = int(candidates[np.argmin(costs[candidates])])
            if index not in chosen:
                chosen.append(index)
    trials = []
    for i in chosen:
        result = local_minimize(objective, screen['grid'][i], settings, gradient)
        z = np.clip(result.x, 0., 1.)
        physical = np.clip(lo + z * (hi - lo), lo, hi)
        f, amplitude = state(z)
        trials.append(dict(start_index=int(i), start_parameters=screen['physical_grid'][i].tolist(),
            start_profiled_S0=float(amplitudes[i]), parameters=np.r_[physical, amplitude].tolist(), z=z.tolist(),
            objective=raw(z), success=bool(result.success), status=int(result.status), message=str(result.message),
            iterations=int(result.nit), optimizer_function_evaluations=int(result.nfev)))
    converged = [t for t in trials if t['success'] and np.isfinite(t['objective'])]
    finite = [t for t in trials if np.isfinite(t['objective'])]
    if not finite:
        raise ArithmeticError('No finite profiled-S0 solution')
    best = min(converged or finite, key=lambda t: (t['objective'], t['start_index']))
    # L-BFGS-B can terminate on small objective changes while still moving
    # along a weak R,D valley. Refine its best endpoint using the same WLS
    # residuals and exact conditional amplitude. Finite differences of the
    # complete residual include the amplitude derivative automatically.
    if settings.get('residual_polish', True):
        start = np.asarray(best['z'])

        def residual(z):
            f, amplitude = state(z)
            return np.sqrt(scale * counts) * (amplitude * f - means)

        result = least_squares(residual, start, bounds=(np.zeros(len(names)), np.ones(len(names))),
                               jac='3-point', ftol=1e-12, xtol=1e-12, gtol=1e-10,
                               max_nfev=settings.get('polish_max_nfev', 200))
        z = np.clip(result.x, 0., 1.)
        physical = np.clip(lo + z * (hi-lo), lo, hi)
        f, amplitude = state(z)
        trials.append(dict(start_index=-1, kind='local_residual_refinement',
            start_parameters=(lo + start*(hi-lo)).tolist(), start_profiled_S0=float(state(start)[1]),
            parameters=np.r_[physical, amplitude].tolist(), z=z.tolist(), objective=raw(z),
            success=bool(result.success), status=int(result.status), message=str(result.message),
            iterations=int(result.nfev), optimizer_function_evaluations=int(result.nfev),
            optimality=float(result.optimality)))
        converged = [t for t in trials if t['success'] and np.isfinite(t['objective'])]
        finite = [t for t in trials if np.isfinite(t['objective'])]
        best = min(converged or finite, key=lambda t: (t['objective'], t['start_index']))
    z = np.asarray(best['z'])
    f, amplitude = state(z)
    near = [t for t in finite if t['objective'] <= best['objective'] + 1e-8]
    spread = max(float(np.linalg.norm(np.asarray(t['z']) - z)) for t in near)
    return dict(parameters=dict(zip([*names, 'S0'], best['parameters'])), objective=float(best['objective']), success=bool(converged),
        boundary_hits={**{k: bool(min(v, 1-v) <= 1e-4) for k, v in zip(names, z)}, 'S0': False},
        S0_initial=initial, S0_bounds=None, amplitude_method='analytic_unbounded_WLS', optimization_dimension=len(names),
        screen_ranking='Original WLS screen at measured S0_init; local optimization profiles S0',
        effective_starts=len(chosen),joint_diffusivity_coverage=bool('D' in names),
        forward_evaluations=calls, seconds=perf_counter()-tic, screen_evaluations=len(costs),
        screen_minimum_objective=float(costs.min()), successful_but_screen_better=bool(best['objective'] > costs.min()+1e-10),
        competing_near_minima=bool(len(near)>1 and spread>.02), near_minimum_scaled_parameter_span=spread,
        unsuccessful_starts=sum(not t['success'] for t in trials), trials=trials,
        prediction=(amplitude*f).tolist(), normalized_prediction=f.tolist(),
        optimizer='screened_multistart_L-BFGS-B_with_local_residual_refinement', representation='linear physical bounds scaled to [0,1]; S0 analytically eliminated',
        optimized_objective_scale=scale, settings=settings)


# Uniform safeguard used for every dataset
def fit_dataset(data,f,screened,physical_bounds,optimizer,model):
    """The same physical multistart policy with S0 eliminated analytically."""
    kwargs=dict(S0_init=data['S0_initial'],screen=screened)
    fit=fit_weighted_means_profiled_s0(f,data['means'],data['counts'],physical_bounds,settings=optimizer,**kwargs)
    better_failed=fit['objective']>min(t['objective'] for t in fit['trials'])+1e-8
    if fit['successful_but_screen_better'] or better_failed:
        settings=optimizer.copy();settings['signal_derivative_step']=1e-5
        retry=fit_weighted_means_profiled_s0(f,data['means'],data['counts'],physical_bounds,settings=settings,**kwargs)
        retry['original_attempt']=fit
        retry['objective_guard']='One same-grid/same-optimizer retry with physical derivative step1e-5; original retained'
        retry['seconds']+=fit['seconds'];retry['forward_evaluations']+=fit['forward_evaluations']
        if not retry['success'] or retry['successful_but_screen_better'] or retry['objective']>min(fit['objective'],min(t['objective'] for t in fit['trials']))+1e-8:
            candidates=[dict(t,guard_attempt=j) for j,attempt in enumerate([fit,retry]) for t in attempt['trials'] if np.isfinite(t['objective'])]
            best=min(candidates,key=lambda t:(t['objective'],t['guard_attempt'],t['start_index']))
            out=dict(retry);z=np.asarray(best['z']);parameters=dict(zip([*physical_bounds,'S0'],best['parameters']))
            normal=np.asarray(f([parameters[k] for k in physical_bounds]))
            parameters['S0']=profile_amplitude(normal,data['means'],data['counts']);prediction=parameters['S0']*normal
            near=[t for t in candidates if t['objective']<=best['objective']+1e-8]
            spread=max(float(np.linalg.norm(np.asarray(t['z'])-z)) for t in near)
            out.update(parameters=parameters,objective=float(np.dot(data['counts'],(prediction-data['means'])**2)),success=False,
                unresolved_optimizer_guard=True,trials=candidates,normalized_prediction=normal.tolist(),prediction=prediction.tolist(),
                guard_retry_attempt=retry,boundary_hits={**{k:bool(min(v,1-v)<=1e-4) for k,v in zip(physical_bounds,z)},'S0':False},
                unsuccessful_starts=sum(not t['success'] for t in candidates),competing_near_minima=bool(len(near)>1 and spread>.02),near_minimum_scaled_parameter_span=spread,
                successful_but_screen_better=bool(best['objective']>min(fit['screen_minimum_objective'],retry['screen_minimum_objective'])+1e-10),
                objective_guard='One same-grid derivative retry unresolved: best finite endpoint retained with optimizer failure flag; no extra starts')
            out['forward_evaluations']+=1
            return dict(model=model,**out)
        fit=retry
    return dict(model=model,**fit)


# Original random-number primitives
def rician_noise(signal, snr, rng, S0=1., return_components=False):
    signal=np.asarray(signal,dtype=float)
    if np.any(~np.isfinite(signal)) or np.any(signal<0) or S0<=0 or snr<=0:
        raise ValueError('Nonnegative signals and positive SNR/S0 required')
    sigma=S0/float(snr)
    if np.isinf(snr):
        y=signal.copy();z=np.zeros_like(y)
        return (y,z,z.copy()) if return_components else y
    z1=rng.standard_normal(signal.shape);z2=rng.standard_normal(signal.shape)
    y=np.hypot(signal+sigma*z1,sigma*z2)
    return (y,z1,z2) if return_components else y


def seeded_rng(seed, *indices):
    """Stable stream identity, independent of worker count/order and Python hash."""
    return np.random.default_rng(np.random.SeedSequence([int(seed),*map(int,indices)]))


# Canonical geometry hashing for original paired-noise seeds
def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return jsonable(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)) and not np.isfinite(x):return str(float(x))
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x


def content_hash(value):
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


# Public functions used directly by run_M0.py ... run_M3.py.
def fit_forward(theta,protocol,method,D_fixed=None):
    R=float(theta[0]);D=float(theta[1]) if D_fixed is None else float(D_fixed)
    if method=='M0-Strang':s=signals.m0_strang_signal(R,D,protocol,64,10,14)
    elif method=='M0-GPA':s=signals.gpa_m0_signal(R,D,protocol,terms=128)
    else:raise ValueError('Choose M0-Strang or M0-GPA')
    return np.r_[s,1.]  # The last entry is the normalized b=0 prediction.


def make_screen(protocol,method,D_fixed=None,R_bounds=R_BOUNDS,D_bounds=D_BOUNDS):
    bounds={'R':R_bounds,'D':D_bounds} if D_fixed is None else {'R':R_bounds}
    grid=list(itertools.product(R_STARTS,D_STARTS)) if D_fixed is None else [[R] for R in R_STARTS]
    def forward(theta):return fit_forward(theta,protocol,method,D_fixed)
    return prepare_weighted_screen(forward,bounds,grid)


def fit_radius(means,counts,protocol,method='M0-Strang',D_fixed=None,S0_initial=None,
               screen=None,R_bounds=R_BOUNDS,D_bounds=D_BOUNDS,optimizer=None):
    """Estimate R or R,D; S0 is profiled, including the measured b=0 group.

    Pass a previously computed screen to reuse its10/50 physical predictions.
    Defaults are the definitive manuscript settings, not a new fitting method.
    """
    bounds={'R':R_bounds,'D':D_bounds} if D_fixed is None else {'R':R_bounds}
    def forward(theta):return fit_forward(theta,protocol,method,D_fixed)
    if screen is None:screen=make_screen(protocol,method,D_fixed,R_bounds,D_bounds)
    data=dict(means=np.asarray(means),counts=np.asarray(counts),
              S0_initial=float(means[-1]) if S0_initial is None else float(S0_initial))
    return fit_dataset(data,forward,screen,bounds,OPTIMIZER if optimizer is None else optimizer,method)


def fit_task(arguments):
    """Small adapter for multiprocessing.Pool.map; arguments are explicit arrays."""
    return fit_radius(**arguments)


def load_experiment(filename):
    with np.load(filename,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    data['conditions']=json.loads(str(data['conditions_json']))
    data['fem_jobs']=json.loads(str(data['fem_jobs_json']))
    return data


def observations(data,case,snr,rep=0):
    n=5 if data['protocols'][case]=='P1' else 8
    if snr==0:return data['noiseless'][case,:n+1],data['counts'][case,:n+1],1.
    j=list(data['snrs']).index(snr)
    return data['means'][case,j,rep,:n+1],data['counts'][case,:n+1],float(data['S0_initial'][case,j,rep])


def fit_row(result,condition,case_id,estimator,snr,rep):
    """One flat result row; no per-fit files or run-record architecture."""
    c=condition;p=result['parameters'];R=float(c.get('Rc',c.get('R')));D=float(c.get('Dc',c.get('D')))
    return dict(**c,configuration=case_id,estimator=estimator,snr=snr,rep=rep,
        R_hat=p['R'],D_hat=p.get('D',D),S0_hat=p['S0'],e_R_pct=100*(p['R']/R-1),
        objective=result['objective'],success=result['success'],R_bound=result['boundary_hits']['R'],
        D_bound=result['boundary_hits'].get('D',False),seconds=result['seconds'],
        unsuccessful_starts=result['unsuccessful_starts'],forward_evaluations=result['forward_evaluations'],
        effective_starts=result['effective_starts'],screen_violation=result['successful_but_screen_better'],
        near_minimum_scaled_parameter_span=result['near_minimum_scaled_parameter_span'],
        optimizer_guard_retry='original_attempt' in result or 'guard_retry_attempt' in result,
        competing_near_minima=result['competing_near_minima'],
        unresolved_optimizer_guard=result.get('unresolved_optimizer_guard',False))


def metrics(frame):
    """Signed percent bias, percent RMSE, mean, sample SD and bound/failure rates."""
    f=frame.rename(columns={'Rhat':'R_hat','Dhat':'D_hat','S0hat':'S0_hat','realization':'rep',
        'boundary_R':'R_bound','boundary_D':'D_bound','model':'estimator'}).copy()
    if 'Rc' not in f:f['Rc']=f.R;f['Dc']=f.D
    if 'configuration' not in f:f['configuration']=f.apply(lambda r:f'{r.protocol}_D{r.Dc:g}_R{r.Rc:g}',axis=1)
    rows=[]
    for (case,method,snr),g in f.groupby(['configuration','estimator','snr']):
        row=dict(configuration=case,protocol=g.protocol.iloc[0],estimator=method,snr=snr,n=len(g))
        for parameter,true in [('R','Rc'),('D','Dc')]:
            x=g[parameter+'_hat'].to_numpy();error=100*(x/g[true].to_numpy()-1)
            row.update({parameter+'_mean':float(x.mean()),parameter+'_SD':float(x.std(ddof=1)) if len(x)>1 else np.nan,
                parameter+'_bias_pct':float(error.mean()),parameter+'_RMSE_pct':float(np.sqrt(np.mean(error**2))),
                parameter+'_bound_frequency':float(g[parameter+'_bound'].mean())})
        row.update(optimizer_failure_frequency=float((~g.success.astype(bool)).mean()),WLS_mean=float(g.objective.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def add_eccentricity_contrasts(frame):
    """Pair each M2 estimate with its concentric case at the SAME repetition."""
    keys=['protocol','ratio','estimator','snr','rep']
    base=frame[frame.offset_index==0][keys+['e_R_pct','R_hat','D_hat']]
    out=frame.merge(base,on=keys,suffixes=('','_concentric'),validate='many_to_one')
    out['delta_e_R_pp']=out.e_R_pct-out.e_R_pct_concentric
    out['delta_R_um']=out.R_hat-out.R_hat_concentric
    out['delta_D_um2_ms']=out.D_hat-out.D_hat_concentric
    return out


def generate_observations(signal,protocol_name,condition,snr,repetitions=50,campaign='M1'):
    """Original Rician/noise seeds for the definitive cases; sigma=1/SNR.

    Existing experiments load their saved observations. This function is for
    explicit physical regeneration; it never runs during default figure replay.
    """
    c=condition;counts=signals.published_protocol(protocol_name)['counts'];index=np.repeat(np.arange(len(counts)),counts)
    R=float(c.get('Rc',c.get('R')));D=float(c.get('Dc',c.get('D')))
    geometry=dict(protocol=protocol_name,Rc=R,eta=float(c.get('eta',c.get('Rn',R*(.6 if protocol_name=='P1' else .8))/R)),Dc=D)
    key=content_hash(geometry);entropy=[int(key[j:j+8],16) for j in range(0,32,8)]
    dwi=[];b0=[]
    for rep in range(repetitions):
        if campaign=='M0' and R in [3,5,7,9,11,13,15] and D in [2.,.6] and snr in [20,50,100]:
            stream=[['P1','P2'].index(protocol_name),[2.,.6].index(D),[3,5,7,9,11,13,15].index(R),[20,50,100].index(snr),rep]
            rng=seeded_rng(20261103,*stream);rng0=seeded_rng(20261103,1947001,*stream)
        else:
            rng=seeded_rng(20261112,*entropy,1,rep);rng0=seeded_rng(20261112,*entropy,1947001,rep)
        dwi.append(rician_noise(signal[index],snr,rng));b0.append(rician_noise(np.ones(len(counts)),snr,rng0))
    dwi=np.asarray(dwi);b0=np.asarray(b0)
    means=np.c_[np.column_stack([dwi[:,index==j].mean(axis=1) for j in range(len(counts))]),b0.mean(axis=1)]
    return dict(dwi=dwi,b0=b0,means=means,counts=np.r_[counts,len(counts)],S0_initial=b0.mean(axis=1))
