"""Intracellular finite-PGSE signals. Ordinary functions, no package install.

Main functions: pgse_protocol, published_protocol, m0_signal (real expm),
m0_strang_signal (real Numba), gpa_m0_signal, m1_signal (real expm).
Units: radii um, D um²/ms, timings ms, b s/mm², G T/m, public M1 kappa um/s.

The numerical function bodies are the validated research implementations.
They are grouped here rather than imported through a hierarchy of packages.
M0/M1 are Bessel spectral models. Actual 3D FEM is in signals_fem.py.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn, spherical_jn as jn, spherical_yn as yn
from scipy.optimize import brentq
from scipy.linalg import expm, eigh
from numba import njit

GAMMA_RAD_MS_UM_T_M=0.26752218744
GAMMA_RAD_S_T=GAMMA_RAD_MS_UM_T_M*1e9
GAMMA=GAMMA_RAD_MS_UM_T_M


# PGSE acquisition and the two manuscript protocols
# ------------------------------------------------------------------------
def pgse_protocol(b_s_mm2, delta_ms, Delta_ms, *, Gmax_T_m=None):
    """Broadcast b/delta/Delta into a complete acquisition protocol dictionary.
    b=1000*(gamma*G*delta)^2*(Delta-delta/3) in these units.
    Optional Gmax checks amplitudes only, not slew rate, RF timing or TE.
    """
    inputs=[np.asarray(x,dtype=float) for x in (b_s_mm2,delta_ms,Delta_ms)]
    if any(x.ndim>1 for x in inputs):raise ValueError('protocol inputs must be scalars or 1D arrays')
    b,de,De=[np.atleast_1d(x).copy() for x in np.broadcast_arrays(*inputs)]
    if not len(b) or any(not np.all(np.isfinite(x)) for x in (b,de,De)):
        raise ValueError('nonempty finite acquisitions required')
    if np.any(b<0) or np.any(de<=0) or np.any(De<de):raise ValueError('require b>=0, delta>0 and Delta>=delta')
    q=np.sqrt((b/1000)/(De-de/3));G=q/(GAMMA_RAD_MS_UM_T_M*de)
    if Gmax_T_m is not None:
        if not np.isfinite(Gmax_T_m) or Gmax_T_m<=0:raise ValueError('Gmax must be positive and finite')
        if np.any(G>Gmax_T_m*(1+1e-12)):raise ValueError('requested b/timings exceed Gmax')
    out=dict(b_s_mm2=b,delta_ms=de,Delta_ms=De,q_rad_um=q,G_T_m=G)
    for x in out.values():x.setflags(write=False)
    return out


def _protocol(protocol):
    """Recompute derived values from authoritative b/timing fields, no stale q/G."""
    return pgse_protocol(protocol['b_s_mm2'],protocol['delta_ms'],protocol['Delta_ms'])


def protocol_record(protocol):
    p=_protocol(protocol)
    return {k:p[k].tolist() for k in ['b_s_mm2','delta_ms','Delta_ms']}


# Manuscript acquisition values (counts are independent repeats)
# ------------------------------------------------------------------------
def published_protocol(name):
    """Unique DW acquisitions. P2 rounded G/timings, not rounded b, are primary."""
    if name == 'P1':
        protocol = pgse_protocol([800,1500,3450,4750,6000],6.,13.,Gmax_T_m=.5)
        counts = [32,32,64,64,64]
        metadata = dict(label='SANDI Connectome 2.0', doi='10.1002/mrm.70563',
                        locator='Table 2; Section 3.2.3', TE_ms=53., Gmax_mT_m=500.,
                        primary='b, delta, Delta', published_b=[800,1500,3450,4750,6000])
    elif name == 'P2':
        G = np.array([27,154,36,38,172,141,165,79])/1000
        de = np.array([10.2,5.1,19.6,23.4,5.1,11.,5.4,20.7])
        De = np.array([34.5,8.5,32.5,29.5,28.2,14.2,47.7,30.7])
        b = 1000*(GAMMA_RAD_MS_UM_T_M*G*de)**2*(De-de/3)
        protocol = pgse_protocol(b,de,De,Gmax_T_m=.18)
        counts = [32]*8
        metadata = dict(label='POLAR-CEXI', doi='10.48550/arXiv.2610.28099',
                        locator='v1, Table 1 CEXI entry', TE_ms=63.1, Gmax_mT_m=180.,
                        primary='rounded published G, delta, Delta',
                        published_b=[173,302,910,1235,1460,1803,2618,4558],
                        direction_count_scope='32 per acquisition is our design, not POLAR')
    else:
        raise ValueError('Protocol must be P1 or P2')
    return dict(name=name, protocol=protocol, counts=np.array(counts,dtype=int), metadata=metadata)


# M0: dimensionless Neumann sphere eigenfunctions and coordinate matrix
# ------------------------------------------------------------------------
def assemble(blocks, label):
    sizes = [len(b[0]) for b in blocks]
    offsets = np.cumsum([0] + sizes)
    rates = np.concatenate([b[0] for b in blocks])
    z = np.zeros((len(rates), len(rates)))
    for ell in range(len(blocks)-1):
        _, u, r, w = blocks[ell]
        _, v, _, _ = blocks[ell+1]
        angular = (ell+1)/np.sqrt((2*ell+1)*(2*ell+3))
        cross = angular * u.T @ ((w*r**3)[:, None] * v)
        sl = slice(offsets[ell], offsets[ell+1])
        sr = slice(offsets[ell+1], offsets[ell+2])
        z[sl, sr], z[sr, sl] = cross, cross.T
    val, u, r, w = blocks[0]
    initial = np.zeros(len(rates))
    initial[:len(val)] = np.sqrt(3)*((w*r*r) @ u)
    initial[rates > 1e-7] = 0
    initial /= np.linalg.norm(initial)
    val1, u1, r, w = blocks[1]
    weights = ((w*r**3) @ u1)**2
    return dict(rates=rates, z=z, equilibrium=initial, dipole_rates=val1,
                dipole_weights=weights, label=label, radial_blocks=blocks)


def neumann_roots(ell, count):
    roots = []
    x = 1e-4
    previous = spherical_jn(ell, x, derivative=True)
    while len(roots) < count:
        y = x+.12
        current = spherical_jn(ell, y, derivative=True)
        if current*previous < 0:
            roots.append(brentq(lambda t: spherical_jn(ell, t, derivative=True), x, y))
        x, previous = y, current
    return np.array(roots)


@lru_cache(maxsize=4)
def ball_basis(modes=9, lmax=7):
    gx, gw = leggauss(240)
    r, w = (gx+1)/2, gw/2
    blocks = []
    for ell in range(lmax+1):
        roots = neumann_roots(ell, modes-(ell == 0))
        if ell == 0:
            roots = np.r_[0., roots]
        u = spherical_jn(ell, np.outer(r, roots))
        u /= np.sqrt((w*r*r) @ (u*u))[None, :]
        blocks.append((roots*roots, u, r, w))
    return assemble(blocks, 'analytic Neumann ball')


# M1: concentric Bessel transmission spectrum, including all zero modes
# ------------------------------------------------------------------------
@lru_cache(maxsize=512)
def neumann_ball_roots(ell,count):
    left=max(.001,.8*np.sqrt(ell*(ell+1)))
    stop=(count+ell+4)*np.pi
    grid=np.arange(left,stop,.06)
    values=jn(ell,grid,True)
    intervals=np.flatnonzero(values[:-1]*values[1:]<0)
    roots=[brentq(lambda x:jn(ell,x,True),grid[i],grid[i+1],xtol=2e-13)
           for i in intervals[:count]]
    if len(roots)!=count:raise ArithmeticError('Insufficient ball roots')
    return np.array(roots)


def shell_value(ell,k,r,derivative=False):
    if derivative:
        return k*(jn(ell,k*r,True)*yn(ell,k,True)
                  -yn(ell,k*r,True)*jn(ell,k,True))
    return jn(ell,k*r)*yn(ell,k,True)-yn(ell,k*r)*jn(ell,k,True)


@lru_cache(maxsize=2048)
def neumann_shell_roots(ell,eta,count):
    left=max(.001,.8*np.sqrt(ell*(ell+1)))
    stop=(count+ell+4)*np.pi/(1-eta)
    grid=np.arange(left,stop,.06)
    def characteristic(k):
        ja,ya=jn(ell,k*eta,True),yn(ell,k*eta,True)
        jb,yb=jn(ell,k,True),yn(ell,k,True)
        return (ja/np.hypot(ja,ya))*(yb/np.hypot(jb,yb))-(ya/np.hypot(ja,ya))*(jb/np.hypot(jb,yb))
    values=characteristic(grid)
    intervals=np.flatnonzero(values[:-1]*values[1:]<0)
    roots=[brentq(characteristic,grid[i],grid[i+1],xtol=2e-13)
           for i in intervals[:count]]
    if len(roots)!=count:raise ArithmeticError('Insufficient shell roots')
    return np.array(roots)


def uncoupled_spectrum(ell,eta,d,count):
    modes=[]
    if ell==0:modes=[(0.,'inner'),(0.,'shell')]
    modes += [(float(d*(x/eta)**2),'inner') for x in neumann_ball_roots(ell,count)]
    modes += [(float(x*x),'shell') for x in neumann_shell_roots(ell,eta,count)]
    return sorted(modes)


@lru_cache(maxsize=4096)
def transmission_roots(ell,eta,d,h,count):
    modes=uncoupled_spectrum(ell,eta,d,count+2)
    if h==0:return modes[:count]
    # A positive interface energy is a rank-one radial form perturbation.
    # Its roots interlace the union of the two isolated Neumann spectra.
    # Coincident poles leave one interface-continuous eigenfunction unchanged.
    poles=[];multiplicity=[]
    for value,region in modes:
        if poles and abs(value-poles[-1])<1e-10*max(1,value):
            multiplicity[-1]+=1
        else:poles.append(value);multiplicity.append(1)
    def characteristic(lam):
        k=np.sqrt(lam);kn=k/np.sqrt(d)
        hn=jn(ell,kn*eta)/(d*kn*jn(ell,kn*eta,True))
        hc=shell_value(ell,k,eta)/shell_value(ell,k,eta,True)
        return hn-hc+(0. if np.isinf(h) else 1/h)
    roots=[]
    for i in range(len(poles)-1):
        low,high=poles[i:i+2]
        if multiplicity[i]>1:
            roots += [(low,'global' if low==0 else 'coupled')]*(multiplicity[i]-1)
        width=high-low
        eps=1e-10*max(1.,abs(low),abs(high),width)
        if low==0 and np.isfinite(h):
            leading=3*h*(1/eta+eta**2/(1-eta**3))
            eps=min(eps,max(1e-16,.01*leading))
        left,right=low+eps,high-eps
        fl,fr=characteristic(left),characteristic(right)
        if fl*fr>=0:
            # At small eta/high ell an interface trace can be so small that
            # its root shift is below the resolution of a pole. Monotonicity
            # localizes that root to the endpoint probe interval. Keep the
            # pole eigenvalue, and label this floating-point approximation.
            left_probe=low+eps;right_probe=high-eps
            fp,fq=characteristic(left_probe),characteristic(right_probe)
            if fp>0 and fq>0:
                root,kind=low,'coupled_rounded_lower'
            elif fp<0 and fq<0:
                root,kind=high,'coupled_rounded_upper'
            else:
                raise ArithmeticError(f'Interlacing bracket failed ell={ell}, h={h}, poles={low,high}, F={fl,fr}')
        else:
            root=brentq(characteristic,left,right,xtol=1e-25,rtol=8e-15)
            kind='coupled'
        roots.append((float(root),kind))
        if len(roots)>=count:break
    return sorted(roots)[:count]


def analytic_radial(ell,eta,d,h,count,quadrature=180):
    gx,gw=leggauss(quadrature)
    rn=eta*(gx+1)/2;wn=eta*gw/2
    rc=eta+(1-eta)*(gx+1)/2;wc=(1-eta)*gw/2
    r=np.r_[rn,rc];w=np.r_[wn,wc]
    roots=transmission_roots(ell,eta,d,h,count)
    u=np.zeros((len(r),len(roots)))
    for index,(lam,region) in enumerate(roots):
        if lam==0:
            if region=='global':u[:,index]=1
            elif region=='inner':u[:quadrature,index]=1
            else:u[quadrature:,index]=1
        else:
            k=np.sqrt(lam);kn=k/np.sqrt(d)
            inner=jn(ell,kn*rn)
            if region=='inner':u[:quadrature,index]=inner
            elif region=='shell':
                # Start at the inner Neumann boundary. Starting at the outer
                # boundary amplifies roundoff through y_l(k*eta) for tiny eta.
                b=yn(ell,k*eta,True);c=-jn(ell,k*eta,True)
                scale=np.hypot(b,c);b/=scale;c/=scale
                u[quadrature:,index]=b*jn(ell,k*rc)+c*yn(ell,k*rc)
            else:
                ja=jn(ell,kn*eta);jp=d*kn*jn(ell,kn*eta,True)
                if np.isfinite(h) and abs(jp)<.01*abs(ja):
                    # A core-dominated weakly coupled mode is ill conditioned
                    # when propagated outward: h*ja+jp nearly cancels. Build
                    # its small shell amplitude from the outer Neumann law.
                    va=shell_value(ell,k,eta);vp=shell_value(ell,k,eta,True)
                    ratio=-h*ja/(vp-h*va)
                    u[:quadrature,index]=inner
                    u[quadrature:,index]=ratio*shell_value(ell,k,rc)
                    norm=np.sqrt(np.dot(w*r*r,u[:,index]**2))
                    u[:,index]/=norm
                    if np.sum(u[:,index])<0:u[:,index]*=-1
                    continue
                if np.isinf(h):inner_scale=1.;value=ja;derivative=jp
                else:inner_scale=h;value=h*ja+jp;derivative=h*jp
                # Exact spherical-Bessel Wronskian at the interface.
                wronskian=1/(k*eta**2)
                b=(value*k*yn(ell,k*eta,True)-derivative*yn(ell,k*eta))/wronskian
                c=(derivative*jn(ell,k*eta)-value*k*jn(ell,k*eta,True))/wronskian
                u[:quadrature,index]=inner_scale*inner
                u[quadrature:,index]=b*jn(ell,k*rc)+c*yn(ell,k*rc)
        norm=np.sqrt(np.dot(w*r*r,u[:,index]**2))
        u[:,index]/=norm
        if np.sum(u[:,index])<0:u[:,index]*=-1
    val=np.array([x[0] for x in roots])
    gram=u.T@((w*r*r)[:,None]*u)
    if np.max(abs(gram-np.eye(len(roots))))>2e-7:
        raise ArithmeticError(f'Analytic orthogonality failed ell={ell}, eta={eta}, h={h}: {np.max(abs(gram-np.eye(len(roots))))}')
    return val,u,r,w


def basis(eta,d,h,radial_modes=18,lmax=14,quadrature=180):
    blocks=[analytic_radial(ell,eta,d,h,radial_modes,quadrature) for ell in range(lmax+1)]
    val=np.concatenate([x[0] for x in blocks]);n=len(val)
    z=np.zeros((n,n));offset=np.cumsum([0]+[len(x[0]) for x in blocks])
    for ell in range(lmax):
        _,a,r,w=blocks[ell];_,b,_,_=blocks[ell+1]
        coupling=(ell+1)/np.sqrt((2*ell+1)*(2*ell+3))*a.T@((w*r**3)[:,None]*b)
        si=slice(offset[ell],offset[ell+1]);sj=slice(offset[ell+1],offset[ell+2])
        z[si,sj]=coupling;z[sj,si]=coupling.T
    initial=np.zeros(n)
    nr0=len(blocks[0][0])
    _,u,r,w=blocks[0]
    initial[:nr0]=np.sqrt(3)*((w*r*r)@u)
    initial[val!=0]=0
    initial/=np.linalg.norm(initial)
    rounded=sum('rounded' in kind for ell in range(lmax+1)
        for _,kind in transmission_roots(ell,eta,d,h,radial_modes))
    return dict(rates=val,z=z,initial=initial,blocks=blocks,rounded_roots=rounded,
        solver='analytic_Bessel_transmission_Grebenkov_reimplementation',eta=eta,d=d,h=h)


# Dimensionless spatial matrices and the real angular-parity representation
# ------------------------------------------------------------------------
def model_name(model):
    name=str(model).upper()
    if name not in ['M0','M1']:raise ValueError('model must be M0 or M1')
    return name


def parameters(model,values):
    names=['R','D'] if model_name(model)=='M0' else ['R','a','Dn','Dc','kappa']
    if set(values)!=set(names):raise ValueError('parameter names must be '+', '.join(names))
    p={k:float(values[k]) for k in names}
    for k,v in p.items():
        if k=='kappa' and v==np.inf:continue
        if not np.isfinite(v):raise ValueError('parameters must be finite (kappa may be +inf)')
    if p['R']<=0:raise ValueError('R must be positive')
    if model_name(model)=='M0':
        if p['D']<=0:raise ValueError('D must be positive')
    elif not (0<=p['a']<p['R'] and p['Dn']>0 and p['Dc']>0 and p['kappa']>=0):
        raise ValueError('M1 requires 0<=a<R, Dn,Dc>0, kappa>=0')
    return p


def resolution(radial_modes,lmax,model):
    if int(radial_modes)!=radial_modes or int(lmax)!=lmax:raise ValueError('integer spatial resolutions required')
    nr,L=int(radial_modes),int(lmax)
    if nr<(2 if model_name(model)=='M1' else 1) or L<1:
        raise ValueError('require lmax>=1; M1 needs at least two radial modes for sealed zero modes')
    return nr,L


def matrix_key(model,p,radial_modes,lmax,quadrature=240):
    model=model_name(model);p=parameters(model,p);nr,L=resolution(radial_modes,lmax,model)
    if model=='M0' or p['a']==0:return ('M0',nr,L)
    if int(quadrature)!=quadrature or quadrature<32:raise ValueError('quadrature must be an integer >=32')
    eta=p['a']/p['R'];d=p['Dn']/p['Dc'];h=p['kappa']*p['R']/p['Dc']
    if not 0<eta<1 or not np.isfinite(d) or d<=0:raise ValueError('dimensionless geometry/diffusivity outside floating-point range')
    if np.isfinite(p['kappa']) and (not np.isfinite(h) or (p['kappa']>0 and h==0)):
        raise ValueError('dimensionless permeability outside floating-point range')
    return ('M1',eta,d,h,nr,L,int(quadrature))


@lru_cache(maxsize=12)
def spatial_matrices(key):
    if key[0]=='M0':
        _,nr,L=key;b=ball_basis(nr,L);v=b['equilibrium'];rounded=0
    else:
        _,eta,d,h,nr,L,quad=key
        b=basis(eta,d,h,nr,L,quad);v=b['initial'];rounded=b['rounded_roots']
    rates=b['rates'];z=b['z'];ell=np.repeat(np.arange(L+1),nr)
    even=np.flatnonzero(ell%2==0);odd=np.flatnonzero(ell%2==1)
    K=np.zeros_like(z);K[np.ix_(even,odd)]=z[np.ix_(even,odd)]
    K[np.ix_(odd,even)]=-z[np.ix_(odd,even)]
    if any(not np.all(np.isfinite(a)) for a in [rates,z,v]) or np.min(rates)<0 or np.linalg.norm(rates*v)>1e-11 or abs(v@v-1)>1e-10:
        raise ArithmeticError('invalid or unresolved equilibrium diffusion generator')
    out=dict(rates=rates,z=z,v=v,K=K,ell=ell,even=even,odd=odd,
             zero_modes=int(np.count_nonzero(rates==0)),rounded_roots=rounded,raw_basis=b)
    # These are cached internal arrays. Public diagnostics return copies/lists.
    for a in [rates,z,v,K,ell,even,odd]:a.setflags(write=False)
    return out


@lru_cache(maxsize=12)
def phase_matrices(key):
    p=spatial_matrices(key);ev,V=eigh(p['z'],check_finite=False,driver='evr');pos=np.flatnonzero(ev>1e-12)
    return dict(sigma=ev[pos],E=np.sqrt(2)*V[np.ix_(p['even'],pos)],
                O=np.sqrt(2)*V[np.ix_(p['odd'],pos)])


# Finite rectangular pulses: real expm and the original Strang recurrence
# ------------------------------------------------------------------------
def _fixed_strang_state(p,phase,a,beta,n):
    # Identical real Strang recurrence to validated sphere.state_strang(real=True).
    half=np.exp(-a*p['rates']/(2*n));full=half*half;x=half*p['v']
    co=np.cos(beta*phase['sigma']/n)-1;si=np.sin(beta*phase['sigma']/n)
    E,O=phase['E'],phase['O'];e,o=p['even'],p['odd']
    for k in range(n):
        u=E.T@x[e];w=O.T@x[o]
        y=x.copy();y[e]+=E@(co*u+si*w);y[o]+=O@(-si*u+co*w)
        x=y*(half if k==n-1 else full)
    return x


def _calibrated_steps(calibration,model,p,protocol,nr,L,quadrature):
    if calibration is None:raise ValueError('Strang requires an offline fixed-step calibration')
    if calibration.get('calibration_format')!=1 or calibration.get('gamma_rad_s_T')!=GAMMA_RAD_S_T:
        raise ValueError('calibration format/gyromagnetic-ratio mismatch')
    if calibration.get('model')!=model or calibration.get('resolution')!=[nr,L]:raise ValueError('calibration model/resolution mismatch')
    if model=='M1' and calibration.get('quadrature')!=quadrature:raise ValueError('calibration quadrature mismatch')
    if calibration.get('protocol')!=protocol_record(protocol):raise ValueError('calibration protocol mismatch')
    if set(calibration.get('parameter_region',{}))!=set(p):raise ValueError('calibration region parameter mismatch')
    for name,value in p.items():
        lo,hi=map(float,calibration['parameter_region'][name])
        if not lo<=value<=hi:raise ValueError(f'{name} is outside the calibrated region')
    steps=np.asarray(calibration['subdivisions'],dtype=float)
    if steps.shape!=protocol['b_s_mm2'].shape or not np.all(np.isfinite(steps)) or np.any(steps!=np.floor(steps)):
        raise ValueError('one fixed integer subdivision count per waveform required')
    if np.any(steps<0) or np.any(steps[protocol['b_s_mm2']>0]<1):raise ValueError('positive steps required for encoded waveforms')
    return steps.astype(int)


def _evaluate(model,p,protocol,radial_modes,lmax,quadrature=240,backend='expm',calibration=None):
    model=model_name(model);p=parameters(model,p);protocol=_protocol(protocol)
    if backend not in ['expm','strang']:raise ValueError('production backends are expm or fixed real Strang')
    key=matrix_key(model,p,radial_modes,lmax,quadrature);m=spatial_matrices(key)
    Dc=p['D'] if model=='M0' else p['Dc'];R=p['R'];rate_scale=Dc/R**2
    steps=None;phase=None
    if backend=='strang':
        steps=_calibrated_steps(calibration,model,p,protocol,int(radial_modes),int(lmax),quadrature)
        phase=phase_matrices(key)
    output=[];pulse_states={}
    for j,(b,de,De,q) in enumerate(zip(protocol['b_s_mm2'],protocol['delta_ms'],protocol['Delta_ms'],protocol['q_rad_um'])):
        if b==0:output.append(1.);continue
        # Reuse a pulse only for identical de,q and subdivision counts within this protocol.
        pulse_key=(float(de),float(q),None if steps is None else int(steps[j]))
        if pulse_key not in pulse_states:
            a=de*rate_scale;beta=q*R
            if backend=='expm':x=expm(beta*m['K']-np.diag(a*m['rates']))@m['v']
            else:x=_fixed_strang_state(m,phase,a,beta,int(steps[j]))
            pulse_states[pulse_key]=x
        x=pulse_states[pulse_key];gap=np.exp(-(De-de)*rate_scale*m['rates'])
        output.append(float(np.dot(x,gap*x)))
    return np.array(output)


def m0_signal(R,D,protocol,*,radial_modes=10,lmax=14,backend='expm',calibration=None):
    """M0: R in um, D in um^2/ms. Default N=150, subject to convergence checks."""
    return _evaluate('M0',dict(R=R,D=D),protocol,radial_modes,lmax,backend=backend,calibration=calibration)


def _m1_signal_um_ms(R,a,Dn,Dc,kappa,protocol,*,radial_modes=12,lmax=14,quadrature=240,backend='expm',calibration=None):
    """Concentric M1. R,a in um; Dn,Dc in um^2/ms; physical kappa in um/ms.
    Recomputes eta=a/R, d=Dn/Dc and h=kappa*R/Dc for every call.
    kappa=0 keeps both stationary modes; kappa=+inf is the transparent interface.
    a=0 is the exact M0(Dc) endpoint. Equal diffusivities alone do not remove a barrier.
    """
    return _evaluate('M1',dict(R=R,a=a,Dn=Dn,Dc=Dc,kappa=kappa),protocol,radial_modes,lmax,quadrature,backend,calibration)


def spectrum_info(model,values,*,radial_modes=None,lmax=14,quadrature=240):
    """Small internal diagnostics. values['kappa'] is um/ms, as in the kernel.

    The public m1_signal takes um/s and performs the conversion explicitly.
    No writable cached matrices are exposed.
    """
    model=model_name(model);p=parameters(model,values)
    if radial_modes is None:radial_modes=10 if model=='M0' else 12
    key=matrix_key(model,p,radial_modes,lmax,quadrature);m=spatial_matrices(key)
    out=dict(model=model,dimension=len(m['rates']),zero_modes=m['zero_modes'],rounded_roots=m['rounded_roots'],
        equilibrium_norm=float(m['v']@m['v']),generator_equilibrium_norm=float(np.linalg.norm(m['rates']*m['v'])),
        zero_mode_weights=(m['v'][m['rates']==0]**2).tolist())
    if model=='M1':out.update(eta=p['a']/p['R'],d=p['Dn']/p['Dc'],h=p['kappa']*p['R']/p['Dc'])
    return out


# Accelerated real Strang: identical compiled matrix-vector recurrence
# ------------------------------------------------------------------------
@njit(cache=True,fastmath=False)
def _state(rates,v,even,odd,E,ET,O,OT,sigma,a,beta,n):
    half=np.exp(-a*rates/(2*n));full=half*half;x=half*v
    co=np.cos(beta*sigma/n)-1.;si=np.sin(beta*sigma/n)
    ev=np.empty(len(even));od=np.empty(len(odd));u=np.empty(len(sigma));w=np.empty(len(sigma))
    z=np.empty(len(sigma));y=np.empty(len(sigma));de=np.empty(len(even));do=np.empty(len(odd))
    for k in range(n):
        for i in range(len(even)):ev[i]=x[even[i]]
        for i in range(len(odd)):od[i]=x[odd[i]]
        np.dot(ET,ev,u);np.dot(OT,od,w)
        for j in range(len(sigma)):
            z[j]=co[j]*u[j]+si[j]*w[j];y[j]=-si[j]*u[j]+co[j]*w[j]
        np.dot(E,z,de);np.dot(O,y,do)
        for i in range(len(even)):x[even[i]]+=de[i]
        for i in range(len(odd)):x[odd[i]]+=do[i]
        factor=half if k==n-1 else full
        for i in range(len(x)):x[i]*=factor[i]
    return x


@lru_cache(maxsize=12)
def _prepared(nr,L):
    key=('M0',nr,L);p=spatial_matrices(key);phase=phase_matrices(key)
    E=np.ascontiguousarray(phase['E']);O=np.ascontiguousarray(phase['O'])
    return p,phase,E,np.ascontiguousarray(E.T),O,np.ascontiguousarray(O.T)


def m0_signal_numba(R,D,protocol,*,radial_modes=10,lmax=14,calibration):
    """Same complete-protocol M0 Strang signal, with a compiled real recurrence.

    Inputs and offline fixed-step calibration have the unchanged forward
    package's units and validation. Compilation/loading is a one-time per-process
    cost; warm the function before measuring repeated evaluations or fitting.
    """
    values=parameters('M0',dict(R=R,D=D));protocol=_protocol(protocol)
    key=matrix_key('M0',values,radial_modes,lmax);nr,L=key[1:]
    steps=_calibrated_steps(calibration,'M0',values,protocol,nr,L,240)
    p,phase,E,ET,O,OT=_prepared(nr,L);rate_scale=values['D']/values['R']**2
    output=[];states={}
    for j,(b,de,De,q) in enumerate(zip(protocol['b_s_mm2'],protocol['delta_ms'],protocol['Delta_ms'],protocol['q_rad_um'])):
        if b==0:output.append(1.);continue
        pulse=(float(de),float(q),int(steps[j]))
        if pulse not in states:
            states[pulse]=_state(p['rates'],p['v'],p['even'],p['odd'],E,ET,O,OT,phase['sigma'],de*rate_scale,q*values['R'],int(steps[j]))
        x=states[pulse];gap=np.exp(-(De-de)*rate_scale*p['rates'])
        output.append(float(np.dot(x,gap*x)))
    return np.asarray(output)


# Spherical Gaussian-phase approximation: separate physical model
# ------------------------------------------------------------------------
@lru_cache(maxsize=8)
def dipole_spectrum(terms=64):
    alpha=neumann_roots(1,terms)
    return alpha*alpha,2/(alpha*alpha*(alpha*alpha-2))


def signal_gpa(R,D,delta,Delta,G,terms=64):
    if R<=0 or D<=0 or delta<=0 or Delta<delta:raise ValueError('invalid physical parameters')
    if G==0:return 1.
    roots2,w=dipole_spectrum(terms);lam=D*roots2/R**2
    x=lam*delta;y=lam*(Delta-delta)
    f=-np.expm1(-x)/x
    # 2*g(x)-f(x)^2, g=(x-1+exp(-x))/x^2. Avoid cancellation at small x.
    h=np.empty_like(x);small=x<1e-3;xs=x[small]
    h[small]=xs*(2/3+xs*(-.5+xs*(7/30+xs*(-1/12+xs*31/1260))))
    h[~small]=2*(x[~small]+np.expm1(-x[~small]))/x[~small]**2-f[~small]**2
    I=2*delta**2*(h-f*f*np.expm1(-y))
    return float(np.exp(-.5*(GAMMA*G)**2*R**2*np.dot(w,I)))


def gpa_m0_signal(R,D,protocol,*,terms=64):
    """Sphere GPA, a separate physical approximation, with the common PGSE interface."""
    p=parameters('M0',dict(R=R,D=D));acq=_protocol(protocol)
    if int(terms)!=terms or terms<1:raise ValueError('positive integer GPA terms required')
    return np.array([signal_gpa(p['R'],p['D'],de,De,G,int(terms))
        for de,De,G in zip(acq['delta_ms'],acq['Delta_ms'],acq['G_T_m'])])


# Simple public entry points for the manuscript's conventions.
def m0_strang_signal(R,D,protocol,subdivisions=64,radial_modes=10,lmax=14):
    """Fixed steps PER PULSE; changing R/D never adapts the time discretization.

    150 modes and64 steps are the manuscript settings. Other parameter regions
    need comparison with real expm; this function does not certify their error.
    The small record only supplies the unchanged kernel's required interface.
    """
    steps=np.broadcast_to(np.asarray(subdivisions),len(protocol['b_s_mm2'])).tolist()
    settings=dict(calibration_format=1,gamma_rad_s_T=GAMMA_RAD_S_T,model='M0',
        resolution=[radial_modes,lmax],quadrature=None,
        parameter_region={'R':[0.,np.inf],'D':[0.,np.inf]},
        protocol=protocol_record(protocol),subdivisions=steps)
    return m0_signal_numba(R,D,protocol,radial_modes=radial_modes,lmax=lmax,calibration=settings)


def m1_signal(R,a,Dn,Dc,kappa_um_s,protocol,radial_modes=24,lmax=24,quadrature=240):
    """Concentric M1. Physical kappa is um/s, converted explicitly to um/ms.

    kappa=0 retains both sealed stationary modes; infinity is the exact
    transparent-interface limit. Default24*(24+1)=600 spatial modes.
    """
    return _m1_signal_um_ms(R,a,Dn,Dc,float(kappa_um_s)/1000.,protocol,
        radial_modes=radial_modes,lmax=lmax,quadrature=quadrature)
