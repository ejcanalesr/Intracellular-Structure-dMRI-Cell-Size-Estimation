function signals_fem(solver,tag)
% Original validated MATLAB drivers collected in one file; no PDE changes.
% Work directory and pinned SpinDoctor checkout are explicit environment inputs.
switch solver
 case 'mesh_M2',build_meshes();
 case 'mesh_M3',build_mesh_stage2(tag);
 case 'run08',run08(tag);
 case 'run09',run09(tag);
 case 'run_global_stage2',run_global_stage2(tag);
 case 'run_components_stage2',run_components_stage2(tag);
 case 'run_coupled',run_coupled(tag);
 case 'local',parts=split(string(tag),'|');check_organelle09(parts(1),str2double(parts(2)));
 otherwise,error('Unknown original production driver');
end
end

function build_mesh_stage2(tag)
% Original SpinDoctor surface writer, bundled TetGen and split_mesh, unchanged.
[root,spin]=paths_stage2();j=jsondecode(fileread(fullfile(root,'configs',string(tag)+"_mesh.json")));
folder=fullfile(root,'meshes',tag);target=fullfile(folder,'spindoctor_mesh.mat');if isfile(target),return;end
surfaces=load(fullfile(folder,'surfaces.mat'));filename=string(fullfile(folder,'mesh'));
save_surfaces(filename,surfaces);
command=sprintf('"%s" -pq1.3Aa%.12gCV "%s.poly"',fullfile(spin,'src/tetgen/lin64/tetgen'),j.max_tetrahedron_volume_um3,filename);
tic;[status,logtext]=system(command);meshing_seconds=toc;
f=fopen(fullfile(folder,'tetgen.log'),'w');fprintf(f,'%s',logtext);fclose(f);assert(status==0,logtext);
femesh_all=read_tetgen(filename+".1");assert(isequal(unique(femesh_all.elementmarkers),1:j.N+2));
assert(isequal(unique(femesh_all.facetmarkers),1:j.N+2));
tic;femesh=split_mesh(femesh_all);import_seconds=toc;
assert(femesh.ncompartment==j.N+2 && femesh.nboundary==j.N+2);
for k=1:j.N+2
 touching=find(~cellfun(@isempty,femesh.facets(:,k)))';
 if k==2,assert(isequal(touching,2));else,assert(isequal(touching,sort([2,k])));end
end
[volumes,areas]=get_vol_sa(femesh);quality=cell(1,j.N+2);
for k=1:j.N+2,quality{k}=mesh_quality(femesh.points{k},femesh.elements{k});end
matlab_version=version;
save(target,'j','surfaces','femesh_all','femesh','volumes','areas','quality','meshing_seconds','import_seconds','matlab_version','-v7');
fprintf('SAVED %s: %d tets, %d duplicated DOF, %g s\n',tag,size(femesh_all.elements,2),sum(cellfun(@(p)size(p,2),femesh.points)),meshing_seconds);
end


function build_meshes()
% Geometry ONLY: no preparation/assembly of PDE and no signal simulation.
[project,source]=public_fem_paths();
% Source root is supplied by SPINDOCTOR_ROOT; numerical mesh settings unchanged.
addpath(genpath(fullfile(source,'src')));
cfg = jsondecode(fileread(fullfile(project,'configs','geometry.json')));
Rc = cfg.cell_radius_um; Rn = cfg.nuclear_radius_um;
po = Rc*fibonacci(cfg.outer_surface_points);
pin = Rn*fibonacci(cfg.nuclear_surface_points);
fo = convhull(po')'; fi = convhull(pin')';
tetgen = fullfile(source,'src','tetgen','lin64','tetgen');
for j=1:numel(cfg.displacement_fractions)
    fraction = cfg.displacement_fractions(j); d = fraction*(Rc-Rn);
    assert(d+Rn<Rc && Rc-Rn-d>0);
    name = string(cfg.case_names{j}); folder = fullfile(project,'meshes',name);
    if ~isfolder(folder),mkdir(folder);end
    filename = string(fullfile(folder,'mesh'));
    if isfile(fullfile(folder,'spindoctor_mesh.mat'))
        fprintf('EXISTS %s\n', name); continue;
    end
    % IN and OUT remain aqueous regions. No TetGen holes are supplied.
    pn = pin+[0;0;d];
    surfaces.points = [pn,po];
    surfaces.facets = [fi,fo+size(pn,2)];
    surfaces.facetmarkers = [ones(1,size(fi,2)),2*ones(1,size(fo,2))];
    surfaces.regions = [[0;0;d],[0;0;-(Rc+Rn)/2]];
    save_surfaces(filename,surfaces);
    % Same bundled TetGen as prior validation. Absolute path avoids chmod of
    % the original binary performed by call_tetgen. Original source unchanged.
    command = sprintf('"%s" %s "%s.poly"',tetgen,cfg.tetgen_options,filename);
    tic;[status,logtext] = system(command); meshing_seconds = toc;
    f=fopen(fullfile(folder,'tetgen.log'),'w');fprintf(f,'%s',logtext);fclose(f);
    assert(status==0,'TetGen failed: %s',logtext);
    femesh_all=read_tetgen(filename+".1");
    assert(isequal(unique(femesh_all.elementmarkers),[1 2]));
    assert(isequal(unique(femesh_all.facetmarkers),[1 2]));
    tic;femesh=split_mesh(femesh_all); import_seconds=toc;
    assert(femesh.ncompartment==2 && femesh.nboundary==2);
    assert(~isempty(femesh.facets{1,1}) && ~isempty(femesh.facets{2,1}));
    assert(isempty(femesh.facets{1,2}) && ~isempty(femesh.facets{2,2}));
    [volumes,areas]=get_vol_sa(femesh);
    quality = cell(1,2);
    for k=1:2,quality{k}=mesh_quality(femesh.points{k},femesh.elements{k});end
    matlab_version=version;
    save(fullfile(folder,'spindoctor_mesh.mat'),'cfg','d','fraction','surfaces', ...
        'femesh_all','femesh','volumes','areas','quality','meshing_seconds', ...
        'import_seconds','matlab_version','-v7');
    fprintf('SAVED %s: %d tets; volumes %g %g; %g seconds\n',name, ...
        size(femesh_all.elements,2),volumes,meshing_seconds);
end
end


function op=build_operator_stage2(j,femesh,tag,Nmax)
% Same original FEM eigenproblem and MF coordinate assembly, reused by block.
[root,~]=paths_stage2();target=fullfile(root,'data',string(tag)+"_operators.mat");
if isfile(target),op=load(target);assert(strcmp(op.identity,j.identity));return;end
[setup,femesh]=setup_stage2(j,femesh);setup.mf.neig_max=Nmax;setup.mf.eigs.subspacedimension=ceil(1.625*Nmax);
nc=femesh.ncompartment;n=cellfun(@(p)size(p,2),femesh.points);off=[0,cumsum(n)];Mc=cell(1,nc);Kc=cell(1,nc);
for ic=1:nc
 p=femesh.points{ic};t=femesh.elements{ic};[~,vol]=get_volume_mesh(p,t);
 Mc{ic}=mass_matrixP1_3D(t',vol');Kc{ic}=stiffness_matrixP1_3D(t',p',setup.pde.diffusivity(:,:,ic));
end
M=blkdiag(Mc{:});K=blkdiag(Kc{:});QB=assemble_flux_matrix(femesh.points,femesh.facets);Q=couple_flux_matrix(femesh,setup.pde,QB,false);A=K+Q;
assert(norm(A-A',inf)<1e-10);assert(norm(A*ones(sum(n),1),inf)<1e-10);
if nc==2
 e=zeros(sum(n),1);e(1:n(1))=1;
 recovered_nuclear_kappa_um_s=e'*Q*e/femesh.areas(1)*1e6;
 assert(abs(recovered_nuclear_kappa_um_s-j.kappa_n_um_s)<1e-8);
else
 assert(nnz(Q)==0);recovered_nuclear_kappa_um_s=0;
end
eigenfile=fullfile(root,'cache',string(tag)+"_eigen.mat");
if isfile(eigenfile),s=load(eigenfile);assert(strcmp(s.identity,j.identity));le=s.le;eigen_seconds=s.eigen_seconds;
else
 rng(j.eigen_seed,'twister');t=tic;le=compute_laplace_eig(femesh,setup.pde,setup.mf);eigen_seconds=toc(t);identity=j.identity;
 temp=fullfile(root,'cache',string(tag)+"_eigen_partial.mat");save(temp,'le','eigen_seconds','identity','-v7.3');movefile(temp,eigenfile);
end
assert(numel(le)==1);lambda=le.values;zero_count=sum(abs(lambda)<1e-11);assert(zero_count==1);
gram_error=max(abs(le.funcs'*M*le.funcs-eye(Nmax)),[],'all');assert(gram_error<1e-8);
t=tic;cachepath=fullfile(root,'cache',string(tag)+"_mf");
txt=evalc('original=solve_mf_cpu(femesh,setup,le,cachepath,false);');MF_initialization_seconds=toc(t);
f=fopen(fullfile(root,'logs',string(tag)+"_original_MF.log"),'w');fprintf(f,'%s',txt);fclose(f);
saved=load(fullfile(add_mf_str_savepath(cachepath,setup.mf),'FEM_lap_eig.mat'));
moments=saved.moments;nu0=saved.nu0;LQT2=saved.LQT2;P=zeros(nc,Nmax);
for ic=1:nc,P(ic,:)=sum(Mc{ic}*le.funcs(off(ic)+1:off(ic+1),:),1);end
norm0=setup.pde.initial_signal;volume=norm0;region_volumes=femesh.volumes;ndof=sum(n);nelements=sum(cellfun(@(t)size(t,2),femesh.elements));
original_signal=reshape(original.signal_allcmpts,setup.nsequence,1)/norm0;
identity=j.identity;
save(target,'lambda','moments','nu0','LQT2','P','norm0','region_volumes','volume','ndof','nelements','zero_count','eigen_seconds','MF_initialization_seconds','original_signal','gram_error','identity','recovered_nuclear_kappa_um_s','-v7');
op=load(target);
end


function check_organelle09(level,Norg)
% Exact disconnected-domain diagnostic, NOT a homogenization/reduction.
% It audits local modes hidden above the global eigenvalue cutoff.
[root,~]=paths09();tag="P1_N"+string(Norg)+"_"+string(level)+"_sealed";
target=fullfile(root,'data',"organelle_local_N"+string(Norg)+"_"+string(level)+".mat");if isfile(target),return;end
j=jsondecode(fileread(fullfile(root,'configs/jobs',tag+".json")));raw=load(j.mesh_file);
bs=[0;j.b_s_mm2(:)];de=[j.delta_ms(1);j.delta_ms(:)]*1000;De=[j.Delta_ms(1);j.Delta_ms(:)]*1000;
ns=numel(bs);nd=size(j.directions,1);seq=cell(1,ns);q=zeros(1,ns);
for k=1:ns,seq{k}=PGSE(de(k),De(k));q(k)=sqrt(bs(k)/seq{k}.bvalue_no_q);end
budgets=[30,60,100];local_signals=zeros(Norg,numel(budgets),ns,nd);local_values=cell(1,Norg);times=zeros(1,Norg);
local_operators=cell(1,Norg);
for io=1:Norg
 t=tic;ic=io+2;
 femesh.ncompartment=1;femesh.nboundary=1;femesh.points=raw.femesh.points(ic);femesh.elements=raw.femesh.elements(ic);
 femesh.facets=raw.femesh.facets(ic,ic);femesh.point_map={1:size(femesh.points{1},2)};
 [femesh.volumes,femesh.areas]=get_vol_sa(femesh);
 pde.diffusivity=j.D_um2_ms/1000*eye(3);pde.mean_diffusivity=j.D_um2_ms/1000;pde.permeability=0;pde.initial_density=1;pde.relaxation=Inf;
 mf.neig_max=100;mf.length_scale=0;mf.surf_relaxation=false;mf.rerun_eigen=true;
 mf.eigs.sigma=-1e-8;mf.eigs.tolerance=1e-10;mf.eigs.maxiter=1000;mf.eigs.subspacedimension=165;
 rng(j.eigen_seed+io,'twister');le=compute_laplace_eig(femesh,pde,mf);
 V=le.funcs;lambda=le.values;points=femesh.points{1};elements=femesh.elements{1};[~,vol]=get_volume_mesh(points,elements);
 M=mass_matrixP1_3D(elements',vol');norm0=sum(vol);nu0=V'*M*ones(size(V,1),1);P=sum(M*V,1);moments=zeros(100,100,3);
 for k=1:3,J=mass_matrixP1_3D(elements',vol',points(k,:)');moments(:,:,k)=V'*J*V;end
 for ib=1:numel(budgets)
  N=budgets(ib);
  for idir=1:nd
   a=sum(moments(1:N,1:N,:).*reshape(j.directions(idir,:),1,1,3),3);
   for iseq=1:ns
    nu=evolve_laplace_coef(nu0(1:N),seq{iseq},1i*q(iseq)*a,lambda(1:N)',3);
    local_signals(io,ib,iseq,idir)=real(P(1:N)*nu/norm0);
   end
  end
 end
 assert(max(abs(local_signals(io,:,1,:)-1),[],'all')<1e-9);
 local_values{io}=lambda;local_operators{io}=struct('lambda',lambda,'moments',moments,'nu0',nu0,'projection',P,'norm0',norm0);
 times(io)=toc(t);fprintf('LOCAL ORG %s %d: %g seconds; first eigenvalues %g %g %g %g\n',level,io,times(io),lambda(1:4));
end
save(target,'j','budgets','local_signals','local_values','local_operators','times','-v7');
end


function femesh=mesh09(raw,transparent)
% The exact transparent constraint merges ONLY organelles with cytoplasm.
% The finite nuclear interface retains its duplicated trace degrees of freedom.
if ~transparent,femesh=raw.femesh;return;end
global_mesh=raw.femesh_all;
global_mesh.elementmarkers(global_mesh.elementmarkers>=3)=2;
keep=global_mesh.facetmarkers<=2;
global_mesh.facets=global_mesh.facets(:,keep);global_mesh.facetmarkers=global_mesh.facetmarkers(keep);
femesh=split_mesh(global_mesh);assert(femesh.ncompartment==2 && femesh.nboundary==2);
end


function femesh=mesh_stage2(raw,transparent)
% The exact transparent constraint merges ONLY organelles with cytoplasm.
% The finite nuclear interface retains its duplicated trace degrees of freedom.
if ~transparent,femesh=raw.femesh;return;end
global_mesh=raw.femesh_all;
global_mesh.elementmarkers(global_mesh.elementmarkers>=3)=2;
keep=global_mesh.facetmarkers<=2;
global_mesh.facets=global_mesh.facets(:,keep);global_mesh.facetmarkers=global_mesh.facetmarkers(keep);
femesh=split_mesh(global_mesh);assert(femesh.ncompartment==2 && femesh.nboundary==2);
end


function [project,work,source]=paths08()
[project,source]=public_fem_paths();work=project;
end


function [root,spin]=paths09()
[root,spin]=public_fem_paths();
end


function [root,spin]=paths_campaign()
[root,spin]=public_fem_paths();
end


function [root,spin]=paths_stage2()
[root,spin]=public_fem_paths();
end


function [signals,components,seconds]=propagate_operator_stage2(op,j,budgets)
% Original finite-PGSE propagator, same coordinate matrices at every budget.
ns=numel(j.b_s_mm2)+1;nd=size(j.directions,1);
bs=[0;j.b_s_mm2(:)];de=[j.delta_ms(1);j.delta_ms(:)]*1000;De=[j.Delta_ms(1);j.Delta_ms(:)]*1000;
seq=cell(1,ns);q=zeros(1,ns);
for iseq=1:ns,seq{iseq}=PGSE(de(iseq),De(iseq));q(iseq)=sqrt(bs(iseq)/seq{iseq}.bvalue_no_q);end
nc=size(op.P,1);signals=zeros(numel(budgets),ns,nd);components=zeros(nc,numel(budgets),ns,nd);seconds=zeros(size(budgets));
for ib=1:numel(budgets)
 N=budgets(ib);t=tic;
 for idir=1:nd
  a=sum(op.moments(1:N,1:N,:).*reshape(j.directions(idir,:),1,1,3),3);
  for iseq=1:ns
   nu=evolve_laplace_coef(op.nu0(1:N),seq{iseq},1i*q(iseq)*a,op.LQT2(1:N),3);
   components(:,ib,iseq,idir)=op.P(:,1:N)*nu/op.norm0;
   signals(ib,iseq,idir)=sum(components(:,ib,iseq,idir));
  end
 end
 seconds(ib)=toc(t);assert(max(abs(signals(ib,1,:)-1),[],'all')<1e-8);assert(max(abs(imag(signals(ib,:,:))),[],'all')<1e-9);
end
if budgets(end)==numel(op.lambda),assert(max(abs(reshape(signals(end,:,1),[],1)-op.original_signal))<1e-9);end
end


function [root,spin]=public_fem_paths()
% Explicit external dependency and a NEW output-only work directory.
root=getenv('INSIDE_SPHERE_FEM_WORK');spin=getenv('SPINDOCTOR_ROOT');
assert(~isempty(root),'Set INSIDE_SPHERE_FEM_WORK to the prepared case directory.');
assert(isfolder(fullfile(spin,'src')),'Set SPINDOCTOR_ROOT to the documented SpinDoctor checkout.');
addpath(genpath(fullfile(spin,'src')));
addpath(fileparts(mfilename('fullpath')));
end


function run08(tag)
% Only an input/cache adapter: original FEM eigenproblem and original MF.
[project,~,~]=paths08();j=jsondecode(fileread(fullfile(project,'configs/jobs',string(tag)+".json")));
target=fullfile(project,'data/forward',string(tag)+".mat");
if isfile(target),s=load(target,'job');assert(strcmp(s.job.identity,j.identity));return;end
c=j.condition;raw=load(j.mesh_file);femesh=raw.femesh;
[femesh.volumes,femesh.areas]=get_vol_sa(femesh);
femesh.total_volume=sum(femesh.volumes);femesh.total_area=sum(femesh.areas);
setup.name=string(fullfile(project,'cache','unused'));
setup.geometry.cell_shape="sphere";setup.geometry.ncell=1;setup.geometry.rmin=c.Rc;setup.geometry.rmax=c.Rc;
setup.geometry.dmin=0;setup.geometry.dmax=0;setup.geometry.include_in=true;
setup.geometry.in_ratio=c.Rn/c.Rc;setup.geometry.ecs_shape="no_ecs";setup.geometry.deformation=[0,0];
setup.pde.diffusivity_in=c.Dc*c.ratio/1000;setup.pde.diffusivity_out=c.Dc/1000;
setup.pde.initial_density_in=1;setup.pde.initial_density_out=1;
setup.pde.relaxation_in=Inf;setup.pde.relaxation_out=Inf;
setup.pde.permeability_in_out=c.kappa_um_s/1e6;setup.pde.permeability_in=0;setup.pde.permeability_out=0;
setup.gamma=j.gamma_rad_us_mT;
% One amplitude per sequence, with a distinct b/timing pair in every column.
% This avoids the unrequested Cartesian product of P2 b values and timings.
bs=[0;j.b_s_mm2(:)];de=[j.delta_ms(1);j.delta_ms(:)];De=[j.Delta_ms(1);j.Delta_ms(:)];
ns=numel(bs);sequences=cell(1,ns);
for k=1:ns,sequences{k}=PGSE(de(k)*1000,De(k)*1000);end
setup.gradient.sequences=sequences;setup.gradient.values=0;setup.gradient.values_type="b";
setup.gradient.directions=j.directions';
setup.geometry=prepare_geometry(setup);setup.pde=prepare_pde(setup);setup=prepare_experiments(setup);
for k=1:ns
 setup.gradient.bvalues(1,k)=bs(k);
 setup.gradient.qvalues(1,k)=sqrt(bs(k)/sequences{k}.bvalue_no_q);
end
setup.gradient.gvalues=setup.gradient.qvalues*1e6/setup.gamma;
setup.namplitude=1;setup.nsequence=ns;setup.ndirection=size(j.directions,1);
setup.ncompartment=numel(setup.pde.compartments);setup.nboundary=numel(setup.pde.boundaries);
setup.pde.initial_signal=setup.pde.initial_density*femesh.volumes';
setup.pde.mean_diffusivity=compute_mean_diffusivity(setup.pde.diffusivity,femesh);
conservation=sd_conservation(femesh,setup);
assert(conservation.global_stationary_residual<1e-10);assert(abs(conservation.recovered_kappa_um_ms-.2)<1e-10);
setup.mf.neig_max=j.eigen_modes;setup.mf.length_scale=0;setup.mf.single=false;setup.mf.gpu=false;
setup.mf.ninterval=3;setup.mf.surf_relaxation=false;setup.mf.rerun=true;setup.mf.rerun_eigen=true;
setup.mf.eigs.sigma=-1e-8;setup.mf.eigs.tolerance=1e-10;setup.mf.eigs.maxiter=1000;
setup.mf.eigs.subspacedimension=ceil(1.625*j.eigen_modes);
eigfile=fullfile(project,'cache',sprintf('%s_lap%d.mat',tag,j.eigen_modes));clock=tic;
if isfile(eigfile)
 saved=load(eigfile);assert(strcmp(saved.identity,j.identity));lap_eig=saved.lap_eig;eigen_seconds=saved.eigen_seconds;
else
 rng(j.eigen_seed,'twister');et=tic;lap_eig=compute_laplace_eig(femesh,setup.pde,setup.mf);eigen_seconds=toc(et);identity=j.identity;
 temp=fullfile(project,'cache',sprintf('partial_%s_lap%d.mat',tag,j.eigen_modes));
 save(temp,'lap_eig','eigen_seconds','identity','-v7.3');movefile(temp,eigfile);
end
budgets=j.budgets(:)';signals=cell(size(budgets));times=zeros(size(budgets));b0_checks=zeros(size(budgets));
for ib=1:numel(budgets)
 N=budgets(ib);le=lap_eig;le.values=le.values(1:N);le.funcs=le.funcs(:,1:N);le.length_scales=le.length_scales(1:N);
 pt=tic;original_output=evalc('result=solve_mf_cpu(femesh,setup,le,fullfile(project,''cache'',sprintf(''%s_N%d'',tag,N)),false);');
 times(ib)=toc(pt);signals{ib}=reshape(result.signal_allcmpts,ns,setup.ndirection)/setup.pde.initial_signal;
 f=fopen(fullfile(project,'logs',sprintf('%s_N%d_original.log',tag,N)),'w');fprintf(f,'%s',original_output);fclose(f);
 % Explicit stationary baseline from the unchanged original spectral propagator.
 % Eigenfunction/mass normalization is checked by the original MF coefficients.
 b0_checks(ib)=max(abs(signals{ib}(1,:)-1));assert(b0_checks(ib)<1e-8);
 fprintf('SIGNAL %s N%d: %g seconds\n',tag,N,times(ib));
end
job=j;n_tetrahedra=sum(cellfun(@(x)size(x,2),femesh.elements));n_dof=sum(cellfun(@(x)size(x,2),femesh.points));
volumes=femesh.volumes;areas=femesh.areas;eigenvalues=lap_eig.values;total_seconds=toc(clock);
% Store timing/unit metadata without MATLAB sequence objects (portable MAT-v7).
acquisition=struct('b',setup.gradient.bvalues,'q',setup.gradient.qvalues,'G_mT_m',setup.gradient.gvalues, ...
 'delta_us',de(:)'*1000,'Delta_us',De(:)'*1000);
temp=fullfile(project,'data/forward',"partial_"+string(tag)+".mat");
save(temp,'job','signals','budgets','times','eigen_seconds','eigenvalues','conservation','n_tetrahedra','n_dof','volumes','areas','total_seconds','acquisition','b0_checks','-v7');movefile(temp,target);
fprintf('SAVED %s: DOF%d tets%d eigen%g seconds\n',tag,n_dof,n_tetrahedra,eigen_seconds);
end


function run09(tag)
% Input/QA adapter only. Original SpinDoctor FEM eigenproblem, original MF and
% original finite-PGSE evolve_laplace_coef are used without source changes.
[root,~]=paths09();j=jsondecode(fileread(fullfile(root,'configs/jobs',string(tag)+".json")));
target=fullfile(root,'data',string(tag)+".mat");if isfile(target),return;end
raw=load(j.mesh_file);femesh=mesh09(raw,j.transparent);
[femesh.volumes,femesh.areas]=get_vol_sa(femesh);femesh.total_volume=sum(femesh.volumes);femesh.total_area=sum(femesh.areas);
nc=femesh.ncompartment;nb=femesh.nboundary;
pde.diffusivity=repmat(j.D_um2_ms/1000*eye(3),1,1,nc);pde.initial_density=ones(1,nc);pde.relaxation=Inf(1,nc);
pde.permeability=zeros(1,nb);pde.permeability(1)=j.kappa_n_um_s/1e6;
pde.compartments=["nucleus","cytoplasm",repmat("organelle",1,nc-2)];pde.boundaries=["nucleus,cytoplasm","outer",repmat("organelle,cytoplasm",1,nb-2)];
pde.initial_signal=sum(femesh.volumes);pde.mean_diffusivity=compute_mean_diffusivity(pde.diffusivity,femesh);
setup.pde=pde;setup.gamma=j.gamma_rad_us_mT;
bs=[0;j.b_s_mm2(:)];de=[j.delta_ms(1);j.delta_ms(:)]*1000;De=[j.Delta_ms(1);j.Delta_ms(:)]*1000;
ns=numel(bs);sequences=cell(1,ns);for i=1:ns,sequences{i}=PGSE(de(i),De(i));end
setup.gradient.sequences=sequences;setup.gradient.values=0;setup.gradient.values_type="b";setup.gradient.directions=[1;0;0];
setup=prepare_experiments(setup);
for i=1:ns,setup.gradient.bvalues(i)=bs(i);setup.gradient.qvalues(i)=sqrt(bs(i)/sequences{i}.bvalue_no_q);end
setup.gradient.gvalues=setup.gradient.qvalues*1e6/setup.gamma;
setup.namplitude=1;setup.nsequence=ns;setup.ndirection=1;setup.ncompartment=nc;setup.nboundary=nb;
mf.neig_max=j.eigen_modes;mf.length_scale=0;mf.single=false;mf.gpu=false;mf.ninterval=3;mf.surf_relaxation=false;mf.rerun=true;mf.rerun_eigen=true;
mf.eigs.sigma=-1e-8;mf.eigs.tolerance=1e-10;mf.eigs.maxiter=1000;mf.eigs.subspacedimension=ceil(1.625*j.eigen_modes);setup.mf=mf;
% Independent assembly diagnostics, including the entire stationary subspace.
n=cellfun(@(p)size(p,2),femesh.points);offset=[0,cumsum(n)];Mcell=cell(1,nc);Kcell=cell(1,nc);
for ic=1:nc
 t=femesh.elements{ic};p=femesh.points{ic};[~,v]=get_volume_mesh(p,t);
 Mcell{ic}=mass_matrixP1_3D(t',v');Kcell{ic}=stiffness_matrixP1_3D(t',p',pde.diffusivity(:,:,ic));
end
M=blkdiag(Mcell{:});K=blkdiag(Kcell{:});QB=assemble_flux_matrix(femesh.points,femesh.facets);Q=couple_flux_matrix(femesh,pde,QB,false);A=K+Q;
C=zeros(sum(n),nc-1);C(1:offset(3),1)=1;
for ic=3:nc,C(offset(ic)+1:offset(ic+1),ic-1)=1;end
diagnostics.stationary_residual_max=max(abs(A*C),[],'all');assert(diagnostics.stationary_residual_max<1e-10);
diagnostics.flux_symmetric_error=norm(Q-Q',inf);assert(diagnostics.flux_symmetric_error<1e-12);
en=zeros(sum(n),1);en(1:n(1))=1;diagnostics.nuclear_jump_energy=en'*Q*en;
diagnostics.recovered_kappa_n_um_s=diagnostics.nuclear_jump_energy/femesh.areas(1)*1e6;
assert(abs(diagnostics.recovered_kappa_n_um_s-j.kappa_n_um_s)<1e-8);
eigfile=fullfile(root,'cache',string(tag)+"_eigen.mat");tic;
if isfile(eigfile),saved=load(eigfile);lap_eig=saved.lap_eig;eigen_seconds=saved.eigen_seconds;
else
 rng(j.eigen_seed,'twister');et=tic;lap_eig=compute_laplace_eig(femesh,pde,mf);eigen_seconds=toc(et);
 save(eigfile,'lap_eig','eigen_seconds','-v7.3');
end
lambda=lap_eig.values;V=lap_eig.funcs;zero_count=sum(abs(lambda)<1e-11);assert(zero_count==nc-1);
diagnostics.mass_orthogonality_max=max(abs(V'*M*V-eye(numel(lambda))),[],'all');assert(diagnostics.mass_orthogonality_max<1e-8);
cc=V'*M*C;diagnostics.stationary_projection_max=max(abs(V*cc-C),[],'all');assert(diagnostics.stationary_projection_max<1e-7);
diagnostics.no_gradient_100ms_max=max(abs(V*(exp(-lambda*1e5).*cc)-C),[],'all');assert(diagnostics.no_gradient_100ms_max<1e-7);
diagnostics.eigenvalue_min=lambda(1);diagnostics.first_positive_eigenvalue=lambda(zero_count+1);
% Original MF computes coordinate matrices and one complete protocol along x.
cachepath=fullfile(root,'cache',string(tag)+"_mf");t=tic;logtext=evalc('original=solve_mf_cpu(femesh,setup,lap_eig,cachepath,false);');original_MF_seconds=toc(t);
f=fopen(fullfile(root,'logs',string(tag)+"_original_MF.log"),'w');fprintf(f,'%s',logtext);fclose(f);
cachepath=add_mf_str_savepath(cachepath,mf);op=load(fullfile(cachepath,'FEM_lap_eig.mat'));moments=op.moments;nu0=op.nu0;LQT2=op.LQT2;
component_projection=zeros(nc,numel(lambda));
for ic=1:nc,component_projection(ic,:)=sum(Mcell{ic}*V(offset(ic)+1:offset(ic+1),:),1);end
norm0=pde.initial_signal;original_signal=reshape(original.signal_allcmpts,ns,1)/norm0;
original_components=reshape(original.signal,nc,ns)/norm0;
qvalues=setup.gradient.qvalues;acquisition=struct('b_s_mm2',bs,'q_um_inv_us_inv',qvalues,'G_mT_m',setup.gradient.gvalues,'delta_us',de,'Delta_us',De);
operatorfile=fullfile(root,'data',string(tag)+"_operators.mat");
save(operatorfile,'lambda','moments','nu0','LQT2','component_projection','norm0','acquisition','diagnostics','zero_count','original_signal','original_components','original_MF_seconds','eigen_seconds','-v7');
% Reuse original propagator, with nested submatrices of the SAME basis. No
% new eigensolve, approximation, or waveform discretization is introduced.
budgets=j.budgets(:)';signals=cell(size(budgets));components=cell(size(budgets));propagation_seconds=zeros(size(budgets));
for ib=1:numel(budgets)
 N=budgets(ib);nd=size(j.directions,1);s=zeros(ns,nd);sc=zeros(nc,ns,nd);t=tic;
 for idir=1:nd
  a=sum(moments(1:N,1:N,:).*reshape(j.directions(idir,:),1,1,3),3);
  for iseq=1:ns
   nu=evolve_laplace_coef(nu0(1:N),sequences{iseq},1i*qvalues(iseq)*a,LQT2(1:N),3);
   sc(:,iseq,idir)=component_projection(:,1:N)*nu/norm0;s(iseq,idir)=sum(sc(:,iseq,idir));
  end
 end
 propagation_seconds(ib)=toc(t);signals{ib}=s;components{ib}=sc;
 assert(max(abs(s(1,:)-1))<1e-8);assert(max(abs(imag(s)),[],'all')<1e-9);
 assert(max(abs(sc(:,1,:)-reshape(femesh.volumes/norm0,nc,1,1)),[],'all')<1e-8);
 if N==numel(lambda),assert(max(abs(s(:,1)-original_signal))<1e-9);assert(max(abs(sc(:,:,1)-original_components),[],'all')<1e-9);end
 fprintf('PROPAGATED %s N%d: %g seconds\n',tag,N,propagation_seconds(ib));
end
job=j;volumes=femesh.volumes;areas=femesh.areas;n_dof=sum(n);n_tetrahedra=sum(cellfun(@(t)size(t,2),femesh.elements));
save(target,'job','signals','components','budgets','propagation_seconds','eigen_seconds','original_MF_seconds','zero_count','lambda','volumes','areas','n_dof','n_tetrahedra','diagnostics','acquisition','-v7');
fprintf('SAVED %s DOF%d modes%d zero%d eigen%g seconds\n',tag,n_dof,numel(lambda),zero_count,eigen_seconds);
end


function run_components_stage2(tag)
% Exact direct-sum decomposition ONLY at kappa_org=0.
[root,~]=paths_stage2();j=jsondecode(fileread(fullfile(root,'configs/jobs',string(tag)+".json")));
assert(~j.transparent && j.kappa_org_um_s==0,'Disconnected decomposition is invalid for finite permeability.');
target=fullfile(root,'data',string(tag)+".mat");if isfile(target),return;end
raw=load(j.mesh_file);full=raw.femesh;[volumes,areas]=get_vol_sa(full);total_volume=sum(volumes);nblocks=j.N+1;
weights=[sum(volumes(1:2)),volumes(3:end)]/total_volume;
block_signals=cell(1,nblocks);block_components=cell(1,nblocks);block_budgets=cell(1,nblocks);block_rates=cell(1,nblocks);block_diagnostics=cell(1,nblocks);block_times=cell(1,nblocks);
for iblock=1:nblocks
 blocktag=string(tag)+sprintf('_block%02d',iblock-1);savedfile=fullfile(root,'data',blocktag+"_signal.mat");
 if isfile(savedfile)
  s=load(savedfile);assert(strcmp(s.identity,j.identity));
  block_signals{iblock}=s.sig;block_components{iblock}=s.comp;block_budgets{iblock}=s.budgets;block_rates{iblock}=s.rates;block_diagnostics{iblock}=s.diagnostics;block_times{iblock}=s.times;
  continue;
 end
 if iblock==1
  % Keep actual punctured cytoplasm and finite nuclear interface. All holes
  % are reflecting on this connected block; their aqueous interiors follow.
  fm.ncompartment=2;fm.nboundary=full.nboundary;
  fm.points=full.points(1:2);fm.elements=full.elements(1:2);fm.point_map=full.point_map(1:2);fm.facets=full.facets(1:2,:);
  budgets=j.bulk_budgets(:)';
 else
  ic=iblock+1;fm.ncompartment=1;fm.nboundary=1;
  fm.points=full.points(ic);fm.elements=full.elements(ic);fm.point_map={1:size(fm.points{1},2)};fm.facets=full.facets(ic,ic);
  budgets=j.local_budgets(:)';
 end
 op=build_operator_stage2(j,fm,blocktag,budgets(end));
 [sig,comp,times]=propagate_operator_stage2(op,j,budgets);
 rates=op.lambda;diagnostics=rmfield(op,{'lambda','moments','nu0','LQT2','P','original_signal'});identity=j.identity;
 save(savedfile,'sig','comp','budgets','rates','diagnostics','times','identity','-v7');
 block_signals{iblock}=sig;block_components{iblock}=comp;block_budgets{iblock}=budgets;block_rates{iblock}=rates;block_diagnostics{iblock}=diagnostics;block_times{iblock}=times;
 fprintf('BLOCK %s %d/%d DOF%d modes%d eig%g s prop%g s\n',tag,iblock,nblocks,op.ndof,budgets(end),op.eigen_seconds,sum(times));
 clear op;
end
ns=numel(j.b_s_mm2)+1;nd=size(j.directions,1);nlevel=numel(j.bulk_budgets);signals=zeros(nlevel,ns,nd);
for iblock=1:nblocks,signals=signals+weights(iblock)*block_signals{iblock};end
assert(max(abs(signals(:,1,:)-1),[],'all')<1e-8);zero_count=sum(cellfun(@(s)s.zero_count,block_diagnostics));assert(zero_count==j.N+1);
job=j;n_dof=sum(cellfun(@(p)size(p,2),full.points));n_tetrahedra=sum(cellfun(@(t)size(t,2),full.elements));
save(target,'job','signals','block_signals','block_components','block_budgets','block_rates','block_diagnostics','block_times','weights','volumes','areas','zero_count','n_dof','n_tetrahedra','-v7');
fprintf('COMPONENTS SAVED %s: %d stationary modes; no finite-permeability reduction\n',tag,zero_count);
end


function run_coupled(tag)
% Input/QA adapter only. Original SpinDoctor FEM eigenproblem, original MF and
% original finite-PGSE evolve_laplace_coef are used without source changes.
[root,~]=paths_campaign();j=jsondecode(fileread(fullfile(root,'configs/jobs',string(tag)+".json")));
target=fullfile(root,'data/forward',string(tag)+".mat");if isfile(target),return;end
raw=load(j.mesh_file);femesh=raw.femesh;assert(~j.transparent && j.kappa_org_um_s>0);
[femesh.volumes,femesh.areas]=get_vol_sa(femesh);femesh.total_volume=sum(femesh.volumes);femesh.total_area=sum(femesh.areas);
nc=femesh.ncompartment;nb=femesh.nboundary;
pde.diffusivity=repmat(j.D_um2_ms/1000*eye(3),1,1,nc);pde.initial_density=ones(1,nc);pde.relaxation=Inf(1,nc);
pde.permeability=zeros(1,nb);pde.permeability(1)=j.kappa_n_um_s/1e6;
if nc>2,pde.permeability(3:end)=j.kappa_org_um_s/1e6;end
pde.compartments=["nucleus","cytoplasm",repmat("organelle",1,nc-2)];pde.boundaries=["nucleus,cytoplasm","outer",repmat("organelle,cytoplasm",1,nb-2)];
pde.initial_signal=sum(femesh.volumes);pde.mean_diffusivity=compute_mean_diffusivity(pde.diffusivity,femesh);
setup.pde=pde;setup.gamma=j.gamma_rad_us_mT;
bs=[0;j.b_s_mm2(:)];de=[j.delta_ms(1);j.delta_ms(:)]*1000;De=[j.Delta_ms(1);j.Delta_ms(:)]*1000;
ns=numel(bs);sequences=cell(1,ns);for i=1:ns,sequences{i}=PGSE(de(i),De(i));end
setup.gradient.sequences=sequences;setup.gradient.values=0;setup.gradient.values_type="b";setup.gradient.directions=[1;0;0];
setup=prepare_experiments(setup);
for i=1:ns,setup.gradient.bvalues(i)=bs(i);setup.gradient.qvalues(i)=sqrt(bs(i)/sequences{i}.bvalue_no_q);end
setup.gradient.gvalues=setup.gradient.qvalues*1e6/setup.gamma;
setup.namplitude=1;setup.nsequence=ns;setup.ndirection=1;setup.ncompartment=nc;setup.nboundary=nb;
mf.neig_max=j.eigen_modes;mf.length_scale=0;mf.single=false;mf.gpu=false;mf.ninterval=3;mf.surf_relaxation=false;mf.rerun=true;mf.rerun_eigen=true;
mf.eigs.sigma=-1e-8;mf.eigs.tolerance=1e-10;mf.eigs.maxiter=1000;mf.eigs.subspacedimension=ceil(1.625*j.eigen_modes);setup.mf=mf;
% Independent assembly diagnostics, including the entire stationary subspace.
n=cellfun(@(p)size(p,2),femesh.points);offset=[0,cumsum(n)];Mcell=cell(1,nc);Kcell=cell(1,nc);
for ic=1:nc
 t=femesh.elements{ic};p=femesh.points{ic};[~,v]=get_volume_mesh(p,t);
 Mcell{ic}=mass_matrixP1_3D(t',v');Kcell{ic}=stiffness_matrixP1_3D(t',p',pde.diffusivity(:,:,ic));
end
M=blkdiag(Mcell{:});K=blkdiag(Kcell{:});QB=assemble_flux_matrix(femesh.points,femesh.facets);Q=couple_flux_matrix(femesh,pde,QB,false);A=K+Q;
if j.transparent || j.kappa_org_um_s>0
 C=ones(sum(n),1);expected_zeros=1;
else
 C=zeros(sum(n),nc-1);C(1:offset(3),1)=1;
 for ic=3:nc,C(offset(ic)+1:offset(ic+1),ic-1)=1;end;expected_zeros=nc-1;
end
diagnostics.stationary_residual_max=max(abs(A*C),[],'all');assert(diagnostics.stationary_residual_max<1e-10);
diagnostics.flux_symmetric_error=norm(Q-Q',inf);assert(diagnostics.flux_symmetric_error<1e-12);
en=zeros(sum(n),1);en(1:n(1))=1;diagnostics.nuclear_jump_energy=en'*Q*en;
diagnostics.recovered_kappa_n_um_s=diagnostics.nuclear_jump_energy/femesh.areas(1)*1e6;
assert(abs(diagnostics.recovered_kappa_n_um_s-j.kappa_n_um_s)<1e-8);
diagnostics.recovered_organelle_permeabilities_um_s=zeros(1,nc-2);
for ic=3:nc
 e=zeros(sum(n),1);e(offset(ic)+1:offset(ic+1))=1;
 diagnostics.recovered_organelle_permeabilities_um_s(ic-2)=e'*Q*e/femesh.areas(ic)*1e6;
end
if nc>2,assert(max(abs(diagnostics.recovered_organelle_permeabilities_um_s-j.kappa_org_um_s))<1e-8);end
assert(norm(A-A',inf)<1e-10);
clear A K Q QB Kcell; % QA assembly only: no change to original eigensolver.
eigfile=fullfile(root,'cache',string(tag)+"_eigen.mat");tic;
if isfile(eigfile),saved=load(eigfile);assert(strcmp(saved.job_identity,j.identity));lap_eig=saved.lap_eig;eigen_seconds=saved.eigen_seconds;
else
 rng(j.eigen_seed,'twister');et=tic;lap_eig=compute_laplace_eig(femesh,pde,mf);eigen_seconds=toc(et);
 job_identity=j.identity;tmp=fullfile(root,'cache',string(tag)+"_eigen_partial.mat");
 save(tmp,'lap_eig','eigen_seconds','job_identity','-v7.3');movefile(tmp,eigfile);
end
lambda=lap_eig.values;V=lap_eig.funcs;zero_count=sum(abs(lambda)<1e-11);assert(zero_count==expected_zeros);
% Same full Gram check in column blocks; avoids an unnecessary DOF-by-mode temporary.
diagnostics.mass_orthogonality_max=0;
for first=1:100:numel(lambda)
 ii=first:min(first+99,numel(lambda));target_gram=zeros(numel(lambda),numel(ii));
 target_gram(sub2ind(size(target_gram),ii,1:numel(ii)))=1;
 diagnostics.mass_orthogonality_max=max(diagnostics.mass_orthogonality_max,max(abs(V'*(M*V(:,ii))-target_gram),[],'all'));
end
assert(diagnostics.mass_orthogonality_max<1e-8);
cc=V'*M*C;diagnostics.stationary_projection_max=max(abs(V*cc-C),[],'all');assert(diagnostics.stationary_projection_max<1e-7);
diagnostics.no_gradient_100ms_max=max(abs(V*(exp(-lambda*1e5).*cc)-C),[],'all');assert(diagnostics.no_gradient_100ms_max<1e-7);
diagnostics.eigenvalue_min=lambda(1);diagnostics.first_positive_eigenvalue=lambda(zero_count+1);
% Original MF computes coordinate matrices and one complete protocol along x.
cachepath=fullfile(root,'cache',string(tag)+"_mf");t=tic;logtext=evalc('original=solve_mf_cpu(femesh,setup,lap_eig,cachepath,false);');original_MF_seconds=toc(t);
f=fopen(fullfile(root,'logs',string(tag)+"_original_MF.log"),'w');fprintf(f,'%s',logtext);fclose(f);
cachepath=add_mf_str_savepath(cachepath,mf);op=load(fullfile(cachepath,'FEM_lap_eig.mat'));moments=op.moments;nu0=op.nu0;LQT2=op.LQT2;
organelle_participation=zeros(nc-2,numel(lambda));
local_dipole_coefficients=cell(1,nc-2);local_dipole_norms=cell(1,nc-2);
for ic=3:nc
 inds=offset(ic)+1:offset(ic+1);Vi=V(inds,:);Mi=Mcell{ic};
 organelle_participation(ic-2,:)=sum(Vi.*(Mi*Vi),1);
 U=femesh.points{ic}'-reshape(j.centers_um(ic-2,:),1,3);
 local_dipole_coefficients{ic-2}=Vi'*Mi*U;
 local_dipole_norms{ic-2}=sum(U.*(Mi*U),1);
end
component_projection=zeros(nc,numel(lambda));
for ic=1:nc,component_projection(ic,:)=sum(Mcell{ic}*V(offset(ic)+1:offset(ic+1),:),1);end
diagnostics.no_gradient_population_error=max(abs(component_projection*(exp(-lambda*1e5).*nu0)/pde.initial_signal-femesh.volumes(:)/pde.initial_signal));
assert(diagnostics.no_gradient_population_error<1e-8);
norm0=pde.initial_signal;original_signal=reshape(original.signal_allcmpts,ns,1)/norm0;
original_components=reshape(original.signal,nc,ns)/norm0;
qvalues=setup.gradient.qvalues;acquisition=struct('b_s_mm2',bs,'q_um_inv_us_inv',qvalues,'G_mT_m',setup.gradient.gvalues,'delta_us',de,'Delta_us',De);
operatorfile=fullfile(root,'data/forward',string(tag)+"_operators.mat");
save(operatorfile,'lambda','moments','nu0','LQT2','component_projection','norm0','acquisition','diagnostics','zero_count','original_signal','original_components','original_MF_seconds','eigen_seconds','organelle_participation','local_dipole_coefficients','local_dipole_norms','-v7');
% Reuse original propagator, with nested submatrices of the SAME basis. No
% new eigensolve, approximation, or waveform discretization is introduced.
budgets=j.budgets(:)';signals=cell(size(budgets));components=cell(size(budgets));propagation_seconds=zeros(size(budgets));
for ib=1:numel(budgets)
 N=budgets(ib);nd=size(j.directions,1);s=zeros(ns,nd);sc=zeros(nc,ns,nd);t=tic;
 for idir=1:nd
  a=sum(moments(1:N,1:N,:).*reshape(j.directions(idir,:),1,1,3),3);
  for iseq=1:ns
   nu=evolve_laplace_coef(nu0(1:N),sequences{iseq},1i*qvalues(iseq)*a,LQT2(1:N),3);
   sc(:,iseq,idir)=component_projection(:,1:N)*nu/norm0;s(iseq,idir)=sum(sc(:,iseq,idir));
  end
 end
 propagation_seconds(ib)=toc(t);signals{ib}=s;components{ib}=sc;
 assert(max(abs(s(1,:)-1))<1e-8);assert(max(abs(imag(s)),[],'all')<1e-9);
 assert(max(abs(sc(:,1,:)-reshape(femesh.volumes/norm0,nc,1,1)),[],'all')<1e-8);
 if N==numel(lambda),assert(max(abs(s(:,1)-original_signal))<1e-9);assert(max(abs(sc(:,:,1)-original_components),[],'all')<1e-9);end
 fprintf('PROPAGATED %s N%d: %g seconds\n',tag,N,propagation_seconds(ib));
end
job=j;volumes=femesh.volumes;areas=femesh.areas;n_dof=sum(n);n_tetrahedra=sum(cellfun(@(t)size(t,2),femesh.elements));
save(target,'job','signals','components','budgets','propagation_seconds','eigen_seconds','original_MF_seconds','zero_count','lambda','volumes','areas','n_dof','n_tetrahedra','diagnostics','acquisition','-v7');
fprintf('SAVED %s DOF%d modes%d zero%d eigen%g seconds\n',tag,n_dof,numel(lambda),zero_count,eigen_seconds);
end


function run_global_stage2(tag)
% Input/QA adapter only. Original SpinDoctor FEM eigenproblem, original MF and
% original finite-PGSE evolve_laplace_coef are used without source changes.
[root,~]=paths_stage2();j=jsondecode(fileread(fullfile(root,'configs/jobs',string(tag)+".json")));
target=fullfile(root,'data',string(tag)+".mat");if isfile(target),return;end
raw=load(j.mesh_file);femesh=mesh_stage2(raw,j.transparent);
[femesh.volumes,femesh.areas]=get_vol_sa(femesh);femesh.total_volume=sum(femesh.volumes);femesh.total_area=sum(femesh.areas);
nc=femesh.ncompartment;nb=femesh.nboundary;
pde.diffusivity=repmat(j.D_um2_ms/1000*eye(3),1,1,nc);pde.initial_density=ones(1,nc);pde.relaxation=Inf(1,nc);
pde.permeability=zeros(1,nb);pde.permeability(1)=j.kappa_n_um_s/1e6;
if nc>2,pde.permeability(3:end)=j.kappa_org_um_s/1e6;end
pde.compartments=["nucleus","cytoplasm",repmat("organelle",1,nc-2)];pde.boundaries=["nucleus,cytoplasm","outer",repmat("organelle,cytoplasm",1,nb-2)];
pde.initial_signal=sum(femesh.volumes);pde.mean_diffusivity=compute_mean_diffusivity(pde.diffusivity,femesh);
setup.pde=pde;setup.gamma=j.gamma_rad_us_mT;
bs=[0;j.b_s_mm2(:)];de=[j.delta_ms(1);j.delta_ms(:)]*1000;De=[j.Delta_ms(1);j.Delta_ms(:)]*1000;
ns=numel(bs);sequences=cell(1,ns);for i=1:ns,sequences{i}=PGSE(de(i),De(i));end
setup.gradient.sequences=sequences;setup.gradient.values=0;setup.gradient.values_type="b";setup.gradient.directions=[1;0;0];
setup=prepare_experiments(setup);
for i=1:ns,setup.gradient.bvalues(i)=bs(i);setup.gradient.qvalues(i)=sqrt(bs(i)/sequences{i}.bvalue_no_q);end
setup.gradient.gvalues=setup.gradient.qvalues*1e6/setup.gamma;
setup.namplitude=1;setup.nsequence=ns;setup.ndirection=1;setup.ncompartment=nc;setup.nboundary=nb;
mf.neig_max=j.eigen_modes;mf.length_scale=0;mf.single=false;mf.gpu=false;mf.ninterval=3;mf.surf_relaxation=false;mf.rerun=true;mf.rerun_eigen=true;
mf.eigs.sigma=-1e-8;mf.eigs.tolerance=1e-10;mf.eigs.maxiter=1000;mf.eigs.subspacedimension=ceil(1.625*j.eigen_modes);setup.mf=mf;
% Independent assembly diagnostics, including the entire stationary subspace.
n=cellfun(@(p)size(p,2),femesh.points);offset=[0,cumsum(n)];Mcell=cell(1,nc);Kcell=cell(1,nc);
for ic=1:nc
 t=femesh.elements{ic};p=femesh.points{ic};[~,v]=get_volume_mesh(p,t);
 Mcell{ic}=mass_matrixP1_3D(t',v');Kcell{ic}=stiffness_matrixP1_3D(t',p',pde.diffusivity(:,:,ic));
end
M=blkdiag(Mcell{:});K=blkdiag(Kcell{:});QB=assemble_flux_matrix(femesh.points,femesh.facets);Q=couple_flux_matrix(femesh,pde,QB,false);A=K+Q;
if j.transparent || j.kappa_org_um_s>0
 C=ones(sum(n),1);expected_zeros=1;
else
 C=zeros(sum(n),nc-1);C(1:offset(3),1)=1;
 for ic=3:nc,C(offset(ic)+1:offset(ic+1),ic-1)=1;end;expected_zeros=nc-1;
end
diagnostics.stationary_residual_max=max(abs(A*C),[],'all');assert(diagnostics.stationary_residual_max<1e-10);
diagnostics.flux_symmetric_error=norm(Q-Q',inf);assert(diagnostics.flux_symmetric_error<1e-12);
en=zeros(sum(n),1);en(1:n(1))=1;diagnostics.nuclear_jump_energy=en'*Q*en;
diagnostics.recovered_kappa_n_um_s=diagnostics.nuclear_jump_energy/femesh.areas(1)*1e6;
assert(abs(diagnostics.recovered_kappa_n_um_s-j.kappa_n_um_s)<1e-8);
diagnostics.recovered_organelle_permeabilities_um_s=zeros(1,nc-2);
for ic=3:nc
 e=zeros(sum(n),1);e(offset(ic)+1:offset(ic+1))=1;
 diagnostics.recovered_organelle_permeabilities_um_s(ic-2)=e'*Q*e/femesh.areas(ic)*1e6;
end
if nc>2,assert(max(abs(diagnostics.recovered_organelle_permeabilities_um_s-j.kappa_org_um_s))<1e-8);end
assert(norm(A-A',inf)<1e-10);
eigfile=fullfile(root,'cache',string(tag)+"_eigen.mat");tic;
if isfile(eigfile),saved=load(eigfile);assert(strcmp(saved.job_identity,j.identity));lap_eig=saved.lap_eig;eigen_seconds=saved.eigen_seconds;
else
 rng(j.eigen_seed,'twister');et=tic;lap_eig=compute_laplace_eig(femesh,pde,mf);eigen_seconds=toc(et);
 job_identity=j.identity;tmp=fullfile(root,'cache',string(tag)+"_eigen_partial.mat");
 save(tmp,'lap_eig','eigen_seconds','job_identity','-v7.3');movefile(tmp,eigfile);
end
lambda=lap_eig.values;V=lap_eig.funcs;zero_count=sum(abs(lambda)<1e-11);assert(zero_count==expected_zeros);
diagnostics.mass_orthogonality_max=max(abs(V'*M*V-eye(numel(lambda))),[],'all');assert(diagnostics.mass_orthogonality_max<1e-8);
cc=V'*M*C;diagnostics.stationary_projection_max=max(abs(V*cc-C),[],'all');assert(diagnostics.stationary_projection_max<1e-7);
diagnostics.no_gradient_100ms_max=max(abs(V*(exp(-lambda*1e5).*cc)-C),[],'all');assert(diagnostics.no_gradient_100ms_max<1e-7);
diagnostics.eigenvalue_min=lambda(1);diagnostics.first_positive_eigenvalue=lambda(zero_count+1);
% Original MF computes coordinate matrices and one complete protocol along x.
cachepath=fullfile(root,'cache',string(tag)+"_mf");t=tic;logtext=evalc('original=solve_mf_cpu(femesh,setup,lap_eig,cachepath,false);');original_MF_seconds=toc(t);
f=fopen(fullfile(root,'logs',string(tag)+"_original_MF.log"),'w');fprintf(f,'%s',logtext);fclose(f);
cachepath=add_mf_str_savepath(cachepath,mf);op=load(fullfile(cachepath,'FEM_lap_eig.mat'));moments=op.moments;nu0=op.nu0;LQT2=op.LQT2;
organelle_participation=zeros(nc-2,numel(lambda));
local_dipole_coefficients=cell(1,nc-2);local_dipole_norms=cell(1,nc-2);
for ic=3:nc
 inds=offset(ic)+1:offset(ic+1);Vi=V(inds,:);Mi=Mcell{ic};
 organelle_participation(ic-2,:)=sum(Vi.*(Mi*Vi),1);
 U=femesh.points{ic}'-reshape(j.centers_um(ic-2,:),1,3);
 local_dipole_coefficients{ic-2}=Vi'*Mi*U;
 local_dipole_norms{ic-2}=sum(U.*(Mi*U),1);
end
component_projection=zeros(nc,numel(lambda));
for ic=1:nc,component_projection(ic,:)=sum(Mcell{ic}*V(offset(ic)+1:offset(ic+1),:),1);end
diagnostics.no_gradient_population_error=max(abs(component_projection*(exp(-lambda*1e5).*nu0)/pde.initial_signal-femesh.volumes(:)/pde.initial_signal));
assert(diagnostics.no_gradient_population_error<1e-8);
norm0=pde.initial_signal;original_signal=reshape(original.signal_allcmpts,ns,1)/norm0;
original_components=reshape(original.signal,nc,ns)/norm0;
qvalues=setup.gradient.qvalues;acquisition=struct('b_s_mm2',bs,'q_um_inv_us_inv',qvalues,'G_mT_m',setup.gradient.gvalues,'delta_us',de,'Delta_us',De);
operatorfile=fullfile(root,'data',string(tag)+"_operators.mat");
save(operatorfile,'lambda','moments','nu0','LQT2','component_projection','norm0','acquisition','diagnostics','zero_count','original_signal','original_components','original_MF_seconds','eigen_seconds','organelle_participation','local_dipole_coefficients','local_dipole_norms','-v7');
% Reuse original propagator, with nested submatrices of the SAME basis. No
% new eigensolve, approximation, or waveform discretization is introduced.
budgets=j.budgets(:)';signals=cell(size(budgets));components=cell(size(budgets));propagation_seconds=zeros(size(budgets));
for ib=1:numel(budgets)
 N=budgets(ib);nd=size(j.directions,1);s=zeros(ns,nd);sc=zeros(nc,ns,nd);t=tic;
 for idir=1:nd
  a=sum(moments(1:N,1:N,:).*reshape(j.directions(idir,:),1,1,3),3);
  for iseq=1:ns
   nu=evolve_laplace_coef(nu0(1:N),sequences{iseq},1i*qvalues(iseq)*a,LQT2(1:N),3);
   sc(:,iseq,idir)=component_projection(:,1:N)*nu/norm0;s(iseq,idir)=sum(sc(:,iseq,idir));
  end
 end
 propagation_seconds(ib)=toc(t);signals{ib}=s;components{ib}=sc;
 assert(max(abs(s(1,:)-1))<1e-8);assert(max(abs(imag(s)),[],'all')<1e-9);
 assert(max(abs(sc(:,1,:)-reshape(femesh.volumes/norm0,nc,1,1)),[],'all')<1e-8);
 if N==numel(lambda),assert(max(abs(s(:,1)-original_signal))<1e-9);assert(max(abs(sc(:,:,1)-original_components),[],'all')<1e-9);end
 fprintf('PROPAGATED %s N%d: %g seconds\n',tag,N,propagation_seconds(ib));
end
job=j;volumes=femesh.volumes;areas=femesh.areas;n_dof=sum(n);n_tetrahedra=sum(cellfun(@(t)size(t,2),femesh.elements));
save(target,'job','signals','components','budgets','propagation_seconds','eigen_seconds','original_MF_seconds','zero_count','lambda','volumes','areas','n_dof','n_tetrahedra','diagnostics','acquisition','-v7');
fprintf('SAVED %s DOF%d modes%d zero%d eigen%g seconds\n',tag,n_dof,numel(lambda),zero_count,eigen_seconds);
end


function check=sd_conservation(femesh,setup)
% Diagnostic only: original assembly functions, no alternative propagation.
n=cellfun(@(x)size(x,2),femesh.points);K=cell(1,numel(n));
connectivity=zeros(1,numel(n));
for ic=1:numel(n)
 t=femesh.elements{ic};p=femesh.points{ic};
 K{ic}=stiffness_matrixP1_3D(t',p',setup.pde.diffusivity(:,:,ic));
 edges=[t([1,2],:),t([1,3],:),t([1,4],:),t([2,3],:),t([2,4],:),t([3,4],:)];
 connectivity(ic)=max(conncomp(graph(edges(1,:),edges(2,:))));
end
Kb=blkdiag(K{:});Qb=assemble_flux_matrix(femesh.points,femesh.facets);
Q=couple_flux_matrix(femesh,setup.pde,Qb,true);A=Kb+Q;
v=ones(sum(n),1);offset=[0,cumsum(n)];individual=zeros(1,numel(n));
for ic=1:numel(n)
 e=zeros(sum(n),1);e(offset(ic)+1:offset(ic+1))=1;
 individual(ic)=norm(A*e,inf);
end
if numel(n)==2
 tri=femesh.facets{1,1};p=femesh.points{1};
 u=p(:,tri(2,:))-p(:,tri(1,:));w=p(:,tri(3,:))-p(:,tri(1,:));
 area=sum(vecnorm(cross(u,w,1),2,1))/2;
 e=zeros(sum(n),1);e(1:n(1))=1;
 check.nuclear_interface_area=area;check.interface_jump_energy=e'*Q*e;
 check.recovered_kappa_um_ms=1000*check.interface_jump_energy/area;
else
 check.nuclear_interface_area=0;check.interface_jump_energy=0;check.recovered_kappa_um_ms=0;
end
check.global_stationary_residual=norm(A*v,inf);
check.individual_stationary_residuals=individual;
check.connected_components_per_region=connectivity;
check.permeability_matrix_symmetric_error=norm(Q-Q',inf);
check.volumes=femesh.volumes;
check.equilibrium_volume_weights=femesh.volumes/sum(femesh.volumes);
end


function [setup,femesh]=setup_stage2(j,femesh)
[femesh.volumes,femesh.areas]=get_vol_sa(femesh);femesh.total_volume=sum(femesh.volumes);femesh.total_area=sum(femesh.areas);
nc=femesh.ncompartment;nb=femesh.nboundary;
pde.diffusivity=repmat(j.D_um2_ms/1000*eye(3),1,1,nc);pde.initial_density=ones(1,nc);pde.relaxation=Inf(1,nc);
pde.permeability=zeros(1,nb);
if nc==1
 pde.compartments="organelle";pde.boundaries="organelle_reflecting";
else
 pde.permeability(1)=j.kappa_n_um_s/1e6;
 if nc>2,pde.permeability(3:end)=j.kappa_org_um_s/1e6;end
 pde.compartments=["nucleus","cytoplasm",repmat("organelle",1,nc-2)];
 pde.boundaries=["nucleus,cytoplasm","outer",repmat("organelle,cytoplasm",1,nb-2)];
end
pde.initial_signal=sum(femesh.volumes);pde.mean_diffusivity=compute_mean_diffusivity(pde.diffusivity,femesh);
setup.pde=pde;setup.gamma=j.gamma_rad_us_mT;
bs=[0;j.b_s_mm2(:)];de=[j.delta_ms(1);j.delta_ms(:)]*1000;De=[j.Delta_ms(1);j.Delta_ms(:)]*1000;
ns=numel(bs);sequences=cell(1,ns);for i=1:ns,sequences{i}=PGSE(de(i),De(i));end
setup.gradient.sequences=sequences;setup.gradient.values=0;setup.gradient.values_type="b";setup.gradient.directions=[1;0;0];
setup=prepare_experiments(setup);
for i=1:ns,setup.gradient.bvalues(i)=bs(i);setup.gradient.qvalues(i)=sqrt(bs(i)/sequences{i}.bvalue_no_q);end
setup.gradient.gvalues=setup.gradient.qvalues*1e6/setup.gamma;
setup.namplitude=1;setup.nsequence=ns;setup.ndirection=1;setup.ncompartment=nc;setup.nboundary=nb;
mf.neig_max=j.eigen_modes;mf.length_scale=0;mf.single=false;mf.gpu=false;mf.ninterval=3;mf.surf_relaxation=false;mf.rerun=true;mf.rerun_eigen=true;
mf.eigs.sigma=-1e-8;mf.eigs.tolerance=1e-10;mf.eigs.maxiter=1000;mf.eigs.subspacedimension=ceil(1.625*j.eigen_modes);setup.mf=mf;
end

