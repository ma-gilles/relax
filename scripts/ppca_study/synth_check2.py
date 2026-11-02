"""Positive control with RECOVAR's own projector and CTF, so no convention can be wrong.
For the first 1024 images: true signal and noise are known exactly (noise_probe signal, stack minus signal).
Estimator: correlate the stack with CTF x slice of the weighted mean map at true poses; SNR = c^2/(1-c^2).
Also compares the projector with the noise-free signal, shell by shell, to show any hidden B-factor."""
import os,json,numpy as np,mrcfile
os.environ['JAX_PLATFORMS']='cpu'; os.environ['CUDA_VISIBLE_DEVICES']=''
import jax.numpy as jnp
from recovar import core
from recovar.core import fourier_transform_utils as ftu
from recovar.core import geometry
R='/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_suite_defocus_20261030/datasets/'
N=128; shp=(N,N); vs=(N,N,N); out={}
kk=np.asarray(ftu.get_k_coordinate_of_each_pixel(shp,voxel_size=1,scaled=False)); sh=np.rint(np.linalg.norm(kk,axis=-1)).astype(int).reshape(N,N); M=N//2; L=int(sh.max())+2
bc=lambda x:np.bincount(sh.ravel(),x.ravel(),L)[:M]
print('sh',sh[N//2,N//2],sh[N//2,N//2+3],flush=True)
for name,a in (('H1__realistic__noise1__b128',3.0),('C__realistic__noise1__b128',3.0),('C__realistic__noise3__b128',3.0),('D40__realistic__noise1__b128',4.5)):
    e=R+name+'/evaluation/'; t=np.load(e+'truth.npz',allow_pickle=True); cnt=t['counts'].astype(float); Rot=t['rotations']; ctf=t['ctf']
    V=jnp.asarray((t['volumes_fourier']*(cnt/cnt.sum())[:,None]).sum(0))
    pr=np.load(e+'noise_probe.npz'); sig=pr['signal']; ids=pr['particle_ids']
    mm=mrcfile.mmap(R+name+'/training/particles.mrcs',permissive=True)
    ev=core.CTFEvaluator(); num=np.zeros(M);den=np.zeros(M);yy=np.zeros(M);ss=np.zeros(M);nn=np.zeros(M);ps=np.zeros(M);ppss=np.zeros(M)
    for b in range(0,len(ids),64):
        j=ids[b:b+64]
        C=np.asarray(ev(jnp.asarray(ctf[j]),shp,a)).reshape(len(j),N,N)
        sl=np.asarray(core.slice_volume(V,jnp.asarray(Rot[j]),shp,vs,'linear_interp')).reshape(len(j),N,N)
        p=C*sl
        Y=np.asarray(ftu.get_dft2(jnp.asarray(np.asarray(mm.data[j],np.float32)))).reshape(len(j),N,N)
        Sg=np.asarray(ftu.get_dft2(jnp.asarray(sig[b:b+64]))).reshape(len(j),N,N)
        Nz=Y-Sg
        for q in range(len(j)):
            num+=bc((np.conj(p[q])*Y[q]).real);den+=bc(np.abs(p[q])**2);yy+=bc(np.abs(Y[q])**2);ss+=bc(np.abs(Sg[q])**2);nn+=bc(np.abs(Nz[q])**2);ps+=bc((np.conj(p[q])*Sg[q]).real)
    c=num/np.sqrt(den*yy); est=c*np.abs(c)/(1-c*c); tru=ss/nn; cps=ps/np.sqrt(den*ss); ratio=ps/den
    fm=np.arange(M)/(N*a)
    out[name]=dict(k=fm.tolist(),est=est.tolist(),true=tru.tolist(),corr=c.tolist(),proj_vs_signal_corr=cps.tolist(),alpha_signal=ratio.tolist())
    sel=[3,5,8,11,15,20,30,40,50]
    print(name,'res',[round(float(1/fm[i]),1) for i in sel])
    print(' est ',['%.3g'%est[i] for i in sel]); print(' true',['%.3g'%tru[i] for i in sel]); print(' proj-signal corr',['%.3f'%cps[i] for i in sel]); print(' alpha signal/projector',['%.3f'%ratio[i] for i in sel],flush=True)
    json.dump(out,open('/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/snr_curves_20261102/synth_check2.json','w'))
