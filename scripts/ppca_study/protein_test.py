"""Protein-only test: half-map cross power against the atomic model, with and without the solvent correction."""
import os,gzip,json,numpy as np,mrcfile,sys
os.environ['JAX_PLATFORMS']='cpu'; os.environ['CUDA_VISIBLE_DEVICES']=''
from recovar.simulation import simulate_scattering_potential as ssp
O='/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/snr_curves_20261102/'
def atoms(id):
    t=gzip.open(O+'pdb_dl/%s.cif.gz'%id,'rt').read().splitlines();cols=[];out=[]
    for l in t:
        if l.startswith('_atom_site.'): cols.append(l.split('.')[1].strip())
        elif l.startswith(('ATOM','HETATM')) and cols:
            p=l.split();out.append((p[cols.index('Cartn_x')],p[cols.index('Cartn_y')],p[cols.index('Cartn_z')],p[cols.index('type_symbol')],p[cols.index('label_alt_id')],p[cols.index('label_comp_id')]))
    r=[x for x in out if x[4] in ('.','A') and x[5]!='HOH']
    xyz=np.array([x[:3] for x in r],float);el=np.array([x[3].upper() for x in r]);ok=np.isin(el,['C','N','O','S','P'])
    print(id,'atoms',len(r),'kept',ok.sum(),flush=True);return xyz[ok]-xyz[ok].mean(0),el[ok]
def halfmap(e):
    def rd(h):
        with gzip.open(O+'emdb_dl/emd_%s_half_map_%d.map.gz'%(e,h),'rb') as f:
            open(O+'emdb_dl/tmp_%s_%d.mrc'%(e,h),'wb').write(f.read())
        m=mrcfile.open(O+'emdb_dl/tmp_%s_%d.mrc'%(e,h),permissive=True);d=np.asarray(m.data,np.float32);a=float(m.voxel_size.x);m.close();os.remove(O+'emdb_dl/tmp_%s_%d.mrc'%(e,h));return d,a
    a,ap=rd(1);b,_=rd(2);N=a.shape[0]
    Fa=np.fft.rfftn(a);Fb=np.fft.rfftn(b)
    f=np.fft.fftfreq(N,1/N);r=np.fft.rfftfreq(N,1/N);sh=np.rint(np.sqrt(f[:,None,None]**2+f[None,:,None]**2+r[None,None,:]**2)).astype(int).ravel()
    w=np.ones(Fa.shape,np.float32);w[:,:,1:-1]=2
    M=N//2;c=np.bincount(sh,w.ravel(),M+200)[:M];p=np.bincount(sh,((Fa*np.conj(Fb)).real*w).ravel(),M+200)[:M]
    return np.arange(M)/(N*ap),p/c,N,ap
H=lambda q,aa=.8,BB=2000:(1-aa*np.exp(-BB*q**2/4))**2
sm=lambda y,w=3:np.exp(np.convolve(np.log(np.maximum(y,1e-30)),np.ones(w)/w,'same'))
at=(150,100,70,50,35,25,20,15,10,7)
res={}
for name,e,id in (('Beta-galactosidase EMD-2984, 5A1A','2984','5A1A'),('Apoferritin EMD-11103, 6Z6U','11103','6Z6U')):
    if not os.path.exists(O+'emdb_dl/emd_%s_half_map_2.map.gz'%e): continue
    kr,Sr,Nr,ap=halfmap(e);print(name,'box',Nr,'apix',ap,'box A',Nr*ap,flush=True)
    xyz,el=atoms(id);N=200;a=4.0
    v=np.fft.ifftshift(np.asarray(ssp.generate_volume_from_atom_positions_and_types(xyz,el,voxel_size=a,grid_size=N)).reshape(N,N,N))
    f=np.fft.fftfreq(N,1/N);sh=np.rint(np.sqrt(f[:,None,None]**2+f[None,:,None]**2+f[None,None,:]**2)).astype(int).ravel();M=N//2
    P=(np.bincount(sh,(np.abs(v)**2).ravel(),int(sh.max())+1)[:M]/np.bincount(sh,minlength=M)[:M]);k=np.arange(M)/(N*a)
    i=lambda kk,x:np.argmin(abs(kk-1/x))
    print(' '*26,' '.join(str(x).rjust(8) for x in at))
    def row(l,kk,y):
        y=sm(np.maximum(y,1e-30));print(l.ljust(26),' '.join(('%8.2f'%(y[i(kk,x)]/y[i(kk,20)])) if kk.max()>1/x and (1/kk[1])>=x*.7 else '       -' for x in at),flush=True)
    row('real half-map cross',kr,Sr);row('model, no correction',k,P);row('model + a=.8',k,P*H(k));row('no corr, B 150',k,P*np.exp(-150*k**2/2));row('model + a=.8, B 150',k,P*H(k)*np.exp(-150*k**2/2));row('model + a=.8, B 50',k,P*H(k)*np.exp(-50*k**2/2));row('no corr, B 50',k,P*np.exp(-50*k**2/2))
    res[name]=dict(kr=kr.tolist(),S=Sr.tolist(),k=k.tolist(),P=P.tolist());json.dump(res,open(O+'protein_test.json','w'))
