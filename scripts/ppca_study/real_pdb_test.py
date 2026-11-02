"""Direct test of the solvent correction: atomic model of each real particle, with and without the correction, against the spectrum of its real map."""
import os,gzip,json,numpy as np
os.environ['JAX_PLATFORMS']='cpu'; os.environ['CUDA_VISIBLE_DEVICES']=''
from recovar.simulation import simulate_scattering_potential as ssp
O='/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/snr_curves_20261102/'
def atoms(id,keep=None):
    t=gzip.open(O+'pdb_dl/%s.cif.gz'%id,'rt').read().splitlines();cols=[];out=[]
    for l in t:
        if l.startswith('_atom_site.'): cols.append(l.split('.')[1].strip())
        elif l.startswith(('ATOM','HETATM')) and cols:
            p=l.split();out.append((p[cols.index('Cartn_x')],p[cols.index('Cartn_y')],p[cols.index('Cartn_z')],p[cols.index('type_symbol')],p[cols.index('auth_asym_id')],p[cols.index('label_alt_id')]))
    r=[x for x in out if x[5] in ('.','A') and (keep is None or keep(x[4]))]
    xyz=np.array([x[:3] for x in r],float);el=np.array([x[3].upper() for x in r])
    ok=np.isin(el,['C','N','O','S','P','H']);print(id,'atoms',len(r),'kept',ok.sum(),'chains',len(set(x[4] for x in r)),flush=True)
    return xyz[ok]-xyz[ok].mean(0),el[ok]
N=200;a=4.0
f=np.fft.fftfreq(N,1/N);kz,ky,kx=np.meshgrid(f,f,f,indexing='ij');sh=np.rint(np.sqrt(kx**2+ky**2+kz**2)).astype(int).ravel();M=N//2
cnt=np.bincount(sh,minlength=M)[:M].astype(float)
k=np.arange(M)/(N*a)
R=json.load(open(O+'snr_curves.json'))
sets=(('EMPIAR-10076 ribosome','4YBB',lambda c:c.startswith('C')),('EMPIAR-10180 spliceosome','5NRL',None),('EMPIAR-10081 HCN1','5U6O',None))
H=lambda q,aa=.8,BB=2000:(1-aa*np.exp(-BB*q**2/4))**2
at=(150,100,70,50,35,25,20,15,10)
res={}
sm=lambda y,w=3:np.exp(np.convolve(np.log(np.maximum(y,1e-30)),np.ones(w)/w,'same'))
for name,id,keep in sets:
    xyz,el=atoms(id,keep)
    v=np.asarray(ssp.generate_volume_from_atom_positions_and_types(xyz,el,voxel_size=a,grid_size=N)).reshape(N,N,N)
    v=np.fft.ifftshift(v)  # recovar grid is centred
    print('peak at',np.unravel_index(np.argmax(np.abs(v)),v.shape),flush=True)
    P=np.bincount(sh,(np.abs(v)**2).ravel(),int(sh.max())+1)[:M]/cnt
    P[0]=P[1]
    r=R[name];kr=np.array(r['k']);Sr=sm(np.array(r['S'],float))
    i=lambda kk,x:np.argmin(abs(kk-1/x))
    print(name,'S(k)/S(20 A)');print(' '*26,' '.join(str(x).rjust(7) for x in at))
    def row(lab,kk,y):
        y=sm(y);print(lab.ljust(26),' '.join('%7.2f'%(y[i(kk,x)]/y[i(kk,20)]) for x in at),flush=True)
    row('real map',kr,Sr);row('model, no correction',k,P);row('model + a=.8',k,P*H(k));row('model + a=.72',k,P*H(k,.72));row('model + a=.8, B 150',k,P*H(k)*np.exp(-150*k**2/2));row('no corr, B 150',k,P*np.exp(-150*k**2/2))
    res[name]=dict(k=k.tolist(),P=P.tolist(),kr=kr.tolist(),S=Sr.tolist())
    json.dump(res,open(O+'real_pdb_test.json','w'))
