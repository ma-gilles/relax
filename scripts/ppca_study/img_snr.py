"""Image-domain per-shell SNR: correlate images with CTF-weighted slices of the refined map at RELION's poses.
SNR(k) = c^2 / (1 - c^2), c the shell correlation. A lower bound: pose error, interpolation and heterogeneity all lower c."""
import numpy as np,mrcfile,pickle,json,sys,collections,os
import starfile
from scipy.ndimage import map_coordinates
def A(r,t,p):
    r,t,p=np.deg2rad([r,t,p]); ca,sa,cb,sb,cg,sg=np.cos(r),np.sin(r),np.cos(t),np.sin(t),np.cos(p),np.sin(p)
    cc,cs,sc,ss=cb*ca,cb*sa,sb*ca,sb*sa
    return np.array([[cg*cc-sg*sa,cg*cs+sg*ca,-cg*sb],[-sg*cc-cg*sa,-sg*cs+cg*ca,sg*sb],[sc,ss,cb]])
def run(V,a,get,n,kv,cs,w,name):
    N=V.shape[0]; Vs=np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(V)))
    f=np.fft.fftfreq(N,1/N); Ky,Kx=np.meshgrid(f,f,indexing='ij'); sh=np.rint(np.sqrt(Kx**2+Ky**2)).astype(int); M=N//2; L=int(sh.max())+2
    lam=12.2639/np.sqrt(kv*1e3+0.97845*kv*kv); C=cs*1e7
    num=np.zeros(M);den=np.zeros(M);yy=np.zeros(M); P=np.stack([Kx.ravel(),Ky.ravel(),np.zeros(Kx.size)])
    for i in range(n):
        y,e,o,dfu,dfv,ang=get(i)
        Y=np.fft.fft2(np.fft.ifftshift(y)); k3=A(*e).T@P; q=np.stack([k3[2],k3[1],k3[0]])+N//2
        sl=(map_coordinates(Vs.real,q,order=1,mode='constant')+1j*map_coordinates(Vs.imag,q,order=1,mode='constant')).reshape(N,N)
        s2=(Kx**2+Ky**2)/(N*a)**2; phi=np.arctan2(Ky,Kx); d=.5*(dfu+dfv)+.5*(dfu-dfv)*np.cos(2*(phi-np.deg2rad(ang)))
        chi=np.pi*lam*d*s2-.5*np.pi*C*lam**3*s2**2; ct=np.sqrt(1-w*w)*np.sin(chi)+w*np.cos(chi)
        p=ct*sl*np.exp(2j*np.pi*(Kx*o[0]+Ky*o[1])/N)
        num+=np.bincount(sh.ravel(),(np.conj(p)*Y).real.ravel(),L)[:M]; den+=np.bincount(sh.ravel(),(np.abs(p)**2).ravel(),L)[:M]; yy+=np.bincount(sh.ravel(),(np.abs(Y)**2).ravel(),L)[:M]
    c=num/np.sqrt(den*yy); k=np.arange(M)/(N*a); sn=c*np.abs(c)/(1-c*c)
    print(name,'n',n,'snr at 100,50,35,20,10 A',[float('%.3g'%np.interp(1/x,k,np.convolve(sn,np.ones(3)/3,'same'))) for x in (100,50,35,20,10)],'alpha 5,10,15',[float('%.3g'%(num[i]/den[i])) for i in (5,10,15)],flush=True)
    return dict(k=k.tolist(),snr=sn.tolist(),corr=c.tolist(),n=n)
out={}
F='/scratch/gpfs/CRYOEM/gilleslab/em_work/ws2_gated_sgd_20261006/fixtures/empiar10076_full_b128/'
c=np.asarray(pickle.load(open(F+'training/ctf.pkl','rb'))); E=np.load(F+'evaluation/relion_refine_eulers_deg.npy'); Off=np.load(F+'evaluation/relion_refine_offsets_px.npy')
sel=np.linspace(0,len(c)-1,1500).astype(int); mm=mrcfile.mmap(F+'training/particles.mrcs',permissive=True)
out['EMPIAR-10076 ribosome']=run(np.asarray(mrcfile.open(F+'evaluation/reference_b128.mrc',permissive=True).data,np.float64),3.275,lambda i:(np.asarray(mm.data[sel[i]],np.float64),E[sel[i]],Off[sel[i]],c[sel[i],2],c[sel[i],3],c[sel[i],4]),len(sel),300.,2.7,.07,'10076')
json.dump(out,open('img_snr.json','w'))
d='/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/empiar_10081_hcn1/relion_refine_c4/'
D=starfile.read(d+'run_data.star',always_dict=True); P=D['particles']; O=D['optics'].iloc[0]; a=float(O['rlnImagePixelSize'])
P=P.sample(frac=1,random_state=0); rows=[]; 
for _,r in P.iterrows():
    i,fn=r['rlnImageName'].split('@'); 
    for b in ('/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/empiar_10081_hcn1/','/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_fixtures/empiar_10081_hcn1/relion_refine_c4/'):
        if os.path.exists(b+fn): rows.append((b+fn,int(i)-1,r)); break
    if len(rows)>=800: break
print('hcn1 images',len(rows),flush=True)
def gh(i):
    fn,j,r=rows[i]
    with mrcfile.mmap(fn,permissive=True) as m: y=np.asarray(m.data[j],np.float64)
    return y,(r['rlnAngleRot'],r['rlnAngleTilt'],r['rlnAnglePsi']),(r['rlnOriginXAngst']/a,r['rlnOriginYAngst']/a),r['rlnDefocusU'],r['rlnDefocusV'],r['rlnDefocusAngle']
if rows:
    out['EMPIAR-10081 HCN1']=run(np.asarray(mrcfile.open(d+'run_class001.mrc',permissive=True).data,np.float64),a,gh,len(rows),float(O['rlnVoltage']),float(O['rlnSphericalAberration']),float(O['rlnAmplitudeContrast']),'HCN1')
    json.dump(out,open('img_snr.json','w'))
