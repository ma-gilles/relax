import numpy as np,json
R='/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round16_lrsched_20261031/runs/'
N=128
def load(a):
    t=np.load(R+a+'/checkpoint_3000.npz',allow_pickle=True)['theta'];return t.reshape(N,N,N//2+1,-1)
T=load('C__realistic__noise1__b128__sinv065_s11')
i=np.unravel_index(np.argmax(np.abs(T[...,0])),T.shape[:3]);print('DC at',i)
f=np.arange(N)-N//2 if i[0]==N//2 else np.fft.fftfreq(N,1/N)
z=np.arange(N//2+1)
sh=np.rint(np.sqrt(f[:,None,None]**2+f[None,:,None]**2+z[None,None,:]**2)).astype(int)
w=np.ones(sh.shape);w[:,:,1:-1]=2  # half grid weights
M=N//2
b=lambda p:np.bincount(sh.ravel(),(p*w).ravel(),int(sh.max())+1)[:M]
def spec(a):
    t=load(a);return b(np.abs(t[...,0])**2),b((np.abs(t[...,1:])**2).sum(-1))
g=np.load('/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_suite_defocus_20261030/datasets/C__realistic__noise1__b128/evaluation/gt_subspace.npz')
Fm=np.fft.fftshift(np.fft.rfftn(g['mean']),axes=(0,1)) if i[0]==N//2 else np.fft.rfftn(g['mean'])
tm=b(np.abs(Fm)**2)
tp=np.zeros(M)
for l,p in zip(g['eigenvalues'],g['pcs']):
    F=np.fft.fftshift(np.fft.rfftn(p),axes=(0,1)) if i[0]==N//2 else np.fft.rfftn(p);tp+=l*b(np.abs(F)**2)
print('true PC/mean: all shells %.3f, shells 1-31 %.3f, 1-15 %.3f'%(tp.sum()/tm.sum(),tp[1:32].sum()/tm[1:32].sum(),tp[1:16].sum()/tm[1:16].sum()))
arms=[('SGD .65 C','C__realistic__noise1__b128__sinv065_s11'),('SGD lr2.4 C','C__realistic__noise1__b128__sgdsched64_s11'),('VDAM-s C','C__realistic__noise1__b128__vsimplesched64_s11'),('SGD lr2.4 H1','H1__realistic__noise1__b128__sgdsched64_s11'),('VDAM-s H1','H1__realistic__noise1__b128__vsimplesched64_s11')]
bands=[(1,5),(6,10),(11,15),(16,20),(21,25),(26,31)]
print('band       ',bands)
print('true PC/mean',['%.3f'%(tp[a:b_+1].sum()/tm[a:b_+1].sum()) for a,b_ in bands])
for n,a in arms:
    m,p=spec(a)
    print(n.ljust(12),'PC/mean all %.3f'%(p[1:32].sum()/m[1:32].sum()),['%.3f'%(p[a_:b_+1].sum()/m[a_:b_+1].sum()) for a_,b_ in bands])
    if ' C' in n:
        scale=np.median((m[2:12]/tm[2:12]));print('   model mean / true mean, shells: ',['%.2f'%((m[a_:b_+1].sum()/tm[a_:b_+1].sum())/scale) for a_,b_ in bands],'unit',round(float(scale),4))
        print('   model PC / true PC abs: ',['%.2f'%((p[a_:b_+1].sum()/tp[a_:b_+1].sum())/scale) for a_,b_ in bands],'total %.2f'%(p[1:32].sum()/tp[1:32].sum()/scale))
