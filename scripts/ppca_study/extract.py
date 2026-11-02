"""Per-update step, fudge, PC power and shell-wise phi for VDAM-simple and SGD arms. Reads iterations.jsonl only."""
import json,numpy as np,pickle,sys
T='/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/round16_lrsched_20261031/runs/'
arms=dict(C1v='C__realistic__noise1__b128__vsimplesched64_s11',C1v2='C__realistic__noise1__b128__vsimplesched64_s12',C3v='C__realistic__noise3__b128__vsimplesched64_s11',D40v='D40__realistic__noise1__b128__vsimplesched64_s11',T10v='T10__realistic__noise1__b128__vsimplesched64_s11',H1v='H1__realistic__noise1__b128__vsimplesched64_s11',
 C1s='C__realistic__noise1__b128__sinv065_s11',C3s='C__realistic__noise3__b128__sinv065_s11',D40s='D40__realistic__noise1__b128__sinv065_s11',T10s='T10__realistic__noise1__b128__sinv065_s11',H1s='H1__realistic__noise1__b128__sgdsched64_s11')
out={}
for k,a in arms.items():
    try: L=open(T+a+'/iterations.jsonl')
    except Exception as e: print(k,'missing'); continue
    R=dict(it=[],step=[],fudge=[],radius=[],hp=[],mp=[],lp=[],sv=[],phi=[],sig=[],dis=[],phim=[],grad=[])
    for l in L:
        d=json.loads(l); R['it'].append(d['iteration']);R['step'].append(d.get('step',np.nan));R['fudge'].append(d.get('fudge',d.get('vdam_scheduled_fudge_unused',np.nan)));R['radius'].append(d['radius']);R['hp'].append(d['healpix_order']);R['mp'].append(d.get('mean_power',np.nan));R['lp'].append(d.get('loading_power',np.nan));R['sv'].append(d.get('loading_singular_values',[np.nan])[:3]+[np.nan]*(3-len(d.get('loading_singular_values',[np.nan])[:3])))
        g=d.get('gates_applied');
        if g is not None:
            g=np.array(g,float);r=d['radius'];R['phi'].append(g[:64,1]);R['phim'].append(g[:64,0]);R['sig'].append(np.array(d['signal'],float)[:64]);R['dis'].append(np.array(d['disagreement'],float)[:64])
    R={k2:(np.array(v,dtype=float) if len(v) else None) for k2,v in R.items()}
    out[k]=R; print(k,len(R['it']),'keys ok',flush=True)
pickle.dump(out,open('/scratch/gpfs/CRYOEM/gilleslab/em_work/ppca_coupled_20261026/vdam_vs_sgd_20261102/ext.pkl','wb'))
sel=[1,50,125,126,250,500,750,1000,1250,1500,1625,1626,2000,2500,3000]
for k,R in out.items():
    print('==',k,'upd step fudge rad hp meanP loadP s1')
    for t in sel:
        i=np.searchsorted(R['it'],t)
        if i>=len(R['it']): continue
        print(int(R['it'][i]),'%.3f %.3f %d %d %.3g %.3g %.3g'%(R['step'][i],R['fudge'][i],R['radius'][i],R['hp'][i],R['mp'][i],R['lp'][i],R['sv'][i][0]))
