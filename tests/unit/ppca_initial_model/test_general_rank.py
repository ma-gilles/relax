"""Independent rank4/10 posterior, moment and optimizer contracts."""
import jax.numpy as jnp
import numpy as np
import pytest
from recovar.ppca.triangular import pack_upper_tri, unpack_tri_to_full

from relax.ppca_initial_model.config import Config
from relax.ppca_initial_model.sgd_update import momentum_step
from relax.ppca_initial_model.update import coupled_direction, empty_moments, stochastic_update
from relax.ppca_refinement.engine import (
    dense_pose_ppca_score_with_moments_blocked,
    dense_pose_ppca_score_with_moments_factor_once,
    pose_moment_images,
)
from relax.ppca_refinement.residual_statistics import full_float32

pytestmark=pytest.mark.unit

@pytest.mark.parametrize('q',[2,4,10])
@pytest.mark.parametrize('dtype,tol',[(np.float32,2e-5),(np.float64,2e-11)])
@pytest.mark.parametrize('engine',[dense_pose_ppca_score_with_moments_blocked,dense_pose_ppca_score_with_moments_factor_once])
def test_real_gaussian_posterior_and_moment_images(q,dtype,tol,engine):
    rng=np.random.default_rng(198)
    B,T,R,F=2,2,3,17
    y=rng.normal(size=(B,T,F)).astype(dtype)
    # Common images, means and loading prefixes give a matched q2 control.
    proj=(rng.normal(size=(R,11,F))*.12)[:,:q+1].astype(dtype)
    precision=np.full((B,F),dtype(1/.7),dtype)
    ynorm=np.sum(y*y*precision[:,None],axis=-1)[:,0]
    # Equal y-norm per translation is required by the actual translated-image contract.
    y[:,1]=np.roll(y[:,0],1,axis=-1)
    result=full_float32(engine)(y*precision[:,None],proj,precision,ynorm)
    refscore=np.empty((B,T,R),np.float64)
    refalpha=np.empty((B,T,R,q+1),np.float64)
    refG=np.empty((B,T,R,q+1,q+1),np.float64)
    for b,t,r in np.ndindex(B,T,R):
        mean=proj[r,0].astype(np.float64);w=proj[r,1:].T.astype(np.float64)
        residual=y[b,t].astype(np.float64)-mean
        latentcov=np.linalg.inv(np.eye(q)+w.T@w/.7)
        latentmean=latentcov@w.T@residual/.7
        obs_cov=.7*np.eye(F)+w@w.T
        refscore[b,t,r]=-.5*(residual@np.linalg.solve(obs_cov,residual)+np.linalg.slogdet(obs_cov)[1]-F*np.log(.7))
        refalpha[b,t,r]=np.r_[1,latentmean]
        refG[b,t,r]=np.outer(refalpha[b,t,r],refalpha[b,t,r]);refG[b,t,r,1:,1:]+=latentcov
    np.testing.assert_allclose(np.asarray(result.score)+np.asarray(result.score_offset)[:,None,None],refscore,atol=tol,rtol=tol)
    np.testing.assert_allclose(result.alpha,refalpha,atol=tol,rtol=tol)
    np.testing.assert_allclose(unpack_tri_to_full(result.G_tri,q+1),refG,atol=tol,rtol=tol)
    posterior=np.exp(refscore-refscore.max(axis=(1,2),keepdims=True));posterior/=posterior.sum(axis=(1,2),keepdims=True)
    rhs,lhs=full_float32(pose_moment_images)(jnp.asarray(posterior,dtype),result.alpha,result.G_tri,
                              jnp.asarray(y*precision[:,None]),jnp.asarray(precision),rhs_dtype=dtype,lhs_dtype=dtype)
    expected_rhs=np.einsum('btr,btrp,btf->prf',posterior,refalpha,y.astype(np.float64)*precision[:,None])
    expected_lhs=np.einsum('btr,btrpq,bf->rfpq',posterior,refG,precision.astype(np.float64))
    np.testing.assert_allclose(rhs,expected_rhs,atol=tol,rtol=tol)
    np.testing.assert_allclose(unpack_tri_to_full(lhs.transpose(1,2,0),q+1),expected_lhs,atol=tol,rtol=tol)
    assert np.asarray(result.alpha).dtype==dtype and np.asarray(result.G_tri).dtype==dtype

@pytest.mark.parametrize('q',[4,10])
@pytest.mark.parametrize('dtype,tol',[(np.float32,4e-5),(np.float64,4e-11)])
def test_coupled_and_vdam_update_independent_reference(q,dtype,tol):
    rng=np.random.default_rng(77);p=q+1;F=5
    cdtype=np.complex64 if dtype==np.float32 else np.complex128
    theta=(rng.normal(size=(F,p))+1j*rng.normal(size=(F,p))).astype(cdtype)
    a=rng.normal(size=(2,F,p,p)).astype(dtype)
    matrices=a@a.swapaxes(-1,-2)+np.eye(p,dtype=dtype)*dtype(.5)
    residual=(rng.normal(size=(2,F,p))+1j*rng.normal(size=(2,F,p))).astype(cdtype)
    directions=jnp.stack([coupled_direction(pack_upper_tri(m),r,floor=.2)[0] for m,r in zip(matrices,residual)])
    refdir=np.stack([np.linalg.solve(m.astype(np.float64),r.astype(np.complex128)[...,None])[...,0] for m,r in zip(matrices,residual)])
    np.testing.assert_allclose(directions,refdir,atol=tol,rtol=tol)
    out,moments,_=stochastic_update(jnp.asarray(theta),empty_moments(jnp.asarray(theta)),directions,
              np.ones((2,F),bool),np.zeros(F,np.int32),step=.8,fudge=1.7,image_size=8)
    first=refdir;difference=first[1]-first[0];average=first.mean(axis=0)
    power=lambda x:np.stack([abs(x[:,0])**2,np.sum(abs(x[:,1:])**2,axis=-1)],axis=-1)
    second=.999+.001*power(difference)/(power(average)+1e-12*8**4)
    adapted=np.concatenate([average[:,:1]/(np.sqrt(second[:,:1])+1e-12),average[:,1:]/(np.sqrt(second[:,1:])+1e-12)],axis=-1)
    amplitude=lambda x:np.stack([abs(x[:,0]),np.linalg.norm(x[:,1:],axis=-1)/np.sqrt(q)],axis=-1)
    signal=amplitude(theta.astype(np.complex128)).mean(axis=0);disagreement=amplitude(difference).mean(axis=0)
    rho=2*1.7*signal/disagreement;gates=rho/(1+rho);gate=np.r_[gates[0],np.repeat(gates[1],q)]
    expected=theta+.8*(gate*adapted-(1-gate)*theta)
    np.testing.assert_allclose(out,expected,atol=tol,rtol=tol)
    np.testing.assert_allclose(moments.second,second,atol=tol,rtol=tol)
    assert np.asarray(moments.second).dtype==dtype

@pytest.mark.parametrize('q',[4,10])
def test_momentum_rank_reference_and_dtype(q):
    rng=np.random.default_rng(7);p=q+1;F=4
    a=rng.normal(size=(F,p,p)).astype(np.float32);h=a@a.swapaxes(-1,-2)+np.eye(p,dtype=np.float32)
    theta=(rng.normal(size=(F,p))+1j*rng.normal(size=(F,p))).astype(np.complex64)
    gradient=(rng.normal(size=(F,p))+1j*rng.normal(size=(F,p))).astype(np.complex64)
    old=theta*np.float32(.01)
    out,velocity,_=momentum_step(theta,old,gradient,pack_upper_tri(h),np.ones(F,bool),learning_rate=1.2,floor=.1)
    expected=np.float32(.9)*old+np.float32(.12)*gradient/np.max(np.trace(h,axis1=-2,axis2=-1))
    np.testing.assert_allclose(velocity,expected,atol=2e-7,rtol=2e-6)
    np.testing.assert_allclose(out,theta+expected,atol=2e-7,rtol=2e-6)
    assert out.dtype==velocity.dtype==jnp.complex64

@pytest.mark.parametrize('q',[4,10])
def test_rank_requires_coarse_stream_and_config_remains_opt_in(q):
    assert Config().q==2
    with pytest.raises(ValueError,match='coarse recompute'):
        Config(q=q)
    config=Config(q=q,oversampling=0,stream_coarse_recompute=True)
    assert config.q==q
