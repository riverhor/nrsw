"""Pure prespecified non-spectral evidence estimators; no files or role discovery.

They supply BOTH allocation and peak strength. Receiving heads must not inherit
the NRSW rank-one auxiliary strength unit. See ESTIMATOR_DESIGN.md and PLAN.md.
"""
from dataclasses import dataclass
import math
import numpy as np


def require(ok, text):
    if not ok:
        raise ValueError(text)


def normal_means(normal):
    normal = np.asarray(normal, dtype=np.float64)
    require(normal.ndim == 3 and len(normal) >= 2 and normal.shape[1] > 1 and normal.shape[2] > 0
            and np.isfinite(normal).all(), 'finite genuine [N,m,d], N>=2,m>=2')
    return np.asarray([[math.fsum(float(x) for x in normal[:, j, a]) / len(normal)
                        for a in range(normal.shape[2])] for j in range(normal.shape[1])])


def center(bags, means):
    bags = np.asarray(bags, dtype=np.float64)
    require(bags.ndim == 3 and means.shape == bags.shape[1:]
            and np.isfinite(bags).all() and np.isfinite(means).all(), 'center shape/finiteness')
    residual = bags - means
    common = np.asarray([[math.fsum(float(v) for v in col)/len(means)
                          for col in row.T] for row in residual], dtype=np.float64).reshape(len(bags),means.shape[1])
    return residual - common[:, None, :]


@dataclass
class DiagonalEvidence:
    kind: str
    means: np.ndarray
    weights: np.ndarray
    denominator: np.ndarray
    active: bool
    stats: dict

    def evaluate(self, bags, centered=None):
        bags = np.asarray(bags)
        n, m, d = bags.shape
        require(self.means.shape == (m,d), 'teacher field schema')
        if self.active:
            r = center(bags, self.means) if centered is None else centered
            require(r.shape == bags.shape and r.dtype == np.float64 and np.isfinite(r).all(), 'center cache')
            require(self.weights.shape == (d,) and self.denominator.shape == (m,)
                    and np.isfinite(self.weights).all() and (self.weights>=0).all()
                    and np.isfinite(self.denominator).all() and (self.denominator>0).all(), 'positive field metric')
            energy = np.sum(r*r*self.weights[None,None,:], axis=2)/self.denominator[None,:]
        else:
            energy = np.zeros((n,m), dtype=np.float64)
        require(np.isfinite(energy).all() and (energy>=0).all(), 'finite nonnegative energy')
        total = np.asarray([math.fsum(float(x) for x in row) for row in energy])
        alpha = np.full((n,m),1/m,dtype=np.float64)
        positive = total>0
        alpha[positive] = energy[positive]/total[positive,None]
        return dict(energy=energy, alpha=alpha.astype(np.float32), T=energy.max(axis=1))


def finish(kind, means, rn, weights, kappa, tau, stats):
    active = bool(tau>0 and np.any(weights>0))
    if active:
        denominator = kappa*np.sum(rn*rn*weights[None,None,:],axis=(0,2))/(len(rn)-1)+tau*float(weights.sum())
        require(np.isfinite(denominator).all() and (denominator>0).all(), 'positive own studentizers')
    else:
        weights = np.zeros(means.shape[1],dtype=np.float64)
        denominator = np.empty(0,dtype=np.float64)
    return DiagonalEvidence(kind, means, weights, denominator, active,
                            dict(stats,kappa=kappa,tau=tau,active=active,positive_coordinates=int((weights>0).sum())))


def fit_diagonal(normal, forged, candidates):
    """Same-tag moments; W_aa=max(CF_aa-kappa CN_aa,0)/(kappa CN_aa+tau)^2."""
    normal, forged = np.asarray(normal,np.float64), np.asarray(forged,np.float64)
    means = normal_means(normal)
    require(forged.ndim == 3 and forged.shape[1:] == means.shape and len(forged)>0
            and np.isfinite(forged).all() and len(candidates)==len(forged), 'forged input/tags')
    m,d = means.shape
    pi = np.zeros((len(forged),m),np.float64)
    for i, tags in enumerate(candidates):
        require(len(tags) in (1,2) and len(set(tags))==len(tags)
                and all(type(j) is int and 0<=j<m for j in tags), 'one/two candidate fields')
        pi[i,list(tags)] = 1/len(tags)
    p = pi.mean(0)
    rn,rf = center(normal,means),center(forged,means)
    nrows = (rn*np.sqrt(m*p)[None,:,None]).reshape(-1,d)
    frows = (rf*np.sqrt(m*pi)[:,:,None]).reshape(-1,d)
    cn = np.sum(nrows*nrows,axis=0)/(m*(len(normal)-1))
    cf = np.sum(frows*frows,axis=0)/(m*len(forged))
    kappa = 1+1/len(normal)
    tau = math.fsum(float(v) for v in cn)/d
    weights = np.maximum(cf-kappa*cn,0)/(kappa*cn+tau)**2 if tau>0 else np.zeros(d)
    return finish('same_tag_diagonal_contrast',means,rn,weights,kappa,tau,
                  dict(n_normal=len(normal),n_forged=len(forged),field_marginal=p.tolist(),normal_diagonal=cn.tolist(),forged_diagonal=cf.tolist()))


def fit_normal_energy(normal):
    """Genuine-only diagonal whitening; no forged inputs, tags or PSD factors."""
    normal = np.asarray(normal,np.float64)
    means = normal_means(normal)
    rn = center(normal,means)
    m,d = means.shape
    cn = np.sum(rn*rn,axis=(0,1))/(m*(len(normal)-1))
    kappa = 1+1/len(normal)
    tau = math.fsum(float(v) for v in cn)/d
    weights = 1/(kappa*cn+tau) if tau>0 else np.zeros(d)
    return finish('genuine_diagonal_energy',means,rn,weights,kappa,tau,
                  dict(n_normal=len(normal),n_forged=0,field_marginal=None,normal_diagonal=cn.tolist()))
