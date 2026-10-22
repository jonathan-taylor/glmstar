"""
The variances of crossed random effects by the method of moments, in time
linear in the number of observations: Gao & Owen's (2017) U statistics, for
any number of crossed (or nested) random intercepts, on the residuals of
the fit without them (Gao & Owen, 2019; Ghosh, Hastie & Owen, 2022). With
the variances fixed at these estimates, `GAM`'s backfitting gives the
generalized least squares fit of the other terms and the random effects'
predictions, each sweep in linear time.

For the model `r_i = sum_k a^k_{g_k(i)} + e_i` (`r` the residuals, the
`a^k` the effects of factor `k`, variances `s_k^2`, errors' `s_E^2`), with
`N` observations, `N_kl` of them at level `l` of factor `k` (`R_k` levels),
and `N_kl,ml'` at level `l` of `k` and `l'` of `m`:

- `U_k`, the sum of squares within the levels of factor `k`, has
  expectation `sum_{m != k} s_m^2 (N - sum_l sum_l' N_kl,ml'^2 / N_kl) +
  s_E^2 (N - R_k)`;
- `U_E`, `N` times the total sum of squares, has expectation
  `sum_m s_m^2 (N^2 - sum_l N_ml^2) + s_E^2 (N^2 - N)`.

Matching them is a linear system of `K + 1` equations, whose coefficients
are sums of the counts' squares. A negative estimate is set to 0 (the
effects then fitted as 0: a smoothing parameter at infinity, in effect).

References
----------
- Gao, K. and Owen, A. B. (2017). Efficient moment calculations for
  variance components in large unbalanced crossed random effects models.
  *Electronic Journal of Statistics* 11(1), 1235–1296.
- Gao, K. and Owen, A. B. (2019). Estimation and inference for very large
  linear mixed effects models. *Statistica Sinica* 30, 1741–1771.
- Ghosh, S., Hastie, T. and Owen, A. B. (2022). Backfitting for large scale
  crossed random effects regressions. *The Annals of Statistics* 50(1).
  arXiv:2007.10612.
"""
import numpy as np
from statsmodels.genmod.families import family as sm_family, links as sm_links

from .selection import SmoothingMethod, estimated_terms
from .reml import SmoothingSelection
from .terms import RandomEffectTerm


def moment_equations(r, codes):
    """
    The U statistics of the residuals `r` and the matrix of their
    expectations' coefficients in `(s_1^2, ..., s_K^2, s_E^2)`, for the
    factors' level codes `codes` (a list of `K` integer arrays).
    """
    r = np.asarray(r, float)
    N = r.shape[0]
    K = len(codes)
    counts = [np.bincount(c) for c in codes]
    U, A = np.zeros(K + 1), np.zeros((K + 1, K + 1))
    for k, (c, n_k) in enumerate(zip(codes, counts)):
        means = np.bincount(c, weights=r) / np.maximum(n_k, 1)
        U[k] = np.sum((r - means[c])**2)
        for m in range(K):
            if m == k:
                continue
            # the counts of the pairs of levels (of k and m) that occur
            pairs = c.astype(np.int64) * (codes[m].max() + 1) + codes[m]
            _, inverse, n_pair = np.unique(pairs, return_inverse=True, return_counts=True)
            level_k = c[np.unique(inverse, return_index=True)[1]]
            A[k, m] = N - np.sum(n_pair.astype(float)**2 / n_k[level_k])
        A[k, K] = N - np.count_nonzero(n_k)
    U[K] = N * np.sum((r - r.mean())**2)
    for m, n_m in enumerate(counts):
        A[K, m] = float(N)**2 - np.sum(n_m.astype(float)**2)
    A[K, K] = float(N)**2 - N
    return U, A


class CrossedMoments(SmoothingMethod):
    """
    The method of moments for crossed (or nested) random intercepts
    (`re(g)` terms, without `by`) in a Gaussian (identity link) GAM with
    equal prior weights: their variances and the dispersion from Gao &
    Owen's U statistics of the residuals of the fit without them, in time
    linear in the number of observations; `GAM` then fits with them fixed
    (by backfitting). See `glmnet.gam.moments`.

    The other terms' smoothing is their own (`lam` or `df`; `s()` must be
    given one).
    """
    name = 'moments'
    joint = False
    covariance_edf = False
    experimental = True

    def select(self, gam, dispersion=None):
        family = gam._family
        base = getattr(family, 'base', None)
        if not (isinstance(base, sm_family.Gaussian) and isinstance(base.link, sm_links.Identity)):
            raise ValueError('the method of moments is for the Gaussian family with the identity link')
        weight = np.asarray(gam._weight, float)
        keep = weight > 0
        if not np.allclose(weight[keep], weight[keep][0]):
            raise ValueError('the method of moments needs equal prior weights')
        estimated = estimated_terms(gam)
        terms = gam.terms_
        for j in estimated:
            term = terms[j]
            if not isinstance(term, RandomEffectTerm) or term.by is not None:
                raise ValueError(f'{term.name}: the method of moments estimates the variances of '
                                 'random intercepts (re(g), without by): give its lam')
        if not estimated:
            raise ValueError('method="moments": no random effect variance to estimate')
        W = weight.sum()
        normed = gam._normed_weight

        # the fit without the random effects (theirs at 0: a very large
        # smoothing parameter), and its residuals
        for j in estimated:
            terms[j].lam = 1e12
        for term in terms:
            term.set_weights(normed)
        state = gam.inner_fit()
        r = (np.asarray(gam._response, float) - np.asarray(state.mu, float))[keep]
        codes = [terms[j].codes_[keep] for j in estimated]
        U, A = moment_equations(r, codes)
        variances = np.linalg.solve(A, U)
        s2 = np.maximum(variances[:-1], 0.)
        phi = float(variances[-1]) if dispersion is None else float(dispersion)
        if not phi > 0:
            raise ValueError('the method of moments gave a non-positive error variance')

        # the smoothing parameters, on the prior weights' scale: phi / s_k^2
        # (at a variance of 0, very large)
        lam_raw = np.where(s2 > 0, phi / np.where(s2 > 0, s2, 1.), 1e12 * W)
        for j, l in zip(estimated, lam_raw):
            terms[j].lam = float(l) / W
        return SmoothingSelection(method=self.name,
                                  terms=[terms[j].name for j in estimated],
                                  lam=lam_raw / W,
                                  lam_raw=lam_raw,
                                  theta=np.log(lam_raw),
                                  dispersion=phi,
                                  score=np.nan,
                                  gradient=np.zeros(0),
                                  converged=bool(gam.converged_),
                                  n_evaluations=1,
                                  message=('negative variance estimates set to 0: '
                                           f'{[terms[j].name for j, v in zip(estimated, variances) if v < 0]}'
                                           if np.any(variances[:-1] < 0) else ''))
