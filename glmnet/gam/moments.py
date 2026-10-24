"""
The method of moments for the variances of crossed random effects, in time
linear in the number of observations: Gao & Owen's (2017) U statistics, for
any number of crossed (or nested) random intercepts, of residuals, such as
those of the fit without them (Gao & Owen, 2019; Ghosh, Hastie & Owen,
2022). With the variances fixed at these estimates, `GAM`'s backfitting
gives the generalized least squares fit of the other terms and the random
effects' predictions, each sweep in linear time. The documentation's page
"Crossed random effects at scale" writes the `SmoothingMethod` that does
this, around `moment_equations`.

For the model `r_i = sum_k a^k_{g_k(i)} + e_i` (`r` the residuals, the
`a^k` the effects of categorical `k`, variances `s_k^2`, errors' `s_E^2`), with
`N` observations, `N_kl` of them at level `l` of categorical `k` (`R_k` levels),
and `N_kl,ml'` at level `l` of `k` and `l'` of `m`:

- `U_k`, the sum of squares within the levels of categorical `k`, has
  expectation `sum_{m != k} s_m^2 (N - sum_l sum_l' N_kl,ml'^2 / N_kl) +
  s_E^2 (N - R_k)`;
- `U_E`, `N` times the total sum of squares, has expectation
  `sum_m s_m^2 (N^2 - sum_l N_ml^2) + s_E^2 (N^2 - N)`.

Matching them is a linear system of `K + 1` equations, whose coefficients
are sums of the counts' squares.

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


def moment_equations(r, codes):
    """
    The U statistics of the residuals `r` and the matrix of their
    expectations' coefficients in `(s_1^2, ..., s_K^2, s_E^2)`, for the
    categoricals' level codes `codes` (a list of `K` integer arrays).
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
