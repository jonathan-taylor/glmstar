"""
`_extract_fits` builds the coefficient path in place in the solver's `ca`
buffer. Check it against the previous allocate-and-scatter version, and
that the memory saving is real.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse

from glmnet import GaussNet, LogNet, FishNet, MultiGaussNet

rng = np.random.default_rng(0)


def make_data(n, p, sparse=False, seed=0):
    rng = np.random.default_rng(seed)
    if sparse:
        X = scipy.sparse.random(n, p, density=min(1., 5 / p), format='csc',
                                random_state=rng)
        Xd = X.toarray()
    else:
        X = Xd = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:5] = rng.standard_normal(5) * 3
    y = Xd @ beta + rng.standard_normal(n)
    return X, y


def fit_with_raw(cls, X, y, **kwargs):
    """Fit, and capture the solver's raw outputs as `_extract_fits` saw them."""
    raw = {}

    class Snapshot(cls):
        def _extract_fits(self, X_shape, response_shape):
            raw.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                        for k, v in self._fit.items()})
            raw['nx'] = self._args['nx']
            return super()._extract_fits(X_shape, response_shape)

    return Snapshot(**kwargs).fit(X, y), raw


def previous_extract(raw, n_features):
    """The previous extraction (single response)."""
    nfits = raw['lmu']
    ninmax = int(raw['nin'][:nfits].max())
    ca = raw['ca']
    if ca.ndim == 1:
        unsort = ca[:(raw['nx'] * nfits)].reshape(nfits, raw['nx'])
    else:
        unsort = ca[:, :nfits].T
    df = (np.fabs(unsort) > 0).sum(1)
    active_seq = raw['ia'].reshape(-1)[:ninmax] - 1
    coefs = np.zeros((nfits, n_features))
    coefs[:, active_seq] = unsort[:, :len(active_seq)]
    return coefs, df


def check_against_previous(L, raw, n_features):
    coefs, df = previous_extract(raw, n_features)
    assert L.coefs_.shape == coefs.shape
    np.testing.assert_array_equal(L.coefs_, coefs)
    df[0] = 0
    np.testing.assert_array_equal(L.summary_['Degrees of Freedom'], df)
    assert 'ca' not in L._fit and 'ca' not in L._args
    assert L.coefs_.flags['C_CONTIGUOUS']


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('cls', [GaussNet, LogNet, FishNet])
def test_matches_previous_extraction(cls, sparse):
    n, p = 300, 40
    X, y = make_data(n, p, sparse=sparse)
    if cls is LogNet:
        y = (y > 0).astype(float)
    elif cls is FishNet:
        y = rng.poisson(np.exp(np.clip(y / 5, -3, 3))).astype(float)
    L, raw = fit_with_raw(cls, X, y, nlambda=30)
    assert raw['nx'] == p
    check_against_previous(L, raw, p)


def test_early_stop_uses_only_fitted_lambdas():
    X, y = make_data(300, 20) # strong signal, stops on devmax before nlambda
    L, raw = fit_with_raw(GaussNet, X, y, nlambda=100, lambda_min_ratio=1e-6)
    assert L.coefs_.shape[0] < 100
    assert L.coefs_.shape[0] == raw['lmu'] == len(L.lambda_values_)
    check_against_previous(L, raw, 20)


@pytest.mark.parametrize('sparse', [False, True])
def test_df_max_shrinks_buffer(sparse):
    # nx = min(2 * df_max + 20, p) < p: the buffer can't hold the dense result
    n, p, df_max = 300, 100, 5
    X, y = make_data(n, p, sparse=sparse)
    L, raw = fit_with_raw(GaussNet, X, y, nlambda=30, df_max=df_max)
    assert raw['nx'] == 2 * df_max + 20 < p
    check_against_previous(L, raw, p)
    full = GaussNet(nlambda=30).fit(X, y)
    k = L.coefs_.shape[0]
    np.testing.assert_allclose(L.coefs_, full.coefs_[:k], rtol=1e-6, atol=1e-10)


def test_no_active_features():
    X, y = make_data(100, 10)
    L = GaussNet(lambda_values=[1e6, 1e5]).fit(X, y)
    assert L.coefs_.shape == (2, 10)
    assert not np.any(L.coefs_)
    assert 'ca' not in L._fit


def test_multi_response_matches_previous():
    n, p, q = 200, 30, 3
    X, _ = make_data(n, p)
    Y = X[:, :4] @ np.random.default_rng(1).standard_normal((4, q)) + 0.1 * rng.standard_normal((n, q))
    L, raw = fit_with_raw(MultiGaussNet, X, Y, nlambda=20)
    nfits = raw['lmu']
    ninmax = int(raw['nin'][:nfits].max())
    unsort = raw['ca'][:(q * p * nfits)].reshape(nfits, q, p).transpose(0, 2, 1)
    active_seq = raw['ia'].reshape(-1)[:ninmax] - 1
    coefs = np.zeros((nfits, p, q))
    coefs[:, active_seq] = unsort[:, :len(active_seq)]
    np.testing.assert_array_equal(L.coefs_, coefs)
    assert 'ca' not in L._fit


PEAK_SCRIPT = textwrap.dedent('''
    import resource, sys
    import numpy as np, scipy.sparse
    from glmnet import GaussNet

    def maxrss():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == 'darwin' else r * 1024

    n, p, nlam = 2000, 200_000, 100
    rng = np.random.default_rng(0)
    X = scipy.sparse.random(n, p, density=3 / p, format='csc', random_state=rng)
    beta = np.zeros(p); beta[rng.choice(p, 50, replace=False)] = rng.standard_normal(50) * 5
    y = X @ beta + rng.standard_normal(n)
    before = maxrss()
    L = GaussNet(nlambda=nlam, lambda_min_ratio=1e-3).fit(X, y)
    after = maxrss()
    print((after - before) / (8 * p * nlam))
''')


@pytest.mark.skipif(sys.platform == 'win32', reason='needs resource.getrusage')
def test_peak_memory_is_about_one_copy(tmp_path):
    # peak RSS over the fit / size of the (nlambda, n_features) path:
    # ~3.2 before, ~1.2 now
    # run outside the source tree: `python -c` puts the working directory
    # first on sys.path, which from the repository root would import the
    # uncompiled glmnet/ instead of the installed package
    out = subprocess.run([sys.executable, '-c', PEAK_SCRIPT], check=True,
                         capture_output=True, text=True,
                         env={'TQDM_DISABLE': '1', 'PATH': ''},
                         cwd=tmp_path)
    ratio = float(out.stdout.strip().splitlines()[-1])
    assert ratio < 2.0, f'peak RSS was {ratio:.2f}x the coefficient path'
