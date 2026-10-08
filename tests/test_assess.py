import numpy as np
import pandas as pd
import pytest

from glmnet.paths import GaussNet, LogNet, FishNet, MultiClassNet, MultiGaussNet, CoxNet
from glmnet.cox import CoxFamily
from glmnet.assess import assess, confusion, roc, _link_predictions

rng = np.random.default_rng(0)

N_TRAIN, N_TEST, P = 200, 150, 6

def get_data(family, use_weights, use_offset):
    n = N_TRAIN + N_TEST
    X = rng.standard_normal((n, P))
    eta = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    if family == 'gaussian':
        D = pd.DataFrame({'Y': eta + rng.standard_normal(n)})
    elif family == 'binomial':
        D = pd.DataFrame({'Y': rng.binomial(1, 1 / (1 + np.exp(-eta)))})
    elif family == 'poisson':
        D = pd.DataFrame({'Y': rng.poisson(np.exp(eta / 2))})
    elif family == 'multinomial':
        logits = np.column_stack([eta, -eta, np.zeros(n)])
        prob = np.exp(logits) / np.exp(logits).sum(1)[:, None]
        D = pd.DataFrame({'Y': [rng.choice(3, p=p_) for p_ in prob]})
    elif family == 'mgaussian':
        D = pd.DataFrame({'Y1': eta + rng.standard_normal(n),
                          'Y2': -eta + rng.standard_normal(n)})
    else:
        T = rng.exponential(np.exp(-eta))
        C = rng.exponential(1.5, size=n)
        D = pd.DataFrame({'stop': np.round(np.minimum(T, C), 1) + 0.2,
                          'status': (T <= C).astype(int)})
    args = {}
    if family not in ['cox', 'mgaussian']:
        args['response_id'] = 'Y'
    if family == 'mgaussian':
        args['response_id'] = ['Y1', 'Y2']
    if use_weights:
        D['weight'] = rng.uniform(0.5, 2, n)
        args['weight_id'] = 'weight'
    if use_offset:
        if family in ['multinomial', 'mgaussian']:
            K = 3 if family == 'multinomial' else 2
            offset_cols = [f'offset{k}' for k in range(K)]
            for c in offset_cols:
                D[c] = 0.3 * rng.standard_normal(n)
            args['offset_id'] = offset_cols
        else:
            D['offset'] = 0.3 * rng.standard_normal(n)
            args['offset_id'] = 'offset'
    return X, D, args

def get_estimator(family, args):
    if family == 'gaussian':
        return GaussNet(**args)
    if family == 'binomial':
        return LogNet(**args)
    if family == 'poisson':
        return FishNet(**args)
    if family == 'multinomial':
        return MultiClassNet(**args)
    if family == 'mgaussian':
        return MultiGaussNet(**args)
    return CoxNet(family=CoxFamily(tie_breaking='breslow', event_id='stop',
                                   status_id='status'), **args)

def fit_and_split(family, use_weights, use_offset):
    X, D, args = get_data(family, use_weights, use_offset)
    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
    L = get_estimator(family, args).fit(X[tr], D.iloc[tr].reset_index(drop=True))
    return L, X[te], D.iloc[te].reset_index(drop=True)

def assign_R(Rinfo, family, link, D_te):
    """Put the link predictions, response and weights of the test set into R."""
    rpy = Rinfo['rpy']
    with Rinfo['np_cv_rules'].context():
        rpy.r('rm(list=intersect(c("w"), ls()))')
        if link.ndim == 3:
            # R expects (n, K, nlambda)
            link = np.ascontiguousarray(np.transpose(link, (0, 2, 1)))
        rpy.r.assign('P', link)
        if family == 'cox':
            rpy.r.assign('st', D_te['stop'].values)
            rpy.r.assign('d', D_te['status'].values.astype(float))
            rpy.r('Y = Surv(as.vector(st), as.vector(d))')
        elif family == 'mgaussian':
            rpy.r.assign('Yv', D_te[['Y1', 'Y2']].values)
            rpy.r('Y = as.matrix(Yv)')
        elif family == 'multinomial':
            rpy.r.assign('Yv', D_te['Y'].values.astype(float))
            rpy.r('Y = factor(as.vector(Yv))')
        else:
            rpy.r.assign('Yv', D_te['Y'].values.astype(float))
            rpy.r('Y = as.vector(Yv)')
        if 'weight' in D_te.columns:
            rpy.r.assign('w', D_te['weight'].values)
        rpy.r('suppressMessages({library(glmnet); library(survival)})')

@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'multinomial',
                                    'mgaussian', 'cox'])
def test_assess(Rinfo, family, use_weights, use_offset):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']

    L, X_te, D_te = fit_and_split(family, use_weights, use_offset)
    A = assess(L, X_te, D_te)
    link, _, _ = _link_predictions(L, X_te, D_te)

    assign_R(Rinfo, family, link, D_te)
    weights = ', weights=as.vector(w)' if use_weights else ''
    with Rinfo['np_cv_rules'].context():
        rpy.r(f'A = assess.glmnet(P, newy=Y, family="{family}"{weights})')
        R = {k: np.asarray(rpy.r(f'as.numeric(A${k})')) for k in rpy.r('names(A)')}

    assert list(A.columns) == list(R.keys())
    for k in R:
        assert np.allclose(A[k], R[k], rtol=1e-8, atol=1e-10), k

@pytest.mark.parametrize('family', ['binomial', 'multinomial'])
def test_confusion(Rinfo, family, use_offset):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']

    L, X_te, D_te = fit_and_split(family, False, use_offset)
    tables = confusion(L, X_te, D_te)
    link, _, _ = _link_predictions(L, X_te, D_te)
    assert len(tables) == L.lambda_values_.shape[0]

    assign_R(Rinfo, family, link, D_te)
    with Rinfo['np_cv_rules'].context():
        if family == 'binomial':
            # R labels the predicted classes "1" and "2"
            rpy.r('Y = factor(Y, labels=c("1", "2"))')
        rpy.r(f'''
CT = confusion.glmnet(P, newy=Y, family="{family}")
nlev = nlevels(Y)
# the full nlev x nlev table for each lambda, as R drops unpredicted classes
full = lapply(CT, function(t) {{
  m = matrix(0, nlev, nlev, dimnames=list(levels(Y), levels(Y)))
  # multinomial predictions are class numbers, binomial ones class labels
  rows = if ("{family}" == "binomial") rownames(t) else levels(Y)[as.integer(rownames(t))]
  m[rows, colnames(t)] = unclass(t)
  m}})
''')
        nlev = int(np.asarray(rpy.r('nlev'))[0])
        R_tables = [np.asarray(rpy.r(f'full[[{j + 1}]]')).reshape((nlev, nlev), order='F')
                    for j in range(len(tables))]

    for T, R_T in zip(tables, R_tables):
        assert list(T.index.names) == ['Predicted'] and list(T.columns.names) == ['True']
        assert np.array_equal(T.values, R_T)

def test_roc(Rinfo, use_offset):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']

    L, X_te, D_te = fit_and_split('binomial', False, use_offset)
    curves = roc(L, X_te, D_te)
    link, _, _ = _link_predictions(L, X_te, D_te)
    assert len(curves) == L.lambda_values_.shape[0]

    assign_R(Rinfo, 'binomial', link, D_te)
    with Rinfo['np_cv_rules'].context():
        rpy.r('RC = roc.glmnet(P, newy=Y)')
        for j in [0, 5, len(curves) - 1]:
            R_FPR = np.asarray(rpy.r(f'RC[[{j + 1}]]$FPR'))
            R_TPR = np.asarray(rpy.r(f'RC[[{j + 1}]]$TPR'))
            assert np.allclose(curves[j]['FPR'], R_FPR)
            assert np.allclose(curves[j]['TPR'], R_TPR)

def test_labels_and_errors():

    # string labels, and test data missing one class
    X, D, args = get_data('binomial', False, False)
    D['Y'] = np.where(D['Y'] == 1, 'yes', 'no')
    L = LogNet(**args).fit(X[:N_TRAIN], D.iloc[:N_TRAIN])
    D_te = D.iloc[N_TRAIN:].reset_index(drop=True)
    T = confusion(L, X[N_TRAIN:], D_te)[10]
    assert list(T.index) == ['no', 'yes'] and list(T.columns) == ['no', 'yes']
    assert T.values.sum() == N_TEST
    only_no = D_te['Y'] == 'no'
    T = confusion(L, X[N_TRAIN:][only_no], D_te[only_no].reset_index(drop=True))[10]
    assert T.shape == (2, 2) and np.all(T['yes'] == 0)

    curve = roc(L, X[N_TRAIN:], D_te)[10]
    assert np.isclose(curve['FPR'].iloc[-1], 1) and np.isclose(curve['TPR'].iloc[-1], 1)
    assert np.all(np.diff(curve['FPR']) >= 0) and np.all(np.diff(curve['TPR']) >= 0)

    G = GaussNet(response_id='Y').fit(X[:N_TRAIN], pd.DataFrame({'Y': X[:N_TRAIN, 0]}))
    with pytest.raises(ValueError, match='binomial or multinomial'):
        confusion(G, X[:10], pd.DataFrame({'Y': X[:10, 0]}))
    with pytest.raises(ValueError, match='binomial'):
        roc(G, X[:10], pd.DataFrame({'Y': X[:10, 0]}))
