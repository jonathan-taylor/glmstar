import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from glmnet.paths import GaussNet, LogNet, FishNet, MultiClassNet, MultiGaussNet, CoxNet
from glmnet.paths.fastnet import FastNetControl
from glmnet.cox import CoxFamily

N_TRAIN, N_TEST, P = 200, 50, 5
FAMILIES = ['gaussian', 'binomial', 'poisson', 'multinomial', 'mgaussian', 'cox']


def get_data(family, seed=0):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    X = rng.standard_normal((n, P))
    eta = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    if family == 'gaussian':
        D = pd.DataFrame({'Y': eta + rng.standard_normal(n)})
    elif family == 'binomial':
        D = pd.DataFrame({'Y': rng.binomial(1, expit(eta)).astype(float)})
    elif family == 'poisson':
        D = pd.DataFrame({'Y': rng.poisson(np.exp(eta / 2)).astype(float)})
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
        D = pd.DataFrame({'stop': np.minimum(T, C),
                          'status': (T <= C).astype(int)})

    control = FastNetControl(thresh=1e-14)
    if family in ['multinomial', 'mgaussian']:
        K = 3 if family == 'multinomial' else 2
        offset_cols = [f'offset{k}' for k in range(K)]
        for c in offset_cols:
            D[c] = 0.3 * rng.standard_normal(n)
        args = dict(offset_id=offset_cols, control=control)
    else:
        D['offset'] = 0.3 * rng.standard_normal(n)
        args = dict(offset_id='offset', control=control)
    if family == 'mgaussian':
        args['response_id'] = ['Y1', 'Y2']
    elif family != 'cox':
        args['response_id'] = 'Y'
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


def fit_and_split(family):
    X, D, args = get_data(family)
    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
    L = get_estimator(family, args).fit(X[tr], D.iloc[tr].reset_index(drop=True))
    offset_te = D.iloc[te][args['offset_id']].values
    return L, X, D, X[te], offset_te


@pytest.mark.parametrize('family', FAMILIES)
def test_offset_adds_to_link(family):
    L, _, _, X_te, offset_te = fit_and_split(family)
    link = L.predict(X_te, prediction_type='link')
    link_off = L.predict(X_te, prediction_type='link', offset=offset_te)
    if link.ndim == 3:
        np.testing.assert_allclose(link_off, link + offset_te[:, None, :])
    else:
        np.testing.assert_allclose(link_off, link + offset_te[:, None])


@pytest.mark.parametrize('family', ['binomial', 'poisson', 'multinomial'])
def test_offset_response_scale(family):
    L, _, _, X_te, offset_te = fit_and_split(family)
    link = L.predict(X_te, prediction_type='link', offset=offset_te)
    response = L.predict(X_te, prediction_type='response', offset=offset_te)
    if family == 'binomial':
        np.testing.assert_allclose(response, expit(link))
        np.testing.assert_allclose(L.predict_proba(X_te, offset=offset_te)[:, :, 1], response)
    elif family == 'poisson':
        np.testing.assert_allclose(response, np.exp(link))
    else:
        prob = np.exp(link - link.max(-1)[:, :, None])
        np.testing.assert_allclose(response, prob / prob.sum(-1)[:, :, None])
        np.testing.assert_allclose(L.predict_proba(X_te, offset=offset_te), response)


def test_offset_vector_for_multiresponse():
    # a vector offset is used for every response
    L, _, _, X_te, offset_te = fit_and_split('mgaussian')
    np.testing.assert_allclose(L.predict(X_te, offset=offset_te[:, 0]),
                               L.predict(X_te, offset=np.column_stack([offset_te[:, 0]] * 2)))


def test_offset_shape_checked():
    L, _, _, X_te, offset_te = fit_and_split('gaussian')
    with pytest.raises(ValueError, match='offset should have shape'):
        L.predict(X_te, offset=offset_te[:-1])
    # a single column is accepted
    np.testing.assert_allclose(L.predict(X_te, offset=offset_te[:, None]),
                               L.predict(X_te, offset=offset_te))


def test_offset_matches_cross_validation():
    # cross_validation_path adds the offset of the held-out rows to predict(X)
    X, D, args = get_data('poisson')
    L = get_estimator('poisson', args).fit(X, D)
    train, test = np.arange(N_TRAIN), np.arange(N_TRAIN, N_TRAIN + N_TEST)
    predictions, _ = L.cross_validation_path(X, D, cv=[(train, test), (test, train)])
    L_tr = get_estimator('poisson', args).fit(X[:N_TRAIN], D.iloc[:N_TRAIN])
    expected = L_tr.predict(X[test], offset=D['offset'].values[test],
                            interpolation_grid=L.lambda_values_)
    np.testing.assert_allclose(predictions[test], expected, rtol=1e-10)


R_FAMILY = {'gaussian': '"gaussian"', 'binomial': '"binomial"', 'poisson': '"poisson"',
            'multinomial': '"multinomial"', 'cox': '"cox"'}


@pytest.mark.parametrize('family', list(R_FAMILY))
def test_offset_matches_R(Rinfo, family):
    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']

    L, X, D, X_te, offset_te = fit_and_split(family)
    tr = slice(0, N_TRAIN)
    with Rinfo['np_cv_rules'].context():
        rpy.r.assign('X', X[tr])
        rpy.r.assign('Xte', X_te)
        rpy.r.assign('lam', L.lambda_values_)
        rpy.r.assign('o', D.iloc[tr][L.offset_id].values)
        rpy.r.assign('ote', offset_te)
        if family == 'cox':
            rpy.r.assign('st', D['stop'].values[tr])
            rpy.r.assign('d', D['status'].values[tr].astype(float))
            rpy.r('Y = Surv(as.vector(st), as.vector(d))')
        elif family == 'multinomial':
            rpy.r.assign('Yv', D['Y'].values[tr].astype(float))
            rpy.r('Y = factor(as.vector(Yv))')
        else:
            rpy.r.assign('Yv', D['Y'].values[tr].astype(float))
            rpy.r('Y = as.vector(Yv)')
        rpy.r('suppressMessages({library(glmnet); library(survival)})')
        rpy.r(f'fit = glmnet(X, Y, family={R_FAMILY[family]}, offset=o, lambda=lam, thresh=1e-14)')
        link_R = np.asarray(rpy.r('predict(fit, Xte, newoffset=ote, type="link")'))
        response_R = np.asarray(rpy.r('predict(fit, Xte, newoffset=ote, type="response")'))

    link = L.predict(X_te, prediction_type='link', offset=offset_te)
    response = L.predict(X_te, prediction_type='response', offset=offset_te)
    if family == 'multinomial':
        # R returns (n, K, nlambda)
        link_R = np.transpose(link_R, (0, 2, 1))
        response_R = np.transpose(response_R, (0, 2, 1))
        # glmnet's multinomial intercepts are only identified up to a constant
        link = link - link.mean(-1)[:, :, None]
        link_R = link_R - link_R.mean(-1)[:, :, None]
    # (for cox, type="response" is the relative risk exp(eta) in both)
    nlam = link_R.shape[1]
    np.testing.assert_allclose(link[:, :nlam], link_R, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(response[:, :nlam], response_R, rtol=1e-4, atol=1e-4)
