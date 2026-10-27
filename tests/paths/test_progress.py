import numpy as np
import pandas as pd
import pytest

from glmnet import GaussNet, LogNet, FishNet, MultiGaussNet, MultiClassNet, CoxNet
from glmnet.cox import CoxFamily
from glmnet.paths.fastnet import FastNetControl

rng = np.random.default_rng(0)
n, p = 100, 5
X = rng.standard_normal((n, p))
eta = X[:, 0] - 0.5 * X[:, 1]
T = rng.exponential(np.exp(-eta))
C = rng.exponential(1.5, size=n)
D = pd.DataFrame({'gaussian': eta + rng.standard_normal(n),
                  'binomial': rng.binomial(1, 1 / (1 + np.exp(-eta))),
                  'poisson': rng.poisson(np.exp(eta / 2)),
                  'multinomial': rng.choice(3, size=n),
                  'Y2': -eta + rng.standard_normal(n),
                  'stop': np.minimum(T, C) + 0.1,
                  'status': (T <= C).astype(int)})

FAMILIES = ['gaussian', 'binomial', 'poisson', 'mgaussian', 'multinomial', 'cox']


def _estimator(family, **kw):
    if family == 'gaussian':
        return GaussNet(**kw)
    if family == 'binomial':
        return LogNet(**kw)
    if family == 'poisson':
        return FishNet(**kw)
    if family == 'mgaussian':
        return MultiGaussNet(**kw)
    if family == 'multinomial':
        return MultiClassNet(**kw)
    return CoxNet(family=CoxFamily(tie_breaking='breslow'), **kw)


def _Y(family):
    # y, its columns named by role
    columns = {'gaussian': ['gaussian'], 'binomial': ['binomial'], 'poisson': ['poisson'],
               'mgaussian': ['gaussian', 'Y2'], 'multinomial': ['multinomial'],
               'cox': ['stop', 'status']}[family]
    return D[columns]


@pytest.mark.parametrize('family', FAMILIES)
def test_silent_by_default(family, capfd):
    fit = _estimator(family).fit(X, _Y(family))
    fit.cross_validation_path(X, _Y(family), cv=3)
    out, err = capfd.readouterr()
    assert out == ''
    assert err == ''


@pytest.mark.parametrize('family', FAMILIES)
def test_itrace_shows_progress(family, capfd):
    fit = _estimator(family, control=FastNetControl(itrace=1)).fit(X, _Y(family))
    out, err = capfd.readouterr()
    assert out == ''
    # the bar advances to the number of lambda values fitted
    assert f'{len(fit.lambda_values_)}/{fit.nlambda}' in err
    # and cross-validation folds inherit the setting
    fit.cross_validation_path(X, _Y(family), cv=3)
    out, err = capfd.readouterr()
    assert err.count(f'/{fit.nlambda}') >= 3
