"""
Tests for lambda selection in `glmnet.scorer._tune`.

The 1SE rule follows R's `cv.glmnet`: the least complex model whose
score is within one standard error of the best score.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from glmnet.scorer import _tune

# lambda decreasing along the path, i.e. complexity increasing
LAMBDA = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
MSE = np.array([5.0, 3.0, 2.0, 1.9, 2.1])
SD = np.full(5, 0.2)


def _scores(mean, sd, name):
    return pd.DataFrame({name: mean, f'SD({name})': sd})


def _R_lambda_1se(lam, cvm, cvsd, maximize=False):
    # R's getOptcv.glmnet
    if maximize:
        cvm = -cvm
    idmin = np.argmin(cvm)
    semin = (cvm + cvsd)[idmin]
    return lam[idmin], lam[cvm <= semin].max()


def test_1se_minimize():
    scorer = SimpleNamespace(name='MSE', maximize=False)
    best, se1 = _tune(pd.Series(LAMBDA),
                      [scorer],
                      _scores(MSE, SD, 'MSE'),
                      complexity_order='increasing')
    assert best['MSE'] == 0.125
    assert se1['MSE'] == 0.25
    assert (best['MSE'], se1['MSE']) == _R_lambda_1se(LAMBDA, MSE, SD)


def test_1se_maximize():
    auc = 1 - MSE / 10
    sd = SD / 10
    scorer = SimpleNamespace(name='AUC', maximize=True)
    best, se1 = _tune(pd.Series(LAMBDA),
                      [scorer],
                      _scores(auc, sd, 'AUC'),
                      complexity_order='increasing')
    assert best['AUC'] == 0.125
    assert se1['AUC'] == 0.25
    assert (best['AUC'], se1['AUC']) == _R_lambda_1se(LAMBDA, auc, sd, maximize=True)


def test_1se_first_entry_eligible():
    mse = np.array([2.0, 1.95, 1.9, 1.92, 2.5])
    scorer = SimpleNamespace(name='MSE', maximize=False)
    best, se1 = _tune(pd.Series(LAMBDA),
                      [scorer],
                      _scores(mse, SD, 'MSE'),
                      complexity_order='increasing')
    assert best['MSE'] == 0.25
    assert se1['MSE'] == 1.0
    assert (best['MSE'], se1['MSE']) == _R_lambda_1se(LAMBDA, mse, SD)


@pytest.mark.parametrize('maximize', [False, True])
def test_decreasing_matches_increasing(maximize):
    # SD varies along the path so that misaligned mean / SD would be caught
    sd = np.array([0.05, 0.4, 0.05, 0.3, 0.05])
    mean = 1 - MSE / 10 if maximize else MSE
    scorer = SimpleNamespace(name='S', maximize=maximize)

    best_inc, se1_inc = _tune(pd.Series(LAMBDA),
                              [scorer],
                              _scores(mean, sd, 'S'),
                              complexity_order='increasing')
    best_dec, se1_dec = _tune(pd.Series(LAMBDA[::-1]),
                              [scorer],
                              _scores(mean[::-1], sd[::-1], 'S'),
                              complexity_order='decreasing')

    assert best_inc['S'] == best_dec['S']
    assert se1_inc['S'] == se1_dec['S']
    assert (best_inc['S'], se1_inc['S']) == _R_lambda_1se(LAMBDA, mean, sd, maximize=maximize)


def test_no_complexity_order():
    scorer = SimpleNamespace(name='MSE', maximize=False)
    best, se1 = _tune(pd.Series(LAMBDA),
                      [scorer],
                      _scores(MSE, SD, 'MSE'),
                      complexity_order=None)
    assert best['MSE'] == 0.125
    assert np.isnan(se1['MSE'])
