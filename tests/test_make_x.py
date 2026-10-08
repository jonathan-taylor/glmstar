import warnings

import numpy as np
import pandas as pd
import pytest

from glmnet import GaussNet
from glmnet.data import make_x

# the same data in R and in pandas
R_DATA = '''
tr <- data.frame(a = c(1, NA, 3, 4),
                 ch = c("b", "a", NA, "b"),
                 f = factor(c("x", "y", "x", NA), levels = c("y", "x")),
                 o = factor(c("lo", "hi", "lo", "hi"), levels = c("lo", "hi"), ordered = TRUE),
                 l = c(TRUE, FALSE, NA, TRUE),
                 i = c(1L, 2L, 3L, 4L),
                 s = factor(c("only", "only", "only", "only")),
                 stringsAsFactors = FALSE)
te <- data.frame(a = c(NA, 2), ch = c("c", "a"), f = factor(c("z", "x")),
                 o = factor(c("hi", "lo"), levels = c("lo", "hi"), ordered = TRUE),
                 l = c(FALSE, NA), i = c(5L, NA), s = factor(c("only", NA)),
                 stringsAsFactors = FALSE)
'''

# R's sparse makeX fails on logical and single-level factor columns
SPARSE_DROP = ['l', 's']


def _data():
    tr = pd.DataFrame({'a': [1, np.nan, 3, 4],
                       'ch': ['b', 'a', None, 'b'],
                       'f': pd.Categorical(['x', 'y', 'x', None], categories=['y', 'x']),
                       'o': pd.Categorical(['lo', 'hi', 'lo', 'hi'], categories=['lo', 'hi'], ordered=True),
                       'l': pd.array([True, False, None, True], dtype='boolean'),
                       'i': pd.array([1, 2, 3, 4], dtype='Int64'),
                       's': pd.Categorical(['only'] * 4)})
    te = pd.DataFrame({'a': [np.nan, 2],
                       'ch': ['c', 'a'],
                       'f': pd.Categorical(['z', 'x']),
                       'o': pd.Categorical(['hi', 'lo'], categories=['lo', 'hi'], ordered=True),
                       'l': pd.array([False, None], dtype='boolean'),
                       'i': pd.array([5, None], dtype='Int64'),
                       's': pd.Categorical(['only', None])})
    return tr, te


def _R_matrix(rpy, expr):
    names = list(rpy.r(f'colnames({expr})'))
    with (rpy.default_converter + rpy.numpy2ri.converter).context():
        values = np.asarray(rpy.r(f'as.matrix({expr})'))
    return names, values


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('na_impute', [False, True])
@pytest.mark.parametrize('with_test', [False, True])
def test_make_x_matches_R(Rinfo, sparse, na_impute, with_test):
    rpy = Rinfo['rpy']
    rpy.numpy2ri = Rinfo['numpy2ri']
    rpy.default_converter = Rinfo['default_converter']
    rpy.r(R_DATA)
    tr, te = _data()
    if sparse:
        rpy.r(f'tr <- tr[, !(names(tr) %in% c({", ".join(repr(c) for c in SPARSE_DROP)}))]')
        rpy.r(f'te <- te[, !(names(te) %in% c({", ".join(repr(c) for c in SPARSE_DROP)}))]')
        tr, te = tr.drop(columns=SPARSE_DROP), te.drop(columns=SPARSE_DROP)
    R_na = 'TRUE' if na_impute else 'FALSE'
    R_sparse = 'TRUE' if sparse else 'FALSE'
    R_test = ', te' if with_test else ''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rpy.r(f'r <- suppressWarnings(makeX(tr{R_test}, na.impute = {R_na}, sparse = {R_sparse}))')
        result = make_x(tr, te if with_test else None, na_impute=na_impute, sparse=sparse)
    X, X_test = result if with_test else (result, None)

    R_x = 'r$x' if with_test else 'r'
    names, values = _R_matrix(rpy, R_x)
    assert list(X.columns) == names
    if sparse:
        assert all(isinstance(d, pd.SparseDtype) for d in X.dtypes)
    np.testing.assert_allclose(X.to_numpy(dtype=float), values, equal_nan=True)
    if with_test:
        names, values = _R_matrix(rpy, 'r$xtest')
        assert list(X_test.columns) == names
        np.testing.assert_allclose(X_test.to_numpy(dtype=float), values, equal_nan=True)
    if na_impute:
        with (rpy.default_converter + rpy.numpy2ri.converter).context():
            means = np.asarray(rpy.r(f'attr({R_x}, "means")'))
        np.testing.assert_allclose(X.attrs['means'].to_numpy(), means, equal_nan=True)
        assert not np.any(np.isnan(X.to_numpy(dtype=float)))
    else:
        assert 'means' not in X.attrs


def test_make_x_single_level_warns():
    tr, _ = _data()
    with pytest.warns(UserWarning, match='sonly'):
        X = make_x(tr[['s']])
    np.testing.assert_array_equal(X['sonly'], 1)


def test_make_x_index_and_errors():
    tr, te = _data()
    te.index = [10, 11]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X, X_test = make_x(tr, te)
    assert list(X_test.index) == [10, 11]
    with pytest.raises(ValueError, match='missing columns'):
        make_x(tr, te.drop(columns=['a']))
    with pytest.raises(TypeError):
        make_x(tr.to_numpy())
    with pytest.raises(TypeError, match='unsupported dtype'):
        make_x(pd.DataFrame({'d': pd.to_datetime(['2020-01-01', '2020-01-02'])}))


@pytest.mark.parametrize('sparse', [False, True])
def test_make_x_fits(sparse):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({'a': rng.standard_normal(n),
                       'g': pd.Categorical(rng.choice(['u', 'v', 'w'], n))})
    y = df['a'] + (df['g'] == 'v') + rng.standard_normal(n)
    X = make_x(df, sparse=sparse)
    fit = GaussNet().fit(X, y)
    assert fit.feature_names_in_ == ['a', 'gu', 'gv', 'gw']
    dense = GaussNet().fit(make_x(df), y)
    np.testing.assert_allclose(fit.coefs_, dense.coefs_, atol=1e-10)
