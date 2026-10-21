import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from glmnet import GaussNet, MultiGaussNet

rng = np.random.default_rng(0)
n, p = 100, 6
X = rng.standard_normal((n, p))
y = X[:, 0] - 0.5 * X[:, 1] + rng.standard_normal(n)
Xdf = pd.DataFrame(X, columns=[f'v{j}' for j in range(p)])


def _labels(ax):
    return {t.get_text(): t for t in ax.texts}


@pytest.mark.parametrize('xvar', ['-lambda', 'lambda', 'norm', 'dev'])
def test_label(xvar):
    fit = GaussNet().fit(Xdf, y)
    path = fit.coef_path_
    fig, ax = plt.subplots()
    path.plot(xvar=xvar, ax=ax, label=True)
    labels = _labels(ax)

    nonzero = [f'v{j}' for j in range(p) if np.any(path.coefs[:, j] != 0)]
    assert sorted(labels) == sorted(nonzero)

    x_end = {'-lambda': -np.log(path.lambda_values[-1]),
             'lambda': np.log(path.lambda_values[-1]),
             'norm': np.fabs(path.coefs[-1]).sum(),
             'dev': path.fracdev[-1]}[xvar]
    for name, text in labels.items():
        j = int(name[1:])
        np.testing.assert_allclose(text.xy, (x_end, path.coefs[-1, j]))
        # the path ends on the left for xvar='lambda'
        assert text.get_ha() == ('right' if xvar == 'lambda' else 'left')
    plt.close(fig)


def test_no_label_by_default():
    fit = GaussNet().fit(X, y)
    fig, ax = plt.subplots()
    fit.coef_path_.plot(ax=ax)
    assert len(ax.texts) == 0
    plt.close(fig)


def test_label_keep_and_existing_lines():
    fit = GaussNet().fit(Xdf, y)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], c='red')
    fit.coef_path_.plot(ax=ax, label=True, keep=['v0', 'v1'])
    labels = _labels(ax)
    assert set(labels) == {'v0', 'v1'}
    # colours match the curves, not the line drawn before
    lines = ax.get_lines()[1:3]
    for line, name in zip(lines, ['v0', 'v1']):
        assert labels[name].get_color() == line.get_color()
    plt.close(fig)


def test_label_multiresponse():
    Y = np.column_stack([y, -y + rng.standard_normal(n)])
    fit = MultiGaussNet().fit(Xdf, Y)
    path = fit.coef_path_
    fig, ax = plt.subplots()
    path.plot(label=True, ax=ax)
    labels = _labels(ax)
    norms = np.sqrt((path.coefs**2).sum(-1))
    assert sorted(labels) == sorted(f'v{j}' for j in range(p) if np.any(norms[:, j] != 0))
    for name, text in labels.items():
        np.testing.assert_allclose(text.xy[1], norms[-1, int(name[1:])])
    plt.close(fig)


def test_df():
    fit = GaussNet().fit(Xdf, y)
    path = fit.coef_path_
    D = path.df_
    assert list(D.columns) == list(Xdf.columns)
    np.testing.assert_allclose(D.index, -np.log(path.lambda_values))
    assert D.index.name == r'$-\log(\lambda)$'
    np.testing.assert_allclose(D.values, path.coefs)


def test_df_multi_response():
    Y = np.column_stack([y, rng.standard_normal(n)])
    path = MultiGaussNet().fit(Xdf, Y).coef_path_
    D = path.df_
    assert D.shape == (path.coefs.shape[0], p * 2)
    np.testing.assert_allclose(D[('v0', 1)], path.coefs[:, 0, 1])


@pytest.mark.parametrize('multi', [False, True])
def test_plot_lines_are_df(multi):
    # the curves are the columns of df_ (their l2 norms over the responses)
    if multi:
        Y = np.column_stack([y, rng.standard_normal(n)])
        path = MultiGaussNet().fit(Xdf, Y).coef_path_
        expected = np.sqrt((path.coefs**2).sum(-1))
    else:
        path = GaussNet().fit(Xdf, y).coef_path_
        expected = path.coefs
    fig, ax = plt.subplots()
    path.plot(ax=ax, legend=True)
    lines = ax.get_lines()[:p]
    assert [l.get_label() for l in lines] == list(Xdf.columns)
    for j, line in enumerate(lines):
        np.testing.assert_allclose(line.get_xdata(), -np.log(path.lambda_values))
        np.testing.assert_allclose(line.get_ydata(), expected[:, j])
    plt.close(fig)
