"""
Tests of an unpenalized `GLM` or `CoxLM`: general linear hypotheses
`L beta = rhs` (`linear_hypothesis`), ANOVA tables by term of a
modelmatrix `ModelSpec` (`anova_table`, types I and III), and the sandwich
covariance with a meat of one's own (`sandwich_covariance`, from the
observations' scores, `score_contributions`).

Each takes the covariance of the coefficients as `covariance=`: the fit's
(the default: model-based, or the sandwich with `sandwich=True`),
`'model'`, `'sandwich'` (HC0), or an array, e.g. `sandwich_covariance(fit,
meat=...)` or `sandwich_covariance(fit, groups=...)`. The statistics are t
and F (on `df_resid_`) for least squares with a model-based covariance,
otherwise z and chi-squared.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dbn, f as f_dbn, norm as normal_dbn, t as t_dbn
from sklearn.base import clone

from modelmatrix import ModelSpec, Column

from .glm import _covariance_of, _use_t
from ._frames import as_pandas


def _check(fit):
    if not getattr(fit, '_unpenalized', False):
        raise ValueError(f'{type(fit).__name__}: tests are for an unpenalized GLM or CoxLM')


def score_contributions(fit):
    """
    The scores of the coefficients (`params_`) of the observations of an
    unpenalized `GLM` or `CoxLM`, one row each, without the dispersion: the
    sum of the outer products of the rows is the meat of HC0's sandwich.
    For a Cox model, the score residuals (Lin and Wei, 1989).

    Returns
    -------
    pd.DataFrame
        n x p, its columns `feature_names_`, on the index of the fit's `X`
        if it was a DataFrame.
    """
    _check(fit)
    S = fit._score_contributions()
    index = getattr(fit, '_train_index', None)
    return pd.DataFrame(S, columns=fit.feature_names_, index=index)


def sandwich_covariance(fit, meat=None, groups=None):
    """
    The sandwich covariance `B M B` of the coefficients of an unpenalized
    `GLM` or `CoxLM`, the bread `B` the inverse of their information
    (without the dispersion).

    Parameters
    ----------
    meat: np.ndarray, optional
        A p x p meat of one's own, on the scale of `score_contributions(fit)`
        (whose outer products sum to HC0's): e.g. HC1, HC2 or HC3 (the rows
        scaled by n / (n - p), 1 / (1 - h) or 1 / (1 - h)^2, `h` the
        `leverage`), sums over blocks for serial correlation, or the
        covariance of the score over a bootstrap.
    groups: array-like, optional
        A label per observation: the cluster-robust meat, the sum over the
        clusters of the outer products of their sums of the scores, without
        a small-sample correction.

    By default the meat is HC0's. As `covariance=` of `linear_hypothesis`,
    `anova_table` or `summarize`.

    Returns
    -------
    pd.DataFrame
        p x p, labelled by `feature_names_`.
    """
    _check(fit)
    if meat is not None and groups is not None:
        raise ValueError('give meat or groups, not both')
    bread = np.linalg.inv(fit._precision())
    p = bread.shape[0]
    if groups is not None:
        S = fit._score_contributions()
        groups = np.asarray(groups)
        if groups.shape[0] != S.shape[0]:
            raise ValueError(f'groups has {groups.shape[0]} labels, the fit '
                             f'{S.shape[0]} observations')
        G = pd.DataFrame(S).groupby(groups, sort=False).sum().to_numpy()
        meat = G.T @ G
    elif meat is None:
        meat = fit._score_meat()
    meat = np.asarray(meat, float)
    if meat.shape != (p, p):
        raise ValueError(f'the meat should be {p} x {p}, as params_, got {meat.shape}')
    names = fit.feature_names_
    return pd.DataFrame(bread @ meat @ bread, index=names, columns=names)


@dataclass
class LinearHypothesis(object):
    """
    A general linear hypothesis `L beta = rhs` of a fit: each row's estimate
    (`estimates`), and the joint test of all the rows.

    Attributes
    ----------
    estimates: pd.DataFrame
        For each row of `L`: `estimate` (`L beta`), `rhs`, `std err`, `t` or
        `z` (of `estimate - rhs`), its p-value, and the confidence interval
        of `L beta`.
    test: str
        `'F'` or `'chi2'`.
    statistic: float
        F, `W' (L V L')^{-1} W / q`, or chi-squared, `W' (L V L')^{-1} W`,
        with `W = L beta - rhs`, `V` the covariance, `q` the rows.
    df: tuple
        `(q, df_resid_)` for F, `(q,)` for chi-squared.
    p_value: float
    """
    estimates: pd.DataFrame
    test: str
    statistic: float
    df: tuple
    p_value: float

    def __repr__(self):
        df = ', '.join(f'{d:g}' for d in self.df)
        return (f'{self.estimates!r}\n\n{self.test}({df}) = {self.statistic:.4g}, '
                f'p = {self.p_value:.4g}')


def linear_hypothesis(fit, L, rhs=0, covariance=None, level=0.95):
    """
    Test the general linear hypothesis `L beta = rhs` of the coefficients
    (`params_`) of an unpenalized `GLM` or `CoxLM`: the Wald test, as R's
    `car::linearHypothesis` and statsmodels' `f_test` / `wald_test`.

    Parameters
    ----------
    fit: GLM or CoxLM
    L: dict, list of dicts, pd.DataFrame, pd.Series or array
        The rows of `L`. By name (`fit.feature_names_`, the intercept
        `'intercept'`): a dict (or Series) is a row, e.g. `{'c[b]': 1,
        'c[c]': -1}` (`c[b] = c[c]`), a list of them or a DataFrame (its
        index naming the rows) several; a name left out is 0, so the
        intercept enters only if named. As an array: a column for each
        coefficient, the intercept first if it is fitted.
    rhs: float or array
        The right-hand side, one per row (default 0).
    covariance: None, 'model', 'sandwich' or np.ndarray
        The covariance of the coefficients (see `glmnet.hypothesis`).
    level: float
        The level of the confidence intervals.

    Returns
    -------
    LinearHypothesis
    """
    _check(fit)
    Lmat, labels = _contrast_matrix(fit, L)
    q = Lmat.shape[0]
    rhs = np.broadcast_to(np.asarray(rhs, float), (q,)).astype(float)
    C, model_based = _covariance_of(fit, covariance)
    use_t = _use_t(fit, model_based)
    beta = fit.params_.to_numpy()
    estimate = Lmat @ beta
    V = Lmat @ C @ Lmat.T
    if np.linalg.matrix_rank(V) < q:
        raise ValueError('the rows of L are linearly dependent (or not estimable at this '
                         'covariance): give independent rows')
    se = np.sqrt(np.diag(V))
    T = (estimate - rhs) / se
    alpha = 1 - level
    if use_t:
        stat, pval = 't', 2 * t_dbn.sf(np.fabs(T), df=fit.df_resid_)
        quantile = t_dbn.ppf(1 - alpha / 2, df=fit.df_resid_)
    else:
        stat, pval = 'z', 2 * normal_dbn.sf(np.fabs(T))
        quantile = normal_dbn.ppf(1 - alpha / 2)
    estimates = pd.DataFrame({'estimate': estimate,
                              'rhs': rhs,
                              'std err': se,
                              stat: T,
                              f'P>|{stat}|': pval,
                              f'[{alpha / 2:g}': estimate - quantile * se,
                              f'{1 - alpha / 2:g}]': estimate + quantile * se},
                             index=labels)
    W = estimate - rhs
    wald = float(W @ np.linalg.solve(V, W))
    if use_t:
        F = wald / q
        return LinearHypothesis(estimates, 'F', F, (q, fit.df_resid_),
                                float(f_dbn.sf(F, q, fit.df_resid_)))
    return LinearHypothesis(estimates, 'chi2', wald, (q,), float(chi2_dbn.sf(wald, q)))


def _contrast_matrix(fit, L):
    """The rows of L as a q x p array over `params_`, and their labels."""
    names = list(fit.feature_names_)
    p = len(names)
    if isinstance(L, (dict, pd.Series)):
        L = [L]
    if isinstance(L, list) and L and all(isinstance(row, (dict, pd.Series)) for row in L):
        L = pd.DataFrame([dict(row) for row in L], index=[_label(row) for row in L])
    if isinstance(L, pd.DataFrame):
        unknown = [c for c in L.columns if c not in names]
        if unknown:
            raise ValueError(f'unknown coefficients {unknown}: the coefficients are {names}')
        return L.reindex(columns=names).fillna(0).to_numpy(float), list(L.index)
    L = np.atleast_2d(np.asarray(L, float))
    if L.shape[1] != p:
        if fit.fit_intercept and L.shape[1] == p - 1:
            raise ValueError(f'L has {L.shape[1]} columns, the fit {p} coefficients with the '
                             'intercept: give a column for the intercept first (0 to leave it '
                             "out), or give L by name, e.g. {'x': 1}")
        raise ValueError(f'L has {L.shape[1]} columns, the fit {p} coefficients ({names})')
    return L, list(range(L.shape[0]))


def _label(row):
    # a row by name, as an expression: 'c[b] - c[c]'
    terms = []
    for name, value in dict(row).items():
        if value == 0:
            continue
        sign = '-' if value < 0 else '+'
        size = '' if abs(value) == 1 else f'{abs(value):g} '
        terms.append(f'{sign} {size}{name}')
    text = ' '.join(terms)
    return text[2:] if text.startswith('+ ') else text


def anova_table(estimator, spec, X, y, type='I', covariance=None):
    """
    The ANOVA table, by term, of an unpenalized `GLM` or `CoxLM` of the
    design of a modelmatrix `ModelSpec`.

    - Type I (sequential, R's `anova(lm)` and `anova(glm, test="Chisq")`):
      each term after those before it, by the drop in deviance (the sum of
      squares, for least squares) of the fits of the terms in turn, from
      the null model; F on the full model's dispersion and residual df for
      least squares, else chi-squared of the drop over the dispersion.
    - Type III (`car::Anova(type=3)`): each term (and the intercept) after
      all the others, the Wald test that its coefficients are 0
      (`linear_hypothesis`), at `covariance`. Its tests of the main effects
      of a factor in an interaction depend on the factor's contrasts: with
      sum contrasts (`contrast('f', 'sum')`) they are the usual ones.

    Parameters
    ----------
    estimator: GLM or CoxLM
        Its options (family, weights...) are used; it is cloned.
    spec: modelmatrix.ModelSpec
        The terms, without an intercept (the estimator fits it); fit on `X`
        unless it is fitted.
    X, y:
        The data of `spec`, and the response (as for `estimator.fit`).
    type: {'I', 'III'}
    covariance:
        For type III: the covariance of the coefficients (see
        `glmnet.hypothesis`).

    Returns
    -------
    pd.DataFrame
        A row for each term (type I: and the residuals; type III: and the
        intercept), with `df`, (type I) `deviance`, and `F` and `P(>F)` or
        `chi2` and `P(>chi2)`.
    """
    type = str(type).upper()
    if type not in ('I', 'III', '1', '3'):
        raise ValueError(f"type should be 'I' or 'III', got {type!r}")
    if not isinstance(spec, ModelSpec):
        raise TypeError(f'spec should be a modelmatrix ModelSpec, got {spec.__class__.__name__}')
    if spec.intercept:
        raise ValueError('the ModelSpec of anova_table has no intercept (the estimator fits '
                         'it): ModelSpec(..., intercept=False)')
    X, y = as_pandas(X), as_pandas(y)
    if not hasattr(spec, 'column_map_'):
        spec = clone(spec).fit(X)
    D = spec.transform(X)
    if not isinstance(D, pd.DataFrame):
        D = pd.DataFrame(np.asarray(D), columns=spec.get_feature_names_out())
    terms = [(feature.name, spec.column_map_[term])
             for term, feature in zip(spec.expanded_terms_, spec.terms_)]
    full = clone(estimator).fit(D, y)
    _check(full)
    if type in ('I', '1'):
        return _type_I(estimator, full, D, y, terms)
    _warn_contrasts(spec)
    rows = []
    if full.fit_intercept:
        rows.append(('intercept', {'intercept': 1}))
    for name, columns in terms:
        rows.append((name, [{c: 1} for c in D.columns[columns]]))
    table = {}
    for name, L in rows:
        result = linear_hypothesis(full, L, covariance=covariance)
        table[name] = {'df': result.df[0], result.test: result.statistic,
                       f'P(>{result.test})': result.p_value}
    return pd.DataFrame(table).T.astype(float)


def _type_I(estimator, full, D, y, terms):
    if full.sandwich:
        warnings.warn('the fit uses the sandwich covariance: the drop in deviance is not '
                      'chi-squared when the model is wrong, so the type I p-values are not '
                      'robust')
    n = int(np.sum(np.asarray(full._weight) > 0))
    df, dev = [n - full.fit_intercept], [full.null_deviance_]
    for k in range(1, len(terms) + 1):
        stop = terms[k - 1][1].stop
        fit = full if k == len(terms) else clone(estimator).fit(D.iloc[:, :stop], y)
        df.append(fit.df_resid_)
        dev.append(fit.deviance_)
    df, dev = np.array(df, float), np.array(dev, float)
    table = pd.DataFrame({'df': -np.diff(df), 'deviance': -np.diff(dev)},
                         index=[name for name, _ in terms])
    scale = full.dispersion_
    if _use_t(full, not full.sandwich):
        table['F'] = table['deviance'] / table['df'] / scale
        table['P(>F)'] = f_dbn.sf(table['F'], table['df'], full.df_resid_)
    else:
        table['chi2'] = table['deviance'] / scale
        table['P(>chi2)'] = chi2_dbn.sf(table['chi2'], table['df'])
    table.loc['Residuals', ['df', 'deviance']] = [full.df_resid_, full.deviance_]
    return table


def _warn_contrasts(spec):
    # type III: the main effects of a factor in an interaction depend on its
    # contrasts; with treatment ('drop') contrasts they are not the usual tests
    for term in spec.expanded_terms_:
        if not isinstance(term, tuple) or isinstance(term, Column):
            continue
        for part in term:
            info = (part if isinstance(part, Column)
                    else spec.column_info_.get(part) if isinstance(part, (str, int)) else None)
            if info is None or not getattr(info, 'is_categorical', False):
                continue
            method = getattr(getattr(info, 'encoder', None), 'method', None)
            if method != 'sum':
                warnings.warn(f'type III: {info.name!r} is in an interaction with '
                              f'{method!r} contrasts, so the tests of its main effects '
                              f"depend on them; with contrast({info.name!r}, 'sum') they "
                              'are the usual ones', UserWarning, stacklevel=3)
                return


__all__ = ['score_contributions', 'sandwich_covariance', 'linear_hypothesis',
           'LinearHypothesis', 'anova_table']
