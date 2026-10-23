"""
Generalized additive models fitted by local scoring (Hastie & Tibshirani):
IRLS, whose working problems are solved by backfitting the terms.

References
----------
- Hastie, T. and Tibshirani, R. (1986). Generalized additive models.
  *Statistical Science* 1(3), 297–318.
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*.
  Chapman & Hall.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and
  marginal likelihood estimation of semiparametric generalized linear
  models. *Journal of the Royal Statistical Society, Series B* 73(1), 3–36.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman & Hall/CRC.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from statsmodels.genmod.families import family as sm_family

from glmnet._utils import _get_data, _parent_dataclass_from_child
from glmnet.family import GLMFamilySpec
from glmnet.cox import CoxFamily, CoxFamilySpec
from glmnet.glm import GLM, GLMControl
from glmnet.irls import IRLS
from .backfit import GAMRegularizer, GAMState
from .display import model_latex
from .joint import JointDesign, JointRegularizer
from .plot import plot_terms
from .reml import ExperimentalWarning, _blocked_term
from .selection import resolve_method, SmoothingMethod
from .terms import SplineTerm, Term, SlopeTerm, ColumnsTerm, FactorTerm


@dataclass
class GAMControl(GLMControl):
    """
    Control parameters for fitting a GAM: those of `GLMControl` for the IRLS
    iterations, and for the backfitting of each working problem.

    Parameters
    ----------
    bf_maxit: int
        Maximum number of backfitting sweeps per IRLS iteration.
    bf_tol: float
        Backfitting stops when the change in the fits of the terms, relative
        to `0.1 + |fits|` (Frobenius norms), is below `bf_tol`.
    bf_modified: bool
        Modified backfitting (the default, as R's `gam`): each sweep fits the
        linear parts of all the terms jointly, then backfits the smooths'
        nonlinear parts. False: plain backfitting.
    solver: str
        How each working problem is solved: `'backfit'` (the default, as R's
        `gam`) or `'joint'`, the penalized least squares fit on the terms'
        bases side by side (as mgcv; for terms with a basis, not loess). They
        have the same solution.
    max_dense_coef: int
        The most coefficients (1 + the terms' basis columns) for which the
        dense `P x P` matrices are formed: the posterior and frequentist
        covariances (standard errors, `edf_influence_`, mgcv's edf) and REML
        and ML's Hessian. Above it they are not formed (None; standard errors
        raise), and REML and ML raise, but for one large random effect (see
        `reml_backend`): a random effect of many levels is fitted by
        backfitting without them.
    reml_backend: str
        The factorization of REML and ML's Hessian: `'dense'`, `'blocked'`
        (the random effect with the most levels kept implicit, and the
        Hessian factored by blocks, `glmnet.gam.blocked`: exact, without the
        `P x P` matrix), or `'auto'` (the default: blocked when there are more
        than `max_dense_coef` coefficients and one random effect has all but
        at most that many).
    """
    bf_maxit: int = 500
    bf_tol: float = 1e-10
    bf_modified: bool = True
    solver: str = 'backfit'
    max_dense_coef: int = 20000
    reml_backend: str = 'auto'


def _cox_information(family, sample_weight, state):
    """The score and the information of the Cox partial likelihood in the linear predictor."""
    return (state.logl_score(family, None, sample_weight),
            family.information(state, sample_weight))


@dataclass
class GAM(BaseEstimator):
    """
    Generalized additive model, `g(E[y]) = intercept + sum_j f_j(x_j)`,
    fitted by local scoring: IRLS, each working problem solved by weighted
    backfitting of the terms (Hastie & Tibshirani, 1986, 1990); or, with
    `GAMControl(solver='joint')`, jointly on the terms' bases, and with
    `method='REML'` or `'ML'` with the smoothing parameters estimated, as
    mgcv (Wood, 2011, 2017). See the References of `glmnet.gam`.

    The model is a list of terms (`glmnet.gam.s`, `lo`, `re`, `gaussian_process`, `linear`,
    `factor`, or a `ColumnsTerm` of several columns fitted as one parametric
    block), each naming a column of `X` (ISLP's `ISLP.gam` builds them from
    a ModelSpec). The smoothing of each term is fixed: by its smoothing parameter
    `lam`, or by its degrees of freedom `df` (as R's `gam`). When every term
    has a fixed smoothing parameter (`s(lam=...)`, `re(lam=...)`, `linear`,
    `factor`), the fit minimizes

        deviance / (2 sum(w)) + sum_j lam_j / 2 * pen_j(f_j) + linear_term @ eta,

    with `pen_j` the integral of the squared second derivative for a spline,
    the sum of the squared effects for a random effect, `eta` the linear
    predictor (without the offset) and `linear_term` the optional argument
    of `fit`. Otherwise (`df` terms, loess) local scoring stops on the change
    in `eta`. The working problems are solved by backfitting (the default) or
    jointly (`control.solver='joint'`): the same fit.

    Parameters
    ----------
    terms: list
        Terms of the model.
    family: sm_family.Family or glmnet.cox.CoxFamily
        Family (statsmodels) of the response. Default Gaussian. Or the Cox
        model's (glmstar's `CoxFamily`, with Efron or Breslow ties, start
        times and strata): `y` is then a DataFrame of the event data (and
        the offset and weights), the fit has no intercept and maximizes the
        penalized partial likelihood, with the diagonal of its Hessian as
        the working weights of backfitting (`solver='joint'`: Newton steps
        with the full Hessian, which also gives the covariance, as mgcv's
        `cox.ph`); `predict` gives the linear predictor (`'link'`) or the
        relative risk `exp(eta)` (`'response'`).
    control: GAMControl
        Parameters for the IRLS iterations and the backfitting.
    offset_id: Union[str,int]
        Column identifier in `y`. (Optional)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional)
    method: str or SmoothingMethod
        How the smoothing parameters are chosen: `'fixed'` (the default, as
        R's `gam`: each term's `lam` or `df`; `s()` without them has
        `df=4`), `'REML'` or `'ML'`, or a `glmnet.gam.selection.SmoothingMethod`
        (a user's: the documentation's page "Crossed random effects at
        scale" writes one, the method of moments for the variances of
        crossed random effects in large data). With `'REML'`
        or `'ML'` (as mgcv's `gam(method=...)`, with JAX) the smoothing
        parameters of the penalized terms without `lam` or `df` (or a given
        precision) maximize the restricted (or full) marginal likelihood,
        by Laplace's method, the dispersion with them. Terms with `lam` keep
        it; terms with `df` have it matched at the prior weights, then fixed.
        The working problems are solved jointly, and `edf_`, `df_resid_`
        and `dispersion_` follow mgcv (its edf, its residual df, the REML
        or ML estimate of the dispersion). **Experimental**: `'REML'` and
        `'ML'` warn (`glmnet.gam.ExperimentalWarning`); check
        `smoothing_.converged`.

    Attributes
    ----------
    terms_: list
        The fitted terms (copies of `terms`).
    intercept_: float
        Intercept: the fits of the terms have mean 0 over the training rows.
    fits_: pd.DataFrame or np.ndarray
        The fits of the terms at the training rows, one column per term.
    edf_: pd.Series
        Degrees of freedom of each term, without the constant: for a spline,
        the trace of its smoother at the final weights less 1 (as R's `gam`).
    edf_influence_: pd.Series
        Degrees of freedom of each term in the joint fit (as mgcv's `edf`):
        the sums over its coefficients of the diagonal of the influence
        matrix `F = (X' W X + S)^{-1} X' W X`, at the final weights. None if
        a term has no basis.
    posterior_covariance_: np.ndarray
        The Bayesian (posterior) covariance of the coefficients of the joint
        design (`design_`), `dispersion (X' W X + S)^{-1}` (W the weights of
        the final working problem, with the prior weights; S the penalties):
        mgcv's `Vp`. None if a term has no basis.
    frequentist_covariance_: np.ndarray
        The frequentist covariance of the same coefficients, `V X' W X V /
        dispersion` with `V = posterior_covariance_`: mgcv's `Ve`. None if
        a term has no basis.
    design_: JointDesign
        The terms' bases side by side (None if a term has none).

    These four are computed when first used (by `predict(...,
    return_std=True)`, `partial_effect`, `summary`).
    df_resid_: float
        Residual degrees of freedom, the number of observations (of positive
        weight) less 1 less the sum of `edf_` (as R's `gam`).
    deviance_, null_deviance_, dispersion_, loglike_, aic_: float
        As for `GLM`; the AIC counts `1 + sum(edf_)` parameters, plus 1 for a
        free dispersion, as R's `gam` (for the Gamma and inverse Gaussian the
        log-likelihood is at the dispersion deviance / sum(weights), as R's),
        plus 1 for an estimated negative binomial theta. A quasi family
        (`glmnet.gam.families`) has no log-likelihood or AIC (NaN).
    negative_binomial_theta_: float
        The negative binomial's theta, estimated (`NegativeBinomial(theta=None)`
        with `method='REML'` or `'ML'`, as mgcv's `nb()`) or given; None for
        other families.
    n_iter_: int
        Number of IRLS iterations.
    smoothing_: SmoothingSelection
        With `method='REML'` or `'ML'`: the estimated smoothing parameters,
        the dispersion and the criterion (`glmnet.gam.reml`).
    reml_score_: float
        The minimized criterion (the negative log REML or ML; as mgcv's
        `gcv.ubre`, up to a constant for REML, which depends on the
        parametrization of the unpenalized coefficients).
    variance_components_: pd.DataFrame
        With REML or ML: the variances (and standard deviations) of the
        penalized terms' coefficients as random effects, the dispersion over
        their smoothing parameters (on the prior weights' scale), and the
        dispersion (`scale`), as mgcv's `gam.vcomp` and lme4's `VarCorr`
        for random effects.
    """
    terms: list = field(default_factory=list)
    family: object = field(default_factory=GLMFamilySpec)
    control: GAMControl = field(default_factory=GAMControl)
    offset_id: Union[str, int] = None
    weight_id: Union[str, int] = None
    response_id: Union[str, int] = None
    method: str = 'fixed'

    fit_intercept = True

    # hooks for a subclass that builds the terms and the columns they name
    # from a model specification (ISLP's GAM, from a ModelSpec)

    def _model_terms(self, X):
        """The terms of the model and the data whose columns they name."""
        terms = list(self.terms)
        for term in terms:
            if not isinstance(term, Term):
                raise ValueError(f'{term!r} is not a GAM term (s(), lo(), re(), '
                                 'gaussian_process(), linear(), factor())')
        return terms, X

    def _design(self, X):
        """The data whose columns the fitted terms name (`X`)."""
        return X

    @staticmethod
    def _columns(X, terms):
        """The columns of `X` of the terms."""
        cols = []
        for term in terms:
            if isinstance(X, pd.DataFrame):
                # a name or a list of names (a ColumnsTerm), else positions
                column = term.columns
                labels = column if isinstance(column, list) else [column]
                if all(isinstance(c, str) for c in labels):
                    cols.append(X[column].to_numpy())
                else:
                    cols.append(X.iloc[:, column].to_numpy())
            else:
                cols.append(np.asarray(X)[:, term.columns])
        return cols

    def fit(self,
            X,
            y,
            linear_term=None,
            dispersion=None):
        """
        Fit the GAM.

        Parameters
        ----------
        X: Union[np.ndarray, pd.DataFrame]
            The data: the terms name its columns.
        y: Union[np.ndarray, pd.DataFrame]
            Response (with offset and weights, see `offset_id`, `weight_id`).
        linear_term: np.ndarray, optional
            A linear term `linear_term @ eta` added to the objective, one entry
            per observation (0 where the weight is 0), with `eta` the linear
            predictor without the offset; as `GLMNet`'s `linear_term`, on the
            scale of the deviance over twice the sum of the weights.
        dispersion: float, optional
            Dispersion. By default it is estimated, by Pearson's chi-squared
            over `df_resid_`, for families with a free dispersion, and is 1
            for the binomial and Poisson.

        Returns
        -------
        self
        """
        terms, X = self._model_terms(X)
        if not terms:
            raise ValueError('a GAM needs at least one term')
        if self.control is None:
            self.control = GAMControl()
        elif isinstance(self.control, dict):
            self.control = _parent_dataclass_from_child(GAMControl, self.control)

        self._train_index = X.index if isinstance(X, pd.DataFrame) else None
        nobs = X.shape[0]

        self._cox = isinstance(self.family, CoxFamily)
        if self._cox:
            family, response, offset, weight = self._cox_data(y, nobs)
            self._family = family
        else:
            # (a negative binomial whose theta is estimated: a copy, its
            # theta set at each fit)
            estimate_theta = getattr(self.family, 'estimate_theta', False)
            family = self._family = GLMFamilySpec.from_family(
                deepcopy(self.family) if estimate_theta else self.family, y)
            _, _, response, offset, weight = _get_data(self,
                                                       np.zeros((nobs, 1)),
                                                       y,
                                                       offset_id=self.offset_id,
                                                       response_id=self.response_id,
                                                       weight_id=self.weight_id,
                                                       check=False)
            response = np.asarray(response, float).reshape(-1)
            weight = np.asarray(weight, float).reshape(-1)
        self._response, self._weight = response, weight
        weight_total = weight.sum()
        normed_weight = self._normed_weight = weight / weight_total
        self.weight_total_ = weight_total

        method = self._method = resolve_method(self.method)
        if getattr(self.family, 'estimate_theta', False) and not method.estimates:
            raise ValueError('the negative binomial\'s theta is estimated by method="REML" or '
                             '"ML": give theta (NegativeBinomial(theta=...)) to fix it')
        self.terms_ = self._resolve_terms([deepcopy(term) for term in terms])
        for term, x in zip(self.terms_, self._columns(X, terms)):
            factor_column = (term.column if isinstance(term, FactorTerm)
                             else getattr(term, 'factor_column', None))
            if factor_column is not None and isinstance(X, pd.DataFrame):
                # a categorical's order of the levels (not the sorted one)
                column = (X[factor_column] if isinstance(factor_column, str)
                          else X.iloc[:, factor_column])
                if isinstance(column.dtype, pd.CategoricalDtype):
                    term.categories = list(column.cat.categories)
            term.setup(x)
        if linear_term is not None:
            linear_term = np.asarray(linear_term, float).reshape(-1)
            if linear_term.shape != (nobs,):
                raise ValueError(f'linear_term should have shape {(nobs,)}')

        if self.control.solver not in ('backfit', 'joint'):
            raise ValueError("solver should be one of 'backfit' or 'joint'")
        # REML and ML: the joint solve, its design needed for the criterion;
        # with a large random effect (the blocked criterion) backfitting,
        # which does not form the design
        blocked = method.joint and _blocked_term(self) is not None
        Regularizer = (JointRegularizer if self.control.solver == 'joint' or
                       (method.joint and not blocked) else GAMRegularizer)
        regularizer = self.regularizer_ = Regularizer(terms=self.terms_,
                                                      linear_term=linear_term,
                                                      bf_maxit=self.control.bf_maxit,
                                                      bf_tol=self.control.bf_tol,
                                                      modified=self.control.bf_modified,
                                                      fit_intercept=not self._cox)
        if self._cox and isinstance(regularizer, JointRegularizer):
            # Newton steps with the full Hessian of the partial likelihood
            regularizer.information = partial(_cox_information, family, normed_weight)

        null_state, null_deviance = family.get_null_deviance(response=response,
                                                             sample_weight=normed_weight,
                                                             offset=offset,
                                                             fit_intercept=not self._cox)
        if self._cox:
            # no intercept: the null fit is the offset (as R's coxph)
            null_deviance = family.deviance(response,
                                            np.zeros(nobs) if offset is None else offset,
                                            normed_weight)
        self.null_deviance_ = weight_total * null_deviance
        null_intercept = 0. if self._cox else float(null_state.coef[0])

        self._offset, self._normed_weight = offset, normed_weight
        self._null_state = GAMState(np.zeros((nobs, len(self.terms_))), null_intercept)

        if method.estimates:
            if method.experimental:
                warnings.warn(f"GAM(method='{method.name}') is experimental: the selection of the "
                              "smoothing parameters may change, and its convergence should be "
                              "checked (smoothing_.converged)", ExperimentalWarning, stacklevel=2)
            self.smoothing_ = method.select(self, dispersion)
            if getattr(self.family, 'estimate_theta', False):
                theta = getattr(self.smoothing_, 'negative_binomial_theta', None)
                if theta is None:
                    raise ValueError(f'the method {method.name} does not estimate the negative '
                                     'binomial\'s theta')
                family.base.set_theta(theta)
                # the null deviance at the estimated theta (as R's glm.nb)
                _, null_deviance = family.get_null_deviance(response=response,
                                                            sample_weight=normed_weight,
                                                            offset=offset,
                                                            fit_intercept=True)
                self.null_deviance_ = weight_total * null_deviance
        base = getattr(family, 'base', None)
        self.negative_binomial_theta_ = (1 / base.alpha if isinstance(base, sm_family.NegativeBinomial)
                                         else None)

        state = self._local_scoring()

        # the terms at the weights of the final working problem
        z, w = family.get_response_and_weights(state, response, offset, normed_weight)
        total = state.fits.sum(1)
        coefs = state.coefs if state.coefs is not None else [None] * len(self.terms_)
        edf = []
        for j, term in enumerate(self.terms_):
            term.set_weights(w)
            edf.append(term.edf())
            term.finalize(state.fits[:, j], z - state.intercept - (total - state.fits[:, j]),
                          coefs[j])

        names = [term.name for term in self.terms_]
        self.state_ = state
        self.intercept_ = state.intercept
        self.fits_ = self._on_train_index_frame(state.fits, names)
        self.edf_ = pd.Series(edf, index=names, name='edf')

        self.deviance_ = weight_total * family.deviance(response, state.mu, normed_weight)
        # the intercept counts as one degree of freedom (none for Cox)
        self.df_resid_ = float(np.sum(weight > 0) - (not self._cox) - self.edf_.sum())

        self._estimated_dispersion = dispersion is None and not self._fixed_dispersion()
        if method.estimates:
            # the method's estimate of the dispersion; mgcv's conventions
            # (its edf and residual df) if the method asks for them
            self.dispersion_ = (self.smoothing_.dispersion if self._estimated_dispersion
                                else (1. if dispersion is None else dispersion))
            self._set_covariance(w)
            if method.covariance_edf:
                edf = self.edf_influence_
                if edf is None and getattr(self.smoothing_, 'edf', None) is not None:
                    # (from the blocked factorization: the dense one is not formed)
                    edf = pd.Series(self.smoothing_.edf, index=self.edf_.index)
                self.edf_ = edf.rename('edf')
                self.df_resid_ = float(np.sum(weight > 0) - (not self._cox) - self.edf_.sum())
            self.reml_score_ = self.smoothing_.score
            # the variances of the penalized terms' coefficients (random
            # effects): the dispersion over the smoothing parameter, on the
            # scale of the prior weights
            names, variances = [], []
            for term in self.terms_:
                if getattr(term, 'penalty_varies', False):
                    # correlated random effects: their covariance's variances
                    # (the covariances in `covariance_`)
                    Sigma = self.dispersion_ * np.linalg.inv(term.precision * weight_total)
                    term.covariance_ = pd.DataFrame(Sigma, index=term.effect_names,
                                                    columns=term.effect_names)
                    for name, v in zip(term.effect_names, np.diag(Sigma)):
                        names.append(f'{term.name}[{name}]')
                        variances.append(v)
                    continue
                lams = term.penalty_lams()
                for m, lam in enumerate(lams):
                    names.append(term.name if len(lams) == 1 else f'{term.name}[{m}]')
                    variances.append(self.dispersion_ / (lam * weight_total))
            names.append('scale')
            variances.append(self.dispersion_)
            self.variance_components_ = pd.DataFrame({'variance': variances,
                                                      'sd': np.sqrt(variances)}, index=names)
        elif self._estimated_dispersion:
            dispersion = (np.sum(self._residuals('pearson')**2) / self.df_resid_
                          if self.df_resid_ > 0 else np.nan)
        elif dispersion is None:
            dispersion = 1.
        if not method.estimates:
            self.dispersion_ = dispersion

        # as R's gam (and glm): the dispersion, when free, counts as a
        # parameter, and the log-likelihood is at R's estimate of it
        k = ((not self._cox) + self.edf_.sum() + (not self._fixed_dispersion()) +
             getattr(self.family, 'estimate_theta', False))
        n = np.sum(weight > 0)
        self.loglike_ = self._loglike()
        self.aic_ = -2 * self.loglike_ + 2 * k
        self.bic_ = -2 * self.loglike_ + np.log(n) * k

        if not method.estimates:
            self._set_covariance(w)
        return self

    def _resolve_terms(self, terms):
        """
        The smoothing of the terms for the method: `s()` with neither `df`
        nor `lam` has `df=4` with `method='fixed'` (as R's `gam`), and its
        smoothing parameter estimated with `'REML'` or `'ML'`, as `re()`
        without them.
        """
        method = resolve_method(self.method)
        for term in terms:
            if (isinstance(term, SplineTerm) and not method.estimates and
                term.df is None and term.lam is None):
                term.df = 4.
        return terms

    def inner_fit(self, state=None, control=None):
        """
        For a `SmoothingMethod`: the fit (local scoring) at the terms' current
        parameters, from `state` (by default the intercept-only fit), as a
        `GAMState` (its fits, intercept and means; whether it converged in
        `converged_`). The terms' parameters are set by the method (e.g.
        `term.set_penalty_lams`).
        """
        return self._local_scoring(state=state, control=control, warn=False)

    def _local_scoring(self, state=None, control=None, warn=True):
        """
        Local scoring (IRLS) from `state` (by default the intercept-only
        fit), with the terms' current smoothing: the fitted state (whether it
        converged in `converged_`).
        """
        family, response = self._family, self._response
        offset, normed_weight = self._offset, self._normed_weight
        regularizer, control = self.regularizer_, control or self.control

        def obj_function(state):
            return (family.deviance(response, state.mu, normed_weight) / 2 +
                    regularizer.objective(state))

        if state is None:
            state = deepcopy(self._null_state)
        state.update(None, family, offset, obj_function)
        info = {}
        converged, boundary, state, _ = IRLS(regularizer,
                                             family,
                                             None,
                                             response,
                                             offset,
                                             normed_weight,
                                             state,
                                             obj_function,
                                             control,
                                             info=info)
        self.n_iter_ = info['n_iter']
        self.converged_ = converged
        if not converged and warn:
            warnings.warn(f'GAM: local scoring did not converge in mxitnr={control.mxitnr} '
                          'iterations; increase control.mxitnr')
        return state

    def _set_covariance(self, w):
        """
        Keep the weights `w` of the final working problem (normalized) for
        the covariance of the joint design's coefficients, computed when
        first used.
        """
        self._covariance_weights, self._covariance = w, None
        self._fixed_covariance = None
        self._covariance_cox_weights = self._normed_weight

    def _get_covariance(self):
        """
        The joint design and the covariances of its coefficients, at the
        final weights and the terms' smoothing parameters there (as mgcv's
        `Vp`, `Ve` and `edf`); all None if a term has no basis.
        """
        if getattr(self, '_covariance', None) is not None:
            return self._covariance
        cov = dict(design=None, posterior=None, frequentist=None, edf=None)
        self._covariance_reason = None
        if not all(term.has_basis for term in self.terms_):
            self._covariance_reason = 'standard errors need a basis for every term (not lo())'
        elif 1 + sum(term.n_coef() for term in self.terms_) > self.control.max_dense_coef:
            n_coef = 1 + sum(term.n_coef() for term in self.terms_)
            self._covariance_reason = (
                f'the posterior covariance is the dense {n_coef} x {n_coef} matrix, more than '
                f'GAMControl(max_dense_coef={self.control.max_dense_coef}): raise it to form it')
        else:
            w = self._covariance_weights
            D = cov['design'] = (self.regularizer_.design_
                                 if isinstance(self.regularizer_, JointRegularizer)
                                 and self.regularizer_.design_ is not None
                                 else JointDesign(self.terms_))
            lams = [term.penalty_lams() for term in self.terms_]
            if self._cox:
                # the full information of the partial likelihood (as mgcv's
                # cox.ph); no intercept: its row and column are 0
                H = self._family.information(self.state_, self._covariance_cox_weights)
                X = D.X[:, 1:]
                XWX = np.zeros((D.n_coef, D.n_coef))
                XWX[1:, 1:] = X.T @ (H @ X)
            else:
                XWX = D.X.T @ (w[:, None] * D.X)
            A = XWX + D.penalty(lams)
            Ainv = np.linalg.pinv(A, hermitian=True)
            # on the scale of the prior weights: X' W X and S times sum(weights)
            cov['posterior'] = self.dispersion_ * Ainv / self.weight_total_
            cov['frequentist'] = self.dispersion_ * Ainv @ XWX @ Ainv / self.weight_total_
            F = np.einsum('ij,ji->i', Ainv, XWX)
            cov['edf'] = pd.Series([F[sl].sum() for sl in D.slices],
                                   index=[term.name for term in self.terms_],
                                   name='edf_influence')
        self._covariance = cov
        return cov

    @property
    def design_(self):
        return self._get_covariance()['design']

    @property
    def posterior_covariance_(self):
        return self._get_covariance()['posterior']

    @property
    def frequentist_covariance_(self):
        return self._get_covariance()['frequentist']

    @property
    def edf_influence_(self):
        return self._get_covariance()['edf']

    def _on_train_index_frame(self, value, names):
        index = getattr(self, '_train_index', None)
        if index is None:
            return value
        return pd.DataFrame(value, index=index, columns=names)

    def partial_effect(self, term, x, return_std=False):
        """
        The fitted function of a term: `term` is its index or its name (e.g.
        `'s(age, df=5)'`), and `x` its values, or a DataFrame of the data
        (with the term's columns).

        With `return_std=True`, also its pointwise standard errors from the
        posterior covariance `posterior_covariance_` (as mgcv's
        `predict(type="terms", se.fit=TRUE)`; the curve of a centered term is
        centered, so they are those of the centered curve): returns `(fit,
        std)`.
        """
        if not isinstance(term, (int, np.integer)):
            term = [t.name for t in self.terms_].index(term)
        j, term = term, self.terms_[term]
        if isinstance(x, pd.DataFrame):
            x = self._columns(self._design(x), [term])[0]
        fit = term.predict(x)
        if not return_std:
            return fit
        self._check_covariance()
        B = self.design_.block(j, x)
        sl = self.design_.slices[j]
        V = self.posterior_covariance_[sl, sl]
        return fit, np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', B, V, B), 0))

    def _check_covariance(self):
        if self.posterior_covariance_ is None:
            raise ValueError(self._covariance_reason)

    def _linear_predictor(self, X, offset=None):
        X = self._design(X)
        eta = self.intercept_ + sum(term.predict(x) for term, x in zip(self.terms_,
                                                                       self._columns(X, self.terms_)))
        if offset is not None:
            eta = eta + offset
        return np.asarray(eta, float)

    def predict(self, X, prediction_type='response', offset=None, return_std=False):
        """
        Predictions at `X`: of the mean (`'response'`) or of the linear
        predictor (`'link'`). With `return_std=True`, also their standard
        errors from the posterior covariance `posterior_covariance_` (as
        mgcv's `predict(se.fit=TRUE)`; for the mean, by the delta method):
        returns `(prediction, std)`.
        """
        if prediction_type not in ('link', 'response'):
            raise ValueError("prediction_type should be one of 'response' or 'link'")
        eta = self._linear_predictor(X, offset)
        if return_std:
            self._check_covariance()
            rows = self.design_.rows(self._columns(self._design(X), self.terms_))
            se_eta = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', rows,
                                                  self.posterior_covariance_, rows), 0))
        if prediction_type == 'link':
            return (eta, se_eta) if return_std else eta
        if self._cox:
            # the relative risk
            mu = dmu = np.exp(eta)
        else:
            mu = self._family.base.link.inverse(eta)
            dmu = self._family.base.link.inverse_deriv(eta) if return_std else None
        if return_std:
            return mu, se_eta * np.abs(dmu)
        return mu

    def _cox_data(self, y, nobs):
        """
        The Cox family of the event data in `y` (a DataFrame with the
        columns of `family`: event times, status, and optionally start times
        and strata), and the offset and weights (`offset_id`, `weight_id`).
        """
        if not isinstance(y, pd.DataFrame):
            raise ValueError('for the Cox family, y should be a DataFrame with the columns '
                             'of the event times and status (and the start times, strata, '
                             'offset and weights, if any)')
        if y.shape[0] != nobs:
            raise ValueError('X and y should have the same number of rows')
        fam = self.family
        family = CoxFamilySpec(event_data=y,
                               tie_breaking=fam.tie_breaking,
                               event_id=fam.event_id,
                               status_id=fam.status_id,
                               start_id=fam.start_id,
                               strata_id=fam.strata_id)
        offset = (None if self.offset_id is None
                  else np.asarray(y[self.offset_id], float).reshape(-1))
        weight = (np.ones(nobs) if self.weight_id is None
                  else np.asarray(y[self.weight_id], float).reshape(-1))
        return family, None, offset, weight

    def plot(self, terms=None, axes=None, **kwargs):
        """
        Plot the fits of the terms (by default every term with a curve), one
        panel each, with standard error bands and rugs, as R's `plot.Gam`:
        see `glmnet.gam.plot.plot_terms` and `plot_term`. Returns the axes.
        """
        return plot_terms(self, terms=terms, axes=axes, **kwargs)

    def model_latex(self):
        """
        The model this GAM specifies, in LaTeX (an `aligned` block): the
        response's distribution and link, the linear predictor and a line
        per term (`glmnet.gam.display`). Jupyter shows it for a `GAM`.
        """
        return model_latex(self)

    def _repr_mimebundle_(self, **kwargs):
        # the model in LaTeX, and the text (not scikit-learn's HTML diagram,
        # which a notebook would show instead)
        return {'text/plain': repr(self), 'text/latex': '$$' + self.model_latex() + '$$'}

    def summary(self):
        """
        The terms, their degrees of freedom (`edf`, as R's `gam`, and
        `edf_influence`, as mgcv's, when every term has a basis) and smoothing
        parameters.
        """
        lam = [getattr(term, 'lam_', np.nan) for term in self.terms_]
        table = pd.DataFrame({'edf': self.edf_.values}, index=self.edf_.index)
        # (mgcv's edf form the dense covariance: not for a method for large
        # data, which says it does without them)
        method = getattr(self, '_method', None)
        if (method is None or not method.estimates or method.covariance_edf) and \
           self.edf_influence_ is not None:
            table['edf_influence'] = self.edf_influence_.values
        table['lam'] = lam
        return table

    @property
    def fixed_effects_(self):
        """
        The intercept and the coefficients of the linear terms (`linear()`,
        and blocks of columns), with their standard errors from the posterior
        covariance (for the Gaussian, the generalized least squares standard
        errors, lme4's `vcov`). With random effects (and blocks of penalized
        columns) as the only penalized terms, they come from backfitting
        those terms on the linear terms' columns, never forming the dense
        design: the linear coefficients' block of the inverse of `H` is that
        of `X_F' W (I - S) X_F`, `S` the random effects' smoother (their joint
        penalized fit). Otherwise from `posterior_covariance_`.
        """
        cov = self.fixed_effects_covariance_
        estimates = [float(self.intercept_)]
        for term in self.terms_:
            if isinstance(term, SlopeTerm):
                estimates.append(float(np.atleast_1d(term.coef_)[0]))
            elif isinstance(term, ColumnsTerm):
                estimates.extend(np.atleast_1d(term.coef_).astype(float))
        return pd.DataFrame({'estimate': estimates, 'std_error': np.sqrt(np.diag(cov))},
                            index=cov.index)

    @property
    def fixed_effects_covariance_(self):
        """The covariance of `fixed_effects_` (see there)."""
        if getattr(self, '_fixed_covariance', None) is None:
            self._fixed_covariance = self._compute_fixed_covariance()
        return self._fixed_covariance

    def _fixed_terms(self):
        """The linear terms' columns (centered) and names, and the other terms."""
        if self._cox:
            raise NotImplementedError('fixed_effects_ of a Cox GAM: no intercept')
        columns, names, others = [np.ones(self.state_.fits.shape[0])], ['intercept'], []
        for j, term in enumerate(self.terms_):
            if isinstance(term, SlopeTerm):
                columns.append(term.linear_columns()[:, 0])
                names.append(term.name)
            elif isinstance(term, ColumnsTerm):
                X = term.linear_columns()
                columns.extend(X.T)
                labels = term.column if isinstance(term.column, list) else [term.column]
                names.extend(f'{term.name}[{c}]' for c in labels)
            elif not term.penalized:
                raise NotImplementedError(f'fixed_effects_: {term.name} (unpenalized, not a '
                                          'linear term) is not supported; give its columns '
                                          'as a block of columns')
            else:
                others.append(j)
        return np.column_stack(columns), names, others

    def _compute_fixed_covariance(self):
        F, names, others = self._fixed_terms()
        w = self._covariance_weights
        terms = [self.terms_[j] for j in others]
        if all(not term.centered for term in terms):
            # backfit the random effects on each column: (I - S) F
            for term in terms:
                term.set_weights(w)
            resid = np.empty_like(F)
            for c in range(F.shape[1]):
                v = F[:, c]
                fits = np.zeros((F.shape[0], len(terms)))
                for sweep in range(self.control.bf_maxit):
                    old = fits.copy()
                    total = fits.sum(1)
                    for k, term in enumerate(terms):
                        f = term.smooth(v - (total - fits[:, k]))
                        total += f - fits[:, k]
                        fits[:, k] = f
                    change = np.linalg.norm(fits - old) / (0.1 + np.linalg.norm(fits))
                    if change < self.control.bf_tol:
                        break
                resid[:, c] = v - fits.sum(1)
            M = F.T @ (w[:, None] * resid)
            cov = self.dispersion_ * np.linalg.inv((M + M.T) / 2) / self.weight_total_
        else:
            # a smooth among the penalized terms: the block of the posterior
            # covariance (the dense design)
            self._check_covariance()
            D = self.design_
            idx = [0]
            for j, term in enumerate(self.terms_):
                if isinstance(term, (SlopeTerm, ColumnsTerm)):
                    idx.extend(range(D.slices[j].start, D.slices[j].stop))
            cov = self.posterior_covariance_[np.ix_(idx, idx)]
        return pd.DataFrame(cov, index=names, columns=names)

    # as for GLM: they use only the response, weights, family and state_
    _on_train_index = GLM._on_train_index

    def _fixed_dispersion(self):
        # (a quasi family's dispersion is free: `glmnet.gam.families`)
        if getattr(getattr(self._family, 'base', None), 'quasi', False):
            return False
        return GLM._fixed_dispersion(self)

    def _residuals(self, kind='response', state=None):
        if getattr(self, '_cox', False):
            raise NotImplementedError('residuals of a Cox GAM are not available')
        return GLM._residuals(self, kind=kind, state=state)

    def _loglike(self):
        if self._cox:
            # the (log) partial likelihood, as R's coxph: the saturated one
            # less half the deviance
            result = self._family._coxdev_at(self._weight)(self.state_.link_parameter,
                                                                  self._weight)
            return float(result.loglik_sat - result.deviance / 2)
        base = self._family.base
        if getattr(base, 'quasi', False):
            # (a quasi-likelihood: none, as R's quasipoisson()$aic)
            return np.nan
        if isinstance(base, (sm_family.Gamma, sm_family.InverseGaussian)):
            # R's Gamma()$aic and inverse.gaussian()$aic: at the dispersion
            # deviance / sum(weights), not Pearson's
            # (over the observations of positive weight, as GLM's)
            y = np.asarray(self._response, float).reshape(-1)
            keep = self._weight > 0
            return base.loglike(y[keep], np.asarray(self.state_.mu)[keep],
                                var_weights=self._weight[keep],
                                scale=self.deviance_ / self._weight.sum())
        return GLM._loglike(self)
