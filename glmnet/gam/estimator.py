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

from modelmatrix import Feature

from glmnet._utils import _extract_y, _binomial_counts, _parent_dataclass_from_child
from glmnet._roles import _role_y, cox_ids, OFFSET, WEIGHT
from glmnet.family import GLMFamilySpec
from glmnet.cox import CoxFamily, CoxFamilySpec
from glmnet.glm import GLM, GLMControl
from glmnet.irls import IRLS
from .backfit import GAMRegularizer, GAMState
from .display import model_latex
from .fixed import expand_terms, augment, feature_column
from .joint import JointDesign, JointRegularizer
from .plot import plot_terms
from .reml import ExperimentalWarning, _blocked_term
from .selection import resolve_method, SmoothingMethod, Fixed
from .terms import SplineSmoother, Smoother, LinearSmoother, ColumnsSmoother, CategoricalSmoother
from .._frames import as_pandas


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
    `GAMControl(solver='joint')`, jointly on the terms' bases, as mgcv
    (Wood, 2011, 2017). See the References of `glmnet.gam`.

    A GAM's smoothing is fixed: each term's `lam` or `df`. To estimate the
    smoothing parameters (and random effects' variances), by REML, ML or a
    method of one's own, fit a `GLMM`, which has the same terms.

    The model is a list of terms (`glmnet.gam.s`, `lo`, `random_effect`,
    `gaussian_process`, `linear`, `categorical`, or a `ColumnsSmoother` of
    several columns fitted as one parametric block), each naming a column
    of `X` (ISLP's `ISLP.gam` builds them from a ModelSpec). The smoothing
    of each term is fixed: by its smoothing parameter `lam`, or by its
    degrees of freedom `df` (as R's `gam`). When every term has a fixed
    smoothing parameter (`s(lam=...)`, `random_effect(lam=...)`, `linear`,
    `categorical`), the fit minimizes

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
        Column identifier in `y`. (Optional; deprecated: name the columns of `y` by role, `offset`, `weight`.)
    weight_id: Union[str,int]
        Weight identifier in `y`. (Optional; deprecated: name the columns of `y` by role, `offset`, `weight`.)
    response_id: Union[str,int]
        Response identifier in `y`. (Optional; deprecated: name the columns of `y` by role, `offset`, `weight`.)
    method: str, optional
        Deprecated. `'fixed'`, the only smoothing of a GAM, is the default;
        `'REML'`, `'ML'` or a `SmoothingMethod` are a `GLMM`'s
        (`GLMM(method=...)`). Either raises a FutureWarning.

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
    working_weights_, penalty_matrix_: np.ndarray
        `W` and `S` above (see there). The sandwich covariance,
        `sandwich_covariance()`, has the same bread `X' W X + S`, and the
        meat from `score_contributions()` (HC0, clustered, or one's own).
    design_: JointDesign
        The terms' bases side by side (None if a term has none).

    These four are computed when first used (by `predict(...,
    return_std=True)`, `partial_effect`, `summary`).
    df_resid_: float
        Residual degrees of freedom, the number of observations (of positive
        weight) less 1 less the sum of `edf_` (as R's `gam`).
    deviance_, null_deviance_, dispersion_, loglike_, aic_, bic_: float
        As for `GLM`; the AIC counts `1 + sum(edf_)` parameters, plus 1 for a
        free dispersion, as R's `gam` (for the Gamma and inverse Gaussian the
        log-likelihood is at the dispersion deviance / sum(weights), as R's),
        plus 1 for an estimated negative binomial theta. A quasi family
        (`glmnet.gam.families`) has no log-likelihood or AIC (NaN).
    negative_binomial_theta_: float
        The negative binomial's theta, given (or estimated by a `GLMM`); None
        for other families.
    fixed_effects_: pd.DataFrame
        The intercept and the linear terms' coefficients, with standard
        errors (see the property).
    n_iter_: int
        Number of IRLS iterations.
    converged_: bool
        Whether local scoring converged.
    """
    terms: list = field(default_factory=list)
    family: object = field(default_factory=GLMFamilySpec)
    control: GAMControl = field(default_factory=GAMControl)
    offset_id: Union[str, int] = None
    weight_id: Union[str, int] = None
    response_id: Union[str, int] = None
    method: Optional[str] = None

    fit_intercept = True

    # hooks for a subclass that builds the terms and the columns they name
    # from a model specification (ISLP's GAM, from a ModelSpec)

    def _model_terms(self, X):
        """
        The terms of the model and the data whose columns they name: the
        fixed effects (items of `terms` other than smoothers, see
        `glmnet.gam.fixed`) expanded, a block of columns for each, their
        ModelSpecs fit (`model_specs_`).
        """
        terms, self.model_specs_, X = expand_terms(self.terms, X)
        return terms, X

    def _design(self, X):
        """The data whose columns the fitted terms name (`X`, with the fixed effects')."""
        return augment(X, getattr(self, 'model_specs_', []))

    @staticmethod
    def _columns(X, terms):
        """The columns of `X` of the terms."""
        cols = []
        for term in terms:
            if isinstance(X, pd.DataFrame):
                # a name or a list of names (a ColumnsSmoother), else positions;
                # a feature is the column `_model_terms` built for it
                column = term.columns
                if isinstance(column, list):
                    column = [feature_column(c) if isinstance(c, Feature) else c for c in column]
                elif isinstance(column, Feature):
                    column = feature_column(column)
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
        X, y = as_pandas(X), as_pandas(y)
        y = _role_y(self, y)
        # (the roles of its columns, as the LaTeX shows them)
        self.y_roles_ = tuple(c for c in (OFFSET, WEIGHT)
                              if isinstance(y, pd.DataFrame) and c in y.columns)
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
            response, offset, weight = _extract_y(y)
            if family.is_binomial:   # (successes, failures), as R's cbind
                response, weight = _binomial_counts(response, weight)
            if np.ndim(response) == 2 and response.shape[1] > 1:
                raise ValueError(f'a GAM has a response of one column, y gives '
                                 f'{response.shape[1]}: the columns of y other than offset '
                                 'and weight')
            response = np.asarray(response, float).reshape(-1)
            weight = np.asarray(weight, float).reshape(-1)
        self._response, self._weight = response, weight
        weight_total = weight.sum()
        normed_weight = self._normed_weight = weight / weight_total
        self.weight_total_ = weight_total

        method = self._method = self._smoothing_method()
        if getattr(self.family, 'estimate_theta', False) and not method.estimates:
            raise ValueError('the negative binomial\'s theta is estimated by a GLMM '
                             '(method="REML" or "ML"): give theta (NegativeBinomial(theta=...)) '
                             'to fix it')
        self.terms_ = self._resolve_terms([deepcopy(term) for term in terms])
        for term, x in zip(self.terms_, self._columns(X, terms)):
            categorical_column = (term.column if isinstance(term, CategoricalSmoother)
                                  else getattr(term, 'categorical_column', None))
            if categorical_column is not None and isinstance(X, pd.DataFrame):
                # a categorical's order of the levels (not the sorted one)
                column = (X[categorical_column] if isinstance(categorical_column, str)
                          else X.iloc[:, categorical_column])
                if isinstance(column.dtype, pd.CategoricalDtype):
                    term.categories = list(column.cat.categories)
            term.setup(x)
            if not hasattr(term, 'x_'):
                # (the training column, to plot: a smoother of one's own need not keep it)
                term.x_ = np.asarray(x)
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
                warnings.warn(f"{type(self).__name__}(method='{method.name}') is experimental: the selection of the "
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

    # scikit-learn's parameters: a GAM's own, its terms (`terms__<j>`, a
    # term replaced whole, as a Pipeline's steps) and their arguments
    # (`terms__<j>__<param>`), so that the smoother of a column, its degrees
    # of freedom, smoothing parameter or basis can be chosen by
    # cross-validation (GridSearchCV); `score` is the held-out fit

    def _term_params(self, term):
        """The parameters of a term to show (a GLMM leaves out those it estimates)."""
        return term.get_params(deep=True)

    def get_params(self, deep=True):
        """
        The GAM's parameters (scikit-learn's); with `deep`, also each term
        (`terms__<j>`) and its arguments (`terms__<j>__<param>`).
        """
        params = super().get_params(deep=deep)
        if deep:
            for j, term in enumerate(self.terms or []):
                params[f'terms__{j}'] = term
                if hasattr(term, 'get_params'):
                    for key, value in self._term_params(term).items():
                        params[f'terms__{j}__{key}'] = value
        return params

    def set_params(self, **params):
        """
        Set the parameters (scikit-learn's): `terms__<j>` replaces term `j`,
        `terms__<j>__<param>` sets its argument (on a copy). Returns `self`.
        """
        whole, nested = {}, {}
        for key in [k for k in params if k.startswith('terms__')]:
            parts = key.split('__', 2)
            j = int(parts[1])
            if len(parts) == 2:
                whole[j] = params.pop(key)
            else:
                nested.setdefault(j, {})[parts[2]] = params.pop(key)
        super().set_params(**params)
        if whole or nested:
            # (copies: the terms given may be used in other models)
            terms = list(self.terms)
            for j, term in whole.items():
                terms[j] = term
            for j, term_params in nested.items():
                terms[j] = deepcopy(terms[j]).set_params(**term_params)
            self.terms = terms
        return self

    def score(self, X, y, sample_weight=None):
        """
        Minus half the deviance of the fit on `(X, y)` (with its offset and
        weights, `offset_id` and `weight_id`, unless `sample_weight` is
        given): a sum over the observations, as `GLM.score`, the larger the
        better, for scikit-learn's cross-validation. For the Cox family, the
        partial log-likelihood of the rows of `y`, the risk sets among them.

        The folds should suit the model: a random effect predicts 0 for
        groups not in the training folds (with `GroupKFold` the score is of
        prediction for new groups, with `KFold` for groups seen), and
        serially correlated rows need blocked folds.
        """
        X, y = as_pandas(X), as_pandas(y)
        y = _role_y(self, y)
        nobs = X.shape[0]
        if self._cox:
            family, _, offset, weight = self._cox_data(y, nobs)
            eta = self._linear_predictor(X, offset)
            w = weight if sample_weight is None else np.asarray(sample_weight, float)
            return -float(family.deviance(None, eta, w)) / 2
        response, offset, weight = _extract_y(y)
        if self._family.is_binomial:   # (successes, failures), as R's cbind
            response, weight = _binomial_counts(response, weight)
        response = np.asarray(response, float).reshape(-1)
        w = (np.asarray(weight, float).reshape(-1) if sample_weight is None
             else np.asarray(sample_weight, float))
        mu = self.predict(X, offset=offset)
        return -float(self._family.deviance(response, mu, w)) / 2

    def _smoothing_method(self):
        """
        The `SmoothingMethod`: `Fixed()` for a GAM. `GAM(method=...)` is
        deprecated: `'fixed'` is a GAM's only smoothing, and the others are
        a `GLMM`'s.
        """
        method = self.method
        if method is None:
            return Fixed()
        method = resolve_method(method)
        if isinstance(method, Fixed):
            warnings.warn("GAM(method='fixed') is deprecated: a GAM's smoothing is always "
                          "fixed (each term's lam or df), so drop method",
                          FutureWarning, stacklevel=3)
        else:
            warnings.warn(f"GAM(method={self.method!r}) is deprecated: smoothing parameters "
                          f"are estimated by a GLMM, GLMM(method={self.method!r})",
                          FutureWarning, stacklevel=3)
        return method

    def _resolve_terms(self, terms):
        """
        The smoothing of the terms for the method: `s()` with neither `df`
        nor `lam` has `df=4` in a GAM (as R's `gam`), and its smoothing
        parameter estimated by a GLMM, as `random_effect()` without them. In a GAM a
        term's `lam` list has no None (one to estimate).
        """
        method = self._method
        for term in terms:
            if (isinstance(term, SplineSmoother) and not method.estimates and
                term.df is None and term.lam is None):
                term.df = 4.
            lam = getattr(term, 'lam', None)
            if (not method.estimates and lam is not None and
                any(v is None or np.isnan(float(v))
                    for v in np.atleast_1d(np.asarray(lam, dtype=object)))):
                raise ValueError(f'{term.name}: a GAM fixes every smoothing parameter: give each '
                                 'lam, or estimate those that are None with a GLMM')
        return terms

    def inner_fit(self, state=None, control=None):
        """
        For a `SmoothingMethod`: the fit (local scoring) at the terms' current
        parameters, from `state` (by default the intercept-only fit), as a
        `GAMState` (its fits, intercept and means; whether it converged in
        `converged_`), with `control` (default the GAM's). The terms'
        parameters are set by the method (e.g. `term.set_penalty_lams`).
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
        cov = dict(design=None, posterior=None, frequentist=None, edf=None,
                   bread_inv=None, penalty=None)
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
            # the bread of the sandwich (`sandwich_covariance`)
            cov['bread_inv'], cov['penalty'] = Ainv, A - XWX
            F = np.einsum('ij,ji->i', Ainv, XWX)
            cov['edf'] = pd.Series([F[sl].sum() for sl in D.slices],
                                   index=[term.name for term in self.terms_],
                                   name='edf_influence')
        self._covariance = cov
        return cov

    @property
    def design_(self):
        """The `JointDesign` of the terms' bases (None if a term has none)."""
        return self._get_covariance()['design']

    @property
    def posterior_covariance_(self):
        """The posterior covariance of the joint design's coefficients (mgcv's `Vp`)."""
        return self._get_covariance()['posterior']

    @property
    def frequentist_covariance_(self):
        """The frequentist covariance of the joint design's coefficients (mgcv's `Ve`)."""
        return self._get_covariance()['frequentist']

    @property
    def edf_influence_(self):
        """Each term's degrees of freedom in the joint fit (mgcv's `edf`)."""
        return self._get_covariance()['edf']

    @property
    def working_weights_(self):
        """
        `W`, the weights of the final IRLS working problem at the training
        rows: `v_i mu'(eta_i)^2 / V(mu_i)`, with `v_i = w_i / sum(w)` the
        prior weights normalized to sum to 1, so that `X' W X` is the
        Hessian of `deviance / (2 sum(w))` (the objective's scale, without
        the dispersion). For the Cox family, the diagonal of the partial
        likelihood's information (its covariances use the full information).
        """
        return self._covariance_weights

    @property
    def penalty_matrix_(self):
        """
        `S_lambda`, the penalty on the coefficients of `design_`: `sum_j
        sum_m lam_jm S_jm`, each term's penalties with its constraint
        absorbed (0 for the intercept and the unpenalized terms), so that
        `beta' S_lambda beta / 2` is the objective's penalty. None when
        `design_` is.
        """
        return self._get_covariance()['penalty']

    def score_contributions(self):
        """
        The score of the penalized objective, observation by observation: an
        `n x P` array whose row `i` is

            s_i = W_i (z_i - eta_i) x_i = v_i (y_i - mu_i) mu'(eta_i) / V(mu_i) x_i,

        the negative gradient in `beta` of observation `i`'s part of
        `deviance / (2 sum(w))`. `x_i` is row `i` of `design_.X` (the column
        of ones, then each term's basis with its sum-to-zero constraint
        absorbed, or centered, as `JointDesign`): the coordinates of
        `posterior_covariance_`. `W_i` is `working_weights_[i]`, `z_i` the
        working response and `v_i = w_i / sum(w)` the normalized prior
        weight. The rows are the training rows (0 where the weight is 0).

        At the penalized fit `sum_i s_i = S_lambda beta` (`penalty_matrix_`,
        `beta = design_.gamma(state_)`; plus `X' linear_term` if `fit` had
        one). The empirical meat is `sum_i s_i s_i'`; its model-based value
        is `dispersion_ X' W X / sum(w)` (`s_i` has no dispersion in it, and
        is per `sum(w)`).

        Not for the Cox family: its risk sets couple the rows
        (NotImplementedError).
        """
        if getattr(self, '_cox', False):
            raise NotImplementedError('score_contributions of a Cox GAM: the risk sets couple '
                                      'the rows, so there are no per-observation scores; '
                                      'give sandwich_covariance a meat of your own')
        self._check_covariance()
        z, w = self._family.get_response_and_weights(self.state_, self._response,
                                                     self._offset, self._normed_weight)
        return (w * (z - self.state_.eta))[:, None] * self.design_.X

    def sandwich_covariance(self, meat=None, groups=None):
        """
        The sandwich (robust) covariance `A^{-1} M A^{-1}` of the
        coefficients of `design_`, on the scale of `posterior_covariance_`
        and `frequentist_covariance_`.

        - The bread `A = X' W X + S_lambda` is the Hessian of the penalized
          objective `deviance / (2 sum(w)) + beta' S_lambda beta / 2` at the
          fit: `X = design_.X` (the column of ones, then each term's basis
          with its constraint absorbed, as `JointDesign`), `W =
          working_weights_` (the final IRLS weights, the prior weights
          normalized to sum to 1), `S_lambda = penalty_matrix_` (at the
          fitted smoothing parameters). `posterior_covariance_` is
          `dispersion_ A^{-1} / sum(w)`.
        - The meat `M` is the covariance of the objective's score `sum_i
          s_i` (`score_contributions`, which gives the scale). The
          model-based meat `dispersion_ X' W X / sum(w)` gives
          `frequentist_covariance_` (mgcv's `Ve`).

        Parameters
        ----------
        meat: np.ndarray, optional
            A `P x P` meat of one's own, on the scale of
            `score_contributions()`: e.g. the covariance, over a grouped
            (cluster) bootstrap, of the sum of the score contributions, or
            one from block sums for serial correlation. For the Cox family
            (no score contributions) the only choice.
        groups: array-like, optional
            A label per training row: the cluster-robust meat `sum_g (sum_{i
            in g} s_i)(sum_{i in g} s_i)'`, without a small-sample correction.

        By default the meat is HC0's, `sum_i s_i s_i'`. The dispersion does
        not enter. With penalties the sandwich is the variance around the
        penalized (biased) fit: it does not account for the bias.
        """
        if meat is not None and groups is not None:
            raise ValueError('give meat or groups, not both')
        self._check_covariance()
        Ainv = self._get_covariance()['bread_inv']
        P = Ainv.shape[0]
        if meat is None:
            s = self.score_contributions()
            if groups is not None:
                groups = np.asarray(groups).reshape(-1)
                if groups.shape[0] != s.shape[0]:
                    raise ValueError(f'groups should have a label per training row ({s.shape[0]})')
                codes, uniques = pd.factorize(groups)
                if np.any(codes < 0):
                    raise ValueError('groups has missing labels')
                sums = np.zeros((len(uniques), P))
                np.add.at(sums, codes, s)
                s = sums
            meat = s.T @ s
        else:
            meat = np.asarray(meat, float)
            if meat.shape != (P, P):
                raise ValueError(f'meat should be {P} x {P}, on the coefficients of design_')
        V = Ainv @ meat @ Ainv
        return (V + V.T) / 2

    def _coef_covariance(self, covariance):
        """The covariance of `design_`'s coefficients that `covariance` names or gives."""
        self._check_covariance()
        if isinstance(covariance, str):
            if covariance == 'posterior':
                return self.posterior_covariance_
            if covariance == 'frequentist':
                return self.frequentist_covariance_
            if covariance == 'sandwich':
                return self.sandwich_covariance()
            raise ValueError("covariance should be 'posterior', 'frequentist', 'sandwich' "
                             "or a P x P array")
        V = np.asarray(covariance, float)
        P = self.posterior_covariance_.shape[0]
        if V.shape != (P, P):
            raise ValueError(f'covariance should be {P} x {P}, on the coefficients of design_')
        return V

    def _on_train_index_frame(self, value, names):
        index = getattr(self, '_train_index', None)
        if index is None:
            return value
        return pd.DataFrame(value, index=index, columns=names)

    def partial_effect(self, term, x, return_std=False, covariance='posterior'):
        """
        The fitted function of a term: `term` is its index or its name (e.g.
        `'s(age, df=5)'`), and `x` its values, or a DataFrame of the data
        (with the term's columns).

        With `return_std=True`, also its pointwise standard errors from the
        posterior covariance `posterior_covariance_` (as mgcv's
        `predict(type="terms", se.fit=TRUE)`; the curve of a centered term is
        centered, so they are those of the centered curve): returns `(fit,
        std)`. `covariance` chooses the coefficients' covariance:
        `'posterior'` (the default), `'frequentist'`
        (`frequentist_covariance_`), `'sandwich'` (HC0's
        `sandwich_covariance()`) or a `P x P` array, such as
        `sandwich_covariance(groups=...)`.
        """
        x = as_pandas(x)
        if not isinstance(term, (int, np.integer)):
            term = [t.name for t in self.terms_].index(term)
        j, term = term, self.terms_[term]
        if isinstance(x, pd.DataFrame):
            x = self._columns(self._design(x), [term])[0]
        fit = term.predict(x)
        if not return_std:
            return fit
        V = self._coef_covariance(covariance)
        B = self.design_.block(j, x)
        sl = self.design_.slices[j]
        V = V[sl, sl]
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

    def predict(self, X, prediction_type='response', offset=None, return_std=False,
                covariance='posterior'):
        """
        Predictions at `X`: of the mean (`'response'`) or of the linear
        predictor (`'link'`). With `return_std=True`, also their standard
        errors from the posterior covariance `posterior_covariance_` (as
        mgcv's `predict(se.fit=TRUE)`; for the mean, by the delta method):
        returns `(prediction, std)`. `offset` is added to the linear
        predictor. `covariance` chooses another covariance of the
        coefficients, as for `partial_effect`: `'frequentist'`,
        `'sandwich'` or a `P x P` array (`sandwich_covariance(groups=...)`).
        For the Cox family `'response'` is the relative risk.
        """
        X = as_pandas(X)
        if prediction_type not in ('link', 'response'):
            raise ValueError("prediction_type should be one of 'response' or 'link'")
        eta = self._linear_predictor(X, offset)
        if return_std:
            V = self._coef_covariance(covariance)
            rows = self.design_.rows(self._columns(self._design(X), self.terms_))
            se_eta = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', rows, V, rows), 0))
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
        columns `stop`, `status`, and optionally `start` and `strata`), and
        the offset and weights (`offset`, `weight`).
        """
        y = _role_y(self, y)
        if not isinstance(y, pd.DataFrame):
            raise ValueError('for the Cox family, y should be a DataFrame with the columns '
                             'of the event times and status (and the start times, strata, '
                             'offset and weights, if any)')
        if y.shape[0] != nobs:
            raise ValueError('X and y should have the same number of rows')
        family = CoxFamilySpec(event_data=y,
                               tie_breaking=self.family.tie_breaking,
                               **cox_ids(y))
        _, offset, weight = _extract_y(y)
        if offset is not None:
            offset = np.asarray(offset, float).reshape(-1)
        weight = np.asarray(weight, float).reshape(-1)
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
            if isinstance(term, LinearSmoother):
                estimates.append(float(np.atleast_1d(term.coef_)[0]))
            elif isinstance(term, ColumnsSmoother):
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
            if isinstance(term, LinearSmoother):
                columns.append(term.linear_columns()[:, 0])
                names.append(term.name)
            elif isinstance(term, ColumnsSmoother):
                X = term.linear_columns()
                columns.extend(X.T)
                if getattr(term, 'coef_names', None) is not None:   # a fixed effect's
                    names.extend(term.coef_names)
                else:
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
                if isinstance(term, (LinearSmoother, ColumnsSmoother)):
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


@dataclass
class GLMM(GAM):
    """
    Generalized linear mixed model: a `GAM` whose penalized terms are random
    effects, their smoothing parameters (the variances of the effects, over
    the dispersion) estimated. Smooths are random effects here too, as in
    mgcv: `s(x)` is a random effect on its basis' wiggly part, of variance
    `dispersion / lam`. The terms, family, control and attributes are a
    GAM's; a GLMM adds the estimated smoothing (`smoothing_`,
    `variance_components_`, `reml_score_`).

    `method` chooses the estimates: `'REML'` (the default) or `'ML'`, the
    restricted (or full) marginal likelihood by Laplace's method, as mgcv's
    `gam(method=...)` and lme4's `glmer` (with JAX, `pip install
    glmstar[gam]`; not needed for the Cox family); or a
    `glmnet.gam.selection.SmoothingMethod` of one's own (the documentation's
    page "Crossed random effects at scale" writes one, the method of moments
    for the variances of crossed random effects in large data).

    Some parameters can be held fixed while the others are estimated:

    - a term given `lam` (or `df`, matched at the prior weights, then fixed,
      as mgcv's `sp`) keeps it: `GLMM([s('x', lam=1e-3), random_effect('g')])`
      estimates only the random effect's variance;
    - a term with several penalties (`tensor_spline`, `penalized_columns`,
      a categorical `varying` smooth) takes a list with `None` for those to
      estimate: `tensor_spline('x', 'z', lam=[1e-2, None])` fixes the first
      margin's and estimates the second's;
    - correlated random effects given `precision` keep it.

    `smoothing_.terms` names the estimated parameters.

    Parameters
    ----------
    terms, family, control, offset_id, weight_id, response_id:
        As for `GAM`.
    method: str or SmoothingMethod
        `'REML'` (the default), `'ML'` or a `SmoothingMethod`. With `'REML'`
        or `'ML'` the working problems are solved jointly (or, for one large
        random effect, by backfitting with the criterion factored by blocks:
        `GAMControl(reml_backend=...)`), and `edf_`, `df_resid_` and
        `dispersion_` follow mgcv (its edf, its residual df, the REML or ML
        estimate of the dispersion). **Experimental**: `'REML'` and `'ML'`
        warn (`glmnet.gam.ExperimentalWarning`); check
        `smoothing_.converged`. (`'fixed'` is a `GAM`: deprecated here, with
        a FutureWarning.)

    Attributes
    ----------
    The attributes of a `GAM`, and:

    smoothing_: SmoothingSelection
        The estimated smoothing parameters, the dispersion and the criterion
        (`glmnet.gam.reml`).
    reml_score_: float
        The minimized criterion (the negative log REML or ML; as mgcv's
        `gcv.ubre`, up to a constant for REML, which depends on the
        parametrization of the unpenalized coefficients).
    variance_components_: pd.DataFrame
        The variances (and standard deviations) of the penalized terms'
        coefficients as random effects, the dispersion over their smoothing
        parameters (on the prior weights' scale), and the dispersion
        (`scale`), as mgcv's `gam.vcomp` and lme4's `VarCorr` for random
        effects.
    negative_binomial_theta_: float
        The negative binomial's theta: estimated with
        `NegativeBinomial(theta=None)`, as mgcv's `nb()`.
    """
    method: Union[str, SmoothingMethod] = 'REML'

    # the parameters the method estimates (None: to estimate) are not
    # scikit-learn's parameters to tune; those given are
    _estimated_params = ('lam', 'df', 'precision')

    def _term_params(self, term):
        return {key: value for key, value in term.get_params(deep=True).items()
                if not (key.rsplit('__', 1)[-1] in self._estimated_params and value is None)}

    def _smoothing_method(self):
        method = resolve_method('REML' if self.method is None else self.method)
        if isinstance(method, Fixed):
            warnings.warn("GLMM(method='fixed') is deprecated: fixed smoothing is a GAM, "
                          "GAM(terms)", FutureWarning, stacklevel=3)
        return method
