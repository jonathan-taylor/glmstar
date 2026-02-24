"""
This module contains the core code needed for post selection
after LASSO.

"""

import warnings
from warnings import warn
from copy import copy

from dataclasses import dataclass, asdict
from typing import Optional, Callable

import numpy as np
import scipy.sparse
import pandas as pd
from scipy.stats import norm as normal_dbn
from sklearn.base import clone

import mpmath as mp
mp.dps = 80

from .base import _get_design
from .glmnet import GLMNet, GLMState
from .glm import compute_grad
from ._bootstrap import parametric_GLM

@dataclass
class TruncatedGaussian(object):
    """
    Represents a truncated Gaussian distribution for post-selection inference.

    Attributes
    ----------
    estimate : float
        The observed value or estimate.
    sigma : float
        Standard deviation of the underlying Gaussian.
    smoothing_sigma : float
        Standard deviation for the smoothing noise.
    lower_bound : float
        Lower truncation bound.
    upper_bound : float
        Upper truncation bound.
    noisy_estimate : float
        Noisy version of the estimate (for randomized selection).
    factor : float
        Scaling factor for the distribution (default 1).
    """
    
    estimate: float
    sigma: float
    smoothing_sigma: float
    lower_bound: float
    upper_bound: float
    noisy_estimate: float
    factor: float = 1.

    def weight(self,
               x):
        t = x * self.factor
        return (normal_dbn.cdf((self.upper_bound - t) / self.smoothing_sigma)
                - normal_dbn.cdf((self.lower_bound - t) / self.smoothing_sigma))
    
@dataclass
class WeightedGaussianFamily(object):
    """
    Represents a family of weighted Gaussian distributions for selective inference.

    Attributes
    ----------
    estimate : float
        The observed value or estimate.
    sigma : float
        Standard deviation of the underlying Gaussian.
    weight_fns : list of callables
        List of weight functions to apply to the distribution.
    num_sd : float, optional
        Number of standard deviations to use for grid (default 10).
    num_grid : int, optional
        Number of grid points (default 4000).
    use_sample : bool, optional
        Whether to use a sample-based approximation (default False).
    seed : int, optional
        Random seed for reproducibility (default 0).
    """

    estimate: float
    sigma: float
    weight_fns: list
    num_sd: Optional[float] = 10
    num_grid: Optional[float] = 4000
    use_sample: Optional[bool] = False
    seed: Optional[int] = 0

    def _get_family(self,
                    basept=None):
   
        if basept is None:
            basept = self.estimate
            
        self._rng = np.random.default_rng(self.seed)
        
        if not self.use_sample:
            grid = np.linspace(basept - self.num_sd * self.sigma,
                               basept + self.num_sd * self.sigma, self.num_grid)

            log_weight = np.sum([np.log(np.clip(w(grid), 1e-16,1)) for w in self.weight_fns], 0)
            log_weight -= log_weight.max() + 10
            weight = np.exp(log_weight)

            weight *= normal_dbn.pdf((grid - basept) / self.sigma)

            return discrete_family(grid, weight)

        else:
            sample = self._rng.standard_normal(self.num_grid) * self.sigma + basept

            log_weight = np.sum([np.log(w(sample)) for w in self.weight_fns])
            log_weight -= log_weight.max() + 10
            weight = np.exp(log_weight)

            return discrete_family(sample, weight)

    def pvalue(self,
               null_value=0,
               alternative='twosided',
               basept=None):

        if basept is None:
            basept = null_value

        if alternative not in ['twosided', 'greater', 'less']:
            raise ValueError("alternative should be one of ['twosided', 'greater', 'less']")
        _family = self._get_family(basept=basept)
        tilt = (null_value - basept) / self.sigma**2
        if alternative in ['less', 'twosided']:
            _cdf = _family.cdf(tilt, x=self.estimate)
            if alternative == 'less':
                pvalue = _cdf
            else:
                pvalue = 2 * min(_cdf, 1 - _cdf)
        else:
            pvalue = _family.sf(tilt, x=self.estimate)
        return pvalue

    def interval(self,
                 basept=None,
                 level=0.9):
                 
        if basept is None:
            basept = self.estimate
            
        _family = self._get_family(basept=basept)

        L, U = _family.equal_tailed_interval(self.estimate,
                                             alpha=1-level)
        L *= self.sigma**2; L += basept
        U *= self.sigma**2; U += basept

        return L, U

    def MLE(self,
            basept=None):

        if basept is None:
            basept = self.estimate
            
        _family = self._get_family(basept=basept)

        mle, _, _ = _family.MLE(self.estimate)
        mle *= self.sigma**2; mle += basept
        return mle

@dataclass
class AffineConstraint(object):
    """
    Represents an affine constraint of the form Az <= b for selective inference.

    Attributes
    ----------
    linear : np.ndarray
        The matrix A in the constraint Az <= b.
    offset : np.ndarray
        The offset vector b in the constraint.
    observed : np.ndarray
        The observed data vector z.
    solver : Callable
        Function to solve the linear system.
    scale : float
        Scaling factor for the constraint.
    bias : np.ndarray
        Bias vector for the constraint.
    """

    linear: np.ndarray
    offset: np.ndarray
    observed: np.ndarray

    def __post_init__(self):

        if not all (self.linear @ self.observed - self.offset <= 0):
            raise ValueError('constraint not satisfied for observed data')
    
    def interval_constraints(self,
                             target,
                             gamma,
                             tol = 1.e-4):
        r"""
        Given an affine constraint $\{z:Az \leq b \leq \}$ (elementwise)
        specified with $A$ as `support_directions` and $b$ as
        `support_offset`, a new direction of interest $\eta$, and
        an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
        with `covariance` matrix $\Sigma$, this
        function returns $\eta^TZ$ as well as an interval
        bounding this value. 

        The interval constructed is such that the endpoints are 
        independent of $\eta^TZ$, hence the $p$-value
        of `Kac Rice`_
        can be used to form an exact pivot.

        Parameters
        ----------

        support_directions : np.float
             Matrix specifying constraint, $A$.

        support_offsets : np.float
             Offset in constraint, $b$.

        covariance : np.float
             Covariance matrix of `observed_data`.

        observed_data : np.float
             Observations.

        direction_of_interest : np.float
             Direction in which we're interested for the
             contrast.

        tol : float
             Relative tolerance parameter for deciding 
             sign of $Az-b$.

        Returns
        -------

        lower_bound : float

        observed : float

        upper_bound : float

        sigma : float

        """

        # shorthand
        A, b, D = (self.linear,
                   self.offset,
                   self.observed)

        N = D - gamma * target

        U = A.dot(N) - b
        V = A @ gamma
        
        # Inequalities are now U + V @ target <= 0
        
        # adding the zero_coords in the denominator ensures that
        # there are no divide-by-zero errors in RHS
        # these coords are never used in upper_bound or lower_bound

        zero_coords = V == 0
        RHS = -U / (V + zero_coords)
        RHS[zero_coords] = np.nan
        pos_coords = V > tol * np.fabs(V).max()
        neg_coords = V < -tol * np.fabs(V).max()

        if np.any(pos_coords):
            upper_bound = RHS[pos_coords].min()
        else:
            upper_bound = np.inf

        if np.any(neg_coords):
            lower_bound = RHS[neg_coords].max()
        else:
            lower_bound = -np.inf

        return lower_bound, upper_bound

    solver: Callable
    scale: float
    bias: np.ndarray 

    def compute_weight(self,
                       estimate,
                       variance,
                       covariance,
                       tol = 1.e-4):
        r"""

        """

        if np.linalg.norm(covariance) > 1e-8 * variance:

            soln = self.solver(covariance)
            num = (covariance * soln).sum()  # C' \Sigma^{-1} C where \Sigma is the (unscaled) covariance, i.e. not scaled by alpha/scale
            sigma = np.sqrt(variance)
            den = self.scale + num / variance # scale + C' \Sigma^{-1} C / sigma^2
            regression_variance = num / den # this is variance of the regression estimate given N

            smoothing_variance = self.scale * variance**2 / num 
            regression_slice_dir = covariance / regression_variance # this needs to be checked in latex and units
            regression_estimate = (soln * (self.observed - self.bias)).sum() / den

            unbiased_estimate = variance * (soln * (self.observed - self.bias)).sum() / num
            unbiased_slice_dir = covariance / variance

            factor = regression_variance / variance
            (lower_bound,
             upper_bound) = self.interval_constraints(unbiased_estimate * factor,
                                                      unbiased_slice_dir / factor,
                                                      tol=tol)
        else:
            smoothing_variance = 1
            lower_bound, upper_bound = -np.inf, np.inf
            noisy_estimate = 0
            unbiased_slice_dir = covariance * 0
            unbiased_estimate = estimate # ignored because slice_dir is 0 and bounds are \pm inf
            factor = 1
            
        tg =  TruncatedGaussian(estimate=estimate,
                                sigma=np.sqrt(variance),
                                smoothing_sigma=np.sqrt(smoothing_variance) * factor,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                noisy_estimate=unbiased_estimate,
                                factor=factor)
        return tg


def lasso_inference(glmnet_obj,
                    lambda_val,
                    selection_data,
                    full_data,
                    proportion, # proportion used in selection
                    level=0.9,
                    dispersion=None,
                    seed=0,
                    num_sd=10,
                    num_grid=4000,
                    use_sample=False,
                    inactive=False):
    """
    Perform post-selection inference for the LASSO using the GLMNet object.

    Parameters
    ----------
    glmnet_obj : GLMNet
        The fitted GLMNet object.
    lambda_val : float
        The regularization parameter value.
    selection_data : tuple
        (X_sel, Df_sel) data used for selection.
    full_data : tuple
        (X, Df) full data for inference.
    proportion : float
        Proportion of data used for selection.
    level : float, optional
        Confidence level for intervals (default 0.9).
    dispersion : float, optional
        Dispersion parameter (default None).
    seed : int, optional
        Random seed (default 0).
    num_sd : float, optional
        Number of standard deviations for grid (default 10).
    num_grid : int, optional
        Number of grid points (default 4000).
    use_sample : bool, optional
        Whether to use sample-based approximation (default False).
    inactive : bool, optional
        Whether to include inference for inactive variables (default False).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with inference results for the active set, or None if no active set.
    """

    fixed_lambda, warm_state = glmnet_obj.get_fixed_lambda(lambda_val)
    X_sel, Df_sel = selection_data

    fixed_lambda.fit(X_sel,
                     Df_sel,
                     warm_state=warm_state)

    FL = fixed_lambda # shorthand
    state = FL.state_
    lambda_val = FL.lambda_val
    
    if inactive:
        _, _, Y_sel, _, weight_sel = glmnet_obj.get_data_arrays(X_sel,
                                                                Df_sel)
        if weight_sel is None:
            weight_sel = np.ones(X_sel.shape[0])
        logl_score = state.logl_score(glmnet_obj._family,
                                      Y_sel,
                                      weight_sel / weight_sel.sum())
        score = (FL.design_.T @ logl_score)[1:]
    else:
        score = None
        
    GNI = GLMNetInference(glmnet_obj,
                          full_data,
                          lambda_val,
                          state,
                          score,
                          proportion,
                          dispersion=dispersion,
                          inactive=inactive)

    if GNI.active_set_.shape[0] > 0:
        return GNI.summarize(level=level)

@dataclass
class GLMNetInference(object):
    """
    Encapsulates post-selection inference for a fitted GLMNet model.

    Attributes
    ----------
    glmnet_obj : GLMNet
        The fitted GLMNet object.
    data : tuple
        (X, Df) data used for inference.
    lambda_val : float
        The regularization parameter value.
    state : GLMState
        The state of the fitted model.
    score : np.ndarray
        The score vector for inference.
    proportion : float
        Proportion of data used for selection.
    dispersion : float, optional
        Dispersion parameter.
    inactive : bool, optional
        Whether to include inference for inactive variables.
    """

    glmnet_obj: GLMNet
    data: tuple
    lambda_val: float
    state: GLMState
    score: np.ndarray
    proportion: float
    dispersion: Optional[float] = None
    inactive: bool = False

    def __post_init__(self):
        
        (glmnet_obj,
         data,
         lambda_val,
         state,
         score,
         proportion) = (self.glmnet_obj,
                        self.data,
                        self.lambda_val,
                        self.state,
                        self.score,
                        self.proportion)  

        G = self.glmnet_obj # shorthand

        # the upper / lower limits introduce "inactive constraints" -- need to
        if (not np.all(G.upper_limits >= G.control.big) or
            not np.all(G.lower_limits <= -G.control.big)):
            raise NotImplementedError('upper/lower limits coming soon')

        # with upper / lower limits this must check which constraints are tight

        self.active_set_ = active_set = np.nonzero(state.coef != 0)[0]

        X_full, Df_full = data
        _, _, Y_full, _, weight_full = G.get_data_arrays(X_full,
                                                         Df_full)
        ridge_coef = (1 - G.alpha) * lambda_val * weight_full.sum()

        if active_set.shape[0] != 0:

            # fit unpenalized model on full data -- this determines
            # the baseline quadratic form used in the LASSO

            # need to fix this for ENet?
            
            unreg_GLM = glmnet_obj.get_GLM(ridge_coef=ridge_coef)
            unreg_GLM.summarize = True
            unreg_GLM.fit(X_full[:,active_set],
                          Df_full,
                          dispersion=self.dispersion)

            # quadratic approximation

            self.D_active = _get_design(X_full[:,active_set],
                                        weight_full,
                                        standardize=glmnet_obj.standardize,
                                        intercept=glmnet_obj.fit_intercept)
            D = self.D_active

            info = unreg_GLM._information
            P_active = D.quadratic_form(info,
                                        transformed=True)

            # our proportion factor on the assumption that
            # P_noisy represents (up to dispersion) the posterior precision of the
            # non-LASSO penalized estimator represented by the selection data

            # that is, P_noisy = proportion * P_active
            # the quadratic from from D above is only the likelihood contribution
            # this is the diagonal contribution and so we add what we added to P_noisy
            # which should be inflated by 1 / proportion

            # in cases where the selection data is a subset of the full data
            # this factor will be captured by weight_full.sum()

            D0 = np.ones(D.shape[1])
            D0[0] = 0
            DIAG_active = np.diag(D0) * ridge_coef

            if not G.fit_intercept:
                if G.penalty_factor is not None:
                    penfac = G.penalty_factor[active_set]
                else:
                    penfac = np.ones_like(active_set)
                P_active = P_active[1:,1:]
                DIAG_active = DIAG_active[1:,1:]
            else:
                penfac = np.ones(active_set.shape[0])

            self.Q_active_ = Q_active = np.linalg.inv(P_active + DIAG_active) 

            # compute the estimated covariance of `active_est_`
            transform_to_raw = self.D_active.unscaler_ @ np.identity(self.D_active.shape[1])
            if not self.glmnet_obj.fit_intercept:
                transform_to_raw = transform_to_raw[1:, 1:]

            self.C_active_ = (transform_to_raw @
                              self.Q_active_ @
                              transform_to_raw.T) * unreg_GLM.dispersion_
            signs = np.sign(state.coef[active_set])

            if G.fit_intercept:
                penfac = np.hstack([0, penfac])
                signs = np.hstack([0, signs])
                stacked = np.hstack([state.intercept,
                                     state.coef[active_set]])
            else:
                stacked = state.coef[active_set]
            self.stacked_ = stacked
            
            # now set up the constraints

            penalized = penfac > 0
            n_penalized = penalized.sum()
            n_coef = penalized.shape[0]
            row_idx = np.arange(n_penalized)
            col_idx = np.nonzero(penalized)[0]
            data = -signs[penalized]
            sel_active = scipy.sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n_penalized, n_coef))

            # store the MLE/MAP from the GLM

            self.active_est_ = np.hstack([unreg_GLM.intercept_,
                                          unreg_GLM.coef_])

            if not G.fit_intercept:
                self.active_est_ = self.active_est_[1:]

            linear = sel_active
            offset = np.zeros(sel_active.shape[0]) 

            # up to the scalar alpha, this should be the precision of the noise added
            active_solver = lambda v: (P_active + DIAG_active) @ v / unreg_GLM.dispersion_

            active_bias = -Q_active @ (penfac * lambda_val * signs) * weight_full.sum()
            self.active_con = AffineConstraint(linear=linear,
                                               offset=offset,
                                               observed=stacked,
                                               scale=(1-proportion)/proportion, # change 06/30/25 -- where does this scale go?
                                               solver=active_solver,
                                               # the bias subtracted from the unpenalized MLE -- needed to get
                                               # a (marginally) unbiased estimate of each target of interest
                                               bias=active_bias) 

            self.dispersion_ = unreg_GLM.dispersion_
        else:
            self.active_con = None
            # this should be fixed for case when there is an intercept but no
            # features selected!!!
            Q_active = None

        # compute inactive set as a boolean mask

    
        inactive_bool = np.ones(X_full.shape[1], bool)
        inactive_bool[active_set] = 0
        self.inactive_bool_ = inactive_bool

        if self.inactive:


            D_all = _get_design(X_full,
                                weight_full,
                                standardize=glmnet_obj.standardize,
                                intercept=glmnet_obj.fit_intercept)
            # we don't want the row corresponding to intercept
            # we also then restrict to inactive set
            
            L_cross = D_all.quadratic_form(info,
                                           columns=active_set,
                                           transformed=True)[1:][inactive_bool]
            # if there was not fitting for the intercept
            # it will nto be in Q_active
            if not glmnet_obj.fit_intercept:
                L_cross = L_cross[:,1:] # remove the intercept column

            # this needs to be handled properly for big p
            self.D_inactive = _get_design(X_full[:,inactive_bool],
                                          weight_full,
                                          standardize=glmnet_obj.standardize,
                                          intercept=glmnet_obj.fit_intercept)
            D_I = self.D_inactive
            # whether or not we fit an intercept
            # we don't want intercept column in the result below
            P_inactive = D_I.quadratic_form(info,
                                            transformed=True)[1:,1:]

            P_inactive += ridge_coef * np.identity(P_inactive.shape[0])
            if Q_active is not None:
                P_inactive -= L_cross @ Q_active @ L_cross.T
            Q_inactive = np.linalg.inv(P_inactive)

            pf = G.penalty_factor
            if pf is None:
                pf = np.ones(G.design_.shape[1]-1)

            # X_{-E}'(Y-X\hat{\beta}_E) -- \hat{\beta}_E is the LASSO soln, not GLM !
            # this is (2nd order Taylor series sense) equivalent to
            # X_{-E}'(Y-X\bar{\beta}_E) + X_{-E}'WX_E(X_E'WX_E)^{-1}\lambda_E s_E
            # with \bar{\beta}_E the GLM soln

            score = score[1:] # drop the intercept column

            score = score[inactive_bool]
            # we now know that `score` is bounded by \pm lambda_val

            I = scipy.sparse.eye(score.shape[0])
            L = scipy.sparse.vstack([I, -I])
            O = (np.ones(L.shape[0]) * lambda_val *
                 np.hstack([pf[inactive_bool],
                            pf[inactive_bool]]))

            fudge_factor = 0.02 # allow 2% relative error on inactive gradient
            inactive_solver = lambda v: (Q_inactive @ v) * unreg_GLM.dispersion_
            self.inactive_con = AffineConstraint(linear=L,
                                                 offset=O * (1 + fudge_factor),
                                                 observed=score,
                                                 scale=(1-proportion)/proportion,
                                                 bias=-L_cross @ active_bias,
                                                 solver=inactive_solver)

        else:
            self.inactive_con = None
    
    def summarize(self,
                  level=0.9,
                  estimate_cov=None,
                  inactive_cov=None,
                  do_pvalue=True,
                  do_confidence_interval=True,
                  do_mle=True,
                  seed=0,
                  num_sd=10,
                  num_grid=4000,
                  use_sample=False):

        if estimate_cov is None:
            # use the parametric estimate of covariance
            estimate_cov = self.C_active_
        if inactive_cov is None:
            inactive_cov = scipy.sparse.csc_array((self.inactive_bool_.sum(),
                                                   estimate_cov.shape[1]))

        ## iterate over coordinates
        Ls = np.zeros_like(self.stacked_)
        Us = np.zeros_like(self.stacked_)
        mles = np.zeros_like(self.stacked_)
        pvals = np.zeros_like(self.stacked_)
        TGs = []
        WGs = []

        for i in range(self.Q_active_.shape[0]):

            if self.inactive:
                if scipy.sparse.issparse(inactive_cov):
                    i_cov = inactive_cov[:,[i]].todense().reshape(-1)
                else:
                    i_cov = inactive_cov[:,i]
            else:
                i_cov = None
            ((L, U),
             mle,
             pval,
             active_,
             WG) = self.summarize_target(self.active_est_[i],
                                         estimate_cov[i, i],
                                         [estimate_cov[i],
                                          i_cov],
                                         level=level,
                                         do_pvalue=do_pvalue,
                                         do_confidence_interval=do_confidence_interval,
                                         do_mle=do_mle,
                                         seed=seed,
                                         num_sd=num_sd,
                                         num_grid=num_grid,
                                         use_sample=use_sample)


            Ls[i] = L
            Us[i] = U
            mles[i] = mle
            pvals[i] = pval
            TGs.append(active_)
            WGs.append(WG)

        idx = self.active_set_.tolist()
        if self.glmnet_obj.fit_intercept:
            idx = ['intercept'] + idx

        WG_df = pd.concat([pd.Series(asdict(wg)) for wg in WGs], axis=1).T
        WG_df.index = idx
        TG_df = pd.concat([pd.Series(asdict(tg)) for tg in TGs], axis=1).T
        TG_df.index = idx
        df = pd.concat([pd.DataFrame({'mle': mles, 'pval': pvals, 'lower': Ls, 'upper': Us},
                                     index=idx), WG_df, TG_df], axis=1)
        df['WG'] = WGs
        df['TG'] = TGs

        return df

    def summarize_target(self,
                         estimate,
                         variance,
                         covariances,
                         do_pvalue=True,
                         do_confidence_interval=True,
                         do_mle=True,
                         level=0.9,
                         seed=0,
                         num_sd=10,
                         num_grid=4000,
                         use_sample=False):
        
        # covariances are covariances with the raw
        # coef / score -- we must convert them back
        
        active_cov, inactive_cov = covariances
        if self.glmnet_obj.fit_intercept:
            active_cov = self.D_active.scaler_ @ active_cov
        else:
            active_cov = (self.D_active.scaler_ @
                          np.hstack([0, active_cov]))[1:]

        active_ = self.active_con.compute_weight(estimate,
                                                 variance,
                                                 active_cov)


        active_W = active_.weight

        if self.inactive:
            # convert to scaled
            inactive_cov = (self.D_inactive.scaler_ @
                            np.hstack([0, inactive_cov]))[1:]
            inactive_ = self.inactive_con.compute_weight(estimate,
                                                         variance,
                                                         inactive_cov)
            inactive_W = inactive_.weight
            weight_fns = [active_W, inactive_W]
        else:
            weight_fns = [active_W]

        _fam_opts = dict(seed=seed,
                         num_sd=num_sd,
                         num_grid=num_grid,
                         use_sample=use_sample)

        WG = WeightedGaussianFamily(estimate=estimate,
                                    sigma=np.sqrt(variance),
                                    weight_fns=weight_fns,
                                    **_fam_opts)
        if do_pvalue:
            pvalue_ = WG.pvalue()
        else:
            pvalue_ = np.nan

        if do_mle:
            mle_ = WG.MLE()
        else:
            mle_ = np.nan

        if do_confidence_interval:
            L, U = WG.interval(level=level)
        else:
            L, U = np.nan, np.nan

        return (L, U), mle_, pvalue_, active_, WG

    @staticmethod
    def from_score(score,
                   cov_score,
                   lambda_val,
                   proportion=0.8,
                   level=0.9,
                   chol_cov=None,
                   perturbation=None,
                   penalty_factor=None,
                   rng=None,
                   compute_fission=False):

        (GNI,
         (beta_perp,
          beta_perp_cov)) = _score_inference(score,
                                             cov_score,
                                             lambda_val,
                                             proportion=proportion,
                                             chol_cov=chol_cov,
                                             perturbation=perturbation,
                                             penalty_factor=penalty_factor,
                                             level=level,
                                             rng=rng)
        if compute_fission:
            active = GNI.active_set_ 
            fission_summary = _simple_score_inference(beta_perp,
                                                      beta_perp_cov,
                                                      active=active,
                                                      level=level)
            fission_summary.index = active
            GNI.fission_summary_ = fission_summary

        return GNI
    
    @staticmethod
    def from_parametric(glmnet_obj,
                        X,
                        Df, # "Y" argument of sklearn
                        lambda_val,
                        level=0.9,
                        proportion=0.8,
                        rng=None,
                        inactive=False,
                        compute_fission=False):

        glm = glmnet_obj.get_GLM()
        glm.summarize = True
        glm.fit(X, Df, dispersion=None)
        
        param = np.hstack([1. * glm.intercept_, glm.coef_.copy()])
        penalty_factor = np.ones_like(param)
        penalty_factor[0] = 1e-8

        (GNI_raw, (beta_perp,
                   beta_perp_cov)) = _coef_inference(param,
                                                     glm.covariance_,
                                                     lambda_val,
                                                     proportion=proportion,
                                                     penalty_factor=penalty_factor,
                                                     rng=rng)                                                 
        # correct the GNI as it doesn't treat intercept
        # in "special" fashion

        new_state = GLMState(coef=GNI_raw.state.coef[1:],
                             intercept=GNI_raw.state.coef[0])
        
        GNI = GLMNetInference(glmnet_obj=glmnet_obj,
                              data=(X, Df),
                              lambda_val=GNI_raw.lambda_val,
                              state=new_state,
                              score=GNI_raw.score,
                              proportion=GNI_raw.proportion,
                              dispersion=None,
                              inactive=GNI_raw.inactive)

        if compute_fission:
            active = [0] + list(GNI.active_set_ + 1)
            fission_summary = _simple_score_inference(beta_perp,
                                                      beta_perp_cov,
                                                      active=active,
                                                      level=level)
            fission_summary.index = ['intercept'] + list(GNI.active_set_)
            GNI.fission_summary_ = fission_summary
        
        return GNI

        # code below tries to use parameteric bootstrap rather
        # than a perturbation of the asymptotic gaussian...
        # param_boot = np.squeeze(parametric_GLM(X,
        #                                        Df,
        #                                        glm=glm,
        #                                        B=1)[0])

        # perturbation = param - param_boot
        # factor = np.sqrt((1 - proportion) / proportion)
        # beta_perp = param - perturbation / factor
        # beta_perp_cov = (1 + 1 / factor**2) * glm.covariance_

        # # the "-" means that beta_star - beta_hat has perturbation
        # # added (rather than subtracted) from it
        
        # O_perturb = factor * glm.design_ @ perturbation
        # if glmnet_obj.offset_id is not None:
        #     O_perturb += Df[glmnet_obj.offset_id]

        # G_star = clone(glmnet_obj)
        # G_star.lambda_values = [lambda_val]
        # G_star.offset_id = f'boot_offset_{id(G_star)}'
        # Df_ = copy(Df)
        # Df_[f'boot_offset_{id(G_star)}'] = O_perturb

        # G_star.fit(X, Df_)
        
        # intercept, coef = G_star.intercepts_[0], G_star.coefs_[0]

        # ## Compute the gradient of the negative
        # # log-likelihood at LASSO solution
        # # we normalize by weights.sum() as this is the objective
        # # used in GLMNET so this scaled_grad actually satisfies our
        # # KKT conditions

        # scaled_grad, resid = compute_grad(G_star,
        #                                   intercept,
        #                                   coef,
        #                                   G_star.design_,
        #                                   Df[G_star.response_id],
        #                                   scaled_output=True,
        #                                   norm_weights=True)

        # ## Convert the LASSO solution to scaled coordinates

        # raw_state = GLMState(intercept=intercept,
        #                      coef=coef)
        # scaled_state = G_star.design_.raw_to_scaled(state=raw_state)

        # W = G_star.get_data_arrays(X, Df_)[-1]

        # GNI = GLMNetInference(glmnet_obj,
        #                       (X, Df),
        #                       lambda_val,
        #                       scaled_state,
        #                       scaled_grad,
        #                       proportion,
        #                       inactive=inactive)

        # if len(GNI.active_set_) > 0:
        #     active = GNI.active_set_
            
        #     if glm.fit_intercept:
        #         fission_summary = _simple_score_inference(
        #                              beta_perp,
        #                              beta_perp_cov,
        #                              active=[0]+list(active+1),
        #                              level=level)
        #         fission_summary.index = ['intercept'] + list(active)
        #         print(np.diag(beta_perp_cov), 'param')
        #     else:
        #         fission_summary = _simple_score_inference(
        #                              beta_perp[1:],
        #                              beta_perp_cov[1:,1:],
        #                              active=active,
        #                              level=level)
        #         fission_summary.index = active
                
        #     GNI.fission_summary_ = fission_summary
        # elif compute_fission:
        #     GNI.fission_summary_ = None

        # return GNI

    def from_split(glmnet_obj,
                   X,
                   Df, # "Y" argument of sklearn
                   lambda_val,
                   level=0.9,
                   proportion=0.8,
                   rng=None,
                   inactive=False,
                   compute_fission=False):

        (GNI,
         (X_split,
          Df_split)) = _split_inference(glmnet_obj,
                                        X,
                                        Df,
                                        lambda_val,
                                        proportion=proportion,
                                        rng=rng,
                                        inactive=inactive)

        if compute_fission and len(GNI.active_set_) > 0:
            active = GNI.active_set_
            glm = glmnet_obj.get_GLM()
            glm.summarize = True
            X_a = X_split[:,active]
            glm.fit(X_a,
                    Df_split,
                    dispersion=None)
            
            if glm.fit_intercept:
                fission_summary = _simple_score_inference(
                                     np.hstack([glm.intercept_,
                                                glm.coef_]),
                                     glm.covariance_,
                                     level=level)
                fission_summary.index = ['intercept'] + list(active)

            else:
                fission_summary = _simple_score_inference(
                                     glm.coef_,
                                     glm.covariance_,
                                     level=level)
                fission_summary.index = active
                
            GNI.fission_summary_ = fission_summary
        elif compute_fission:
            GNI.fission_summary_ = None
            
        return GNI

    @staticmethod
    def from_resample(sample,
                      lam_frac=1,
                      proportion=0.8,
                      level=0.9,
                      random_idx=None,
                      penalty_factor=None,
                      rng=None,
                      estimate=None,
                      standardize=True,
                      inactive=False,
                      compute_fission=False):

        (GNI,
         (beta_perp,
          beta_perp_cov)) = _resampler_inference(sample,
                                                 lam_frac=lam_frac,
                                                 proportion=proportion,
                                                 random_idx=random_idx,
                                                 penalty_factor=penalty_factor,
                                                 rng=rng,
                                                 estimate=estimate,
                                                 standardize=standardize,
                                                 inactive=inactive)

        if compute_fission:
            active = GNI.active_set_
            fission_summary = _simple_score_inference(beta_perp,
                                                      beta_perp_cov,
                                                      active=active,
                                                      level=level)
            fission_summary.index = active
            GNI.fission_summary_ = fission_summary

        return GNI
    
def _norm_interval(lower, upper):
    r"""
    A multiprecision evaluation of

    .. math::

        \Phi(U) - \Phi(L)

    Parameters
    ----------

    lower : float
        The lower limit $L$

    upper : float
        The upper limit $U$

    """
    #cdf = normal_dbn.cdf
    cdf = mp.ncdf
    if lower > 0:
        if lower < 4:
            return cdf(-lower) - cdf(-upper)
        else:
            return np.exp(-np.fabs(lower*(upper-lower)))
    elif upper < 0:
        if upper < -4:
            return np.exp(-np.fabs(upper*(upper-lower)))
        else:
            return cdf(-lower) - cdf(-upper)
    else:
        return cdf(upper) - cdf(lower)

def find_root(f, y, lb, ub, tol=1e-6):
    """
    searches for solution to f(x) = y in (lb, ub), where 
    f is a monotone decreasing function
    """       
    
    # make sure solution is in range
    a, b   = lb, ub
    fa, fb = f(a), f(b)
    
    # assume a < b
    if fa > y and fb > y:
        while fb > y : 
            b, fb = b + (b-a), f(b + (b-a))
    elif fa < y and fb < y:
        while fa < y : 
            a, fa = a - (b-a), f(a - (b-a))
    
    
    # determine the necessary number of iterations
    try:
        max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )
    except OverflowError:
        warnings.warn('root finding failed, returning np.nan')
        return np.nan
        

    # bisect (slow but sure) until solution is obtained
    for _ in range(max_iter):
        try:
            c, fc  = (a+b)/2, f((a+b)/2)
            if fc > y: a = c
            elif fc < y: b = c
        except OverflowError:
            warnings.warn('root finding failed, returning np.nan')
            return np.nan

    return c
        
class discrete_family(object):

    def __init__(self, sufficient_stat, weights, theta=0.):
        r"""
        A  discrete 1-dimensional
        exponential family with reference measure $\sum_j w_j \delta_{X_j}$
        and sufficient statistic `sufficient_stat`. For any $\theta$, the distribution
        is

        .. math::
        
            P_{\theta} = \sum_{j} e^{\theta X_j - \Lambda(\theta)} w_j \delta_{X_j}

        where

        .. math::

            \Lambda(\theta) = \log \left(\sum_j w_j e^{\theta X_j} \right).

        Parameters
        ----------

        sufficient_stat : `np.float((n))`

        weights : `np.float(n)`

        Notes
        -----

        The weights are normalized to sum to 1.
        """
        xw = np.array(sorted(zip(sufficient_stat, weights)), float)
        xw[:,1] = np.maximum(xw[:,1], 1e-40)
        self._x = xw[:,0]
        self._w = xw[:,1]
        self._lw = np.log(xw[:,1])
        
        self._w /= self._w.sum() # make sure they are a pmf
        self.n = len(xw)
        self._theta = np.nan
        self.theta = theta

    @property
    def theta(self):
        """
        The natural parameter of the family.
        """
        return self._theta

    @theta.setter
    def theta(self, _theta):
        if _theta != self._theta:
            _thetaX = _theta * self.sufficient_stat + self._lw
            _largest = _thetaX.max() - 15 # try to avoid over/under flow, 5 seems arbitrary
            _exp_thetaX = np.exp(_thetaX - _largest)
            _prod = _exp_thetaX
            self._partition = np.sum(_prod)
            self._pdf = _prod / self._partition
            self._partition *= np.exp(_largest)
        self._theta = _theta

    @property
    def partition(self):
        r"""
        Partition function at `self.theta`:

        .. math::

            \sum_j e^{\theta X_j} w_j
        """
        if hasattr(self, "_partition"):
            return self._partition

    @property
    def sufficient_stat(self):
        """
        Sufficient statistics of the exponential family.
        """
        return self._x

    @property
    def weights(self):
        """
        Weights of the exponential family.
        """
        return self._w

    def pdf(self, theta):
        r"""
        Density of $P_{\theta}$ with respect to $P_0$.

        Parameters
        ----------

        theta : float
             Natural parameter.

        Returns
        -------

        pdf : np.float
        
        """
        self.theta = theta # compute partition if necessary
        return self._pdf
 
    def cdf(self, theta, x=None, gamma=1):
        r"""
        The cumulative distribution function of $P_{\theta}$ with
        weight `gamma` at `x`

        .. math::

            P_{\theta}(X < x) + \gamma * P_{\theta}(X = x)

        Parameters
        ----------

        theta : float
             Natural parameter.

        x : float (optional)
             Where to evaluate CDF.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        cdf : np.float

        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf) - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.sufficient_stat < x)) 
            if x in self.sufficient_stat:
                tr += gamma * np.sum(pdf[np.where(self.sufficient_stat == x)])
            return tr

    def sf(self, theta, x=None, gamma=0, return_unnorm=False):
        r"""
        The complementary cumulative distribution function 
        (i.e. survival function) of $P_{\theta}$ with
        weight `gamma` at `x`

        .. math::

            P_{\theta}(X > x) + \gamma * P_{\theta}(X = x)

        Parameters
        ----------

        theta : float
             Natural parameter.

        x : float (optional)
             Where to evaluate SF.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        sf : np.float

        """
        pdf = self.pdf(theta)
        if x is None:
            return np.cumsum(pdf[::-1])[::-1] - pdf * (1 - gamma)
        else:
            tr = np.sum(pdf * (self.sufficient_stat > x)) 
            if x in self.sufficient_stat:
                tr += gamma * np.sum(pdf[np.where(self.sufficient_stat == x)])
            return tr

    def E(self, theta, func):
        r"""
        Expectation of `func` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func : callable
             Assumed to be vectorized.

        gamma : float(optional)
             Weight given at `x`.

        Returns
        -------

        E : np.float

        """
        T = np.asarray(func(self.sufficient_stat))
        pdf_ = self.pdf(theta)

        if T.ndim == 1:
            return (T * pdf_).sum()
        else:
            val = (T * pdf_[:,None]).sum(0)
            return val


    def Var(self, theta, func):
        r"""
        Variance of `func` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func : callable
             Assumed to be vectorized.

        Returns
        -------

        var : np.float

        """

        mu = self.E(theta, func)
        return self.E(theta, lambda x: (func(x)-mu)**2)
        
    def Cov(self, theta, func1, func2):
        r"""
        Covariance of `func1` and `func2` under $P_{\theta}$

        Parameters
        ----------

        theta : float
             Natural parameter.

        func1, func2 : callable
             Assumed to be vectorized.

        Returns
        -------

        cov : np.float

        """

        mu1 = self.E(theta, func1)
        mu2 = self.E(theta, func2)
        return self.E(theta, lambda x: (func1(x)-mu1)*(func2(x)-mu2))

    def equal_tailed_interval(self,
                              observed,
                              alpha=0.05,
                              randomize=True,
                              auxVar=None,
                              tol=1e-6):
        """
        Form interval by inverting
        equal-tailed test with $\alpha/2$ in each tail.

        Parameters
        ----------

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        auxVar : [None, float]
             If randomizing and not None, use this
             as the random uniform variate.

        Returns
        -------

        lower, upper : float
             Limits of confidence interval.

        """

        mu = self.E(self.theta, lambda x: x)
        sigma  = np.sqrt(self.Var(self.theta, lambda x: x))
        lb = mu - 20 * sigma
        ub = mu + 20 * sigma
        F = lambda th : self.cdf(th, observed)
        L = find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        U = find_root(F, 0.5 * alpha, lb, ub)
        return L, U

    def equal_tailed_test(self,
                          theta0,
                          observed,
                          alpha=0.05):
        r"""
        Equal tailed test of H_0:theta=theta_0

        Parameters
        ----------

        theta0 : float
             Natural parameter under null hypothesis.

        observed : float
             Observed sufficient statistic.

        alpha : float (optional)
             Size of two-sided test.

        randomize : bool
             Perform the randomized test (or conservative test).

        Returns
        -------

        decision : np.bool
             Is the null hypothesis $H_0:\theta=\theta_0$ rejected?
   
        Notes
        -----

        We need an auxiliary uniform variable to carry out the randomized test.
        Larger auxVar corresponds to x being slightly "larger." It can be passed in,
        or chosen at random. If randomize=False, we get a conservative test.
        """

        pval = self.cdf(theta0, observed, gamma=0.5)
        return min(pval, 1-pval) < alpha

    def MLE(self,
            observed,
            initial=0,
            max_iter=20,
            tol=1.e-4):

        r"""
        Compute the maximum likelihood estimator
        based on observed sufficient statistic `observed`.

        Parameters
        ----------

        observed : float
             Observed value of sufficient statistic

        initial : float
             Starting point for Newton-Raphson

        max_iter : int (optional)
             Maximum number of Newton-Raphson iterations

        tol : float (optional)
             Tolerance parameter for stopping, based
             on relative change in parameter estimate.
             Iteration stops when the change is smaller
             than `tol * max(1, np.fabs(cur_estimate))`.

        Returns
        -------

        theta_hat : float
             Maximum likelihood estimator.
   
        std_err : float
             Estimated variance of `theta_hat` based
             on inverse of variance of sufficient
             statistic at `theta_hat`, i.e. the
             observed Fisher information.

        """

        cur_est = initial

        def first_two_moments(x):
            return np.array([x, x**2]).T
        
        for i in range(max_iter):
            cur_moments = self.E(cur_est, first_two_moments) # gradient and
                                                             # Hessian of CGF
                                                             # (almost)
            grad, hessian = (cur_moments[0] - observed, 
                             cur_moments[1] - cur_moments[0]**2)
            next_est = cur_est - grad / hessian # newton step

            if np.fabs(next_est - cur_est) < tol * max(1, np.fabs(cur_est)):
                break
            cur_est = next_est

            if i == max_iter - 1:
                warnings.warn('Newton-Raphson failed to converge after %d iterations' % max_iter)

        cur_moments = self.E(cur_est, first_two_moments) # gradient and
                                                         # Hessian of CGF
                                                         # (almost)
        grad, hessian = (cur_moments[0] - observed, 
                         cur_moments[1] - cur_moments[0]**2)

        return cur_est, 1. / hessian, grad

def _truncated_inference(estimate,
                         sigma,
                         smoothing_sigma,
                         lower_bound,
                         upper_bound,
                         basept=None,
                         level=0.9):

    if basept is None:
        basept = estimate
    noisy_sigma = np.sqrt(smoothing_sigma**2 + sigma**2)

    if smoothing_sigma > 1e-6 * sigma:

        grid = np.linspace(basept - 10 * sigma, basept + 10 * sigma, 2001)

        weight = (normal_dbn.cdf((upper_bound - grid) / smoothing_sigma)
                  - normal_dbn.cdf((lower_bound - grid) / smoothing_sigma))

        weight *= normal_dbn.pdf((grid - basept) / sigma)

        sel_distr = discrete_family(grid, weight)

        use_sample = False
        if use_sample:
            rng = np.random.default_rng(0)
            sample = rng.standard_normal(2000) * sigma + basept
            weight_s = (normal_dbn.cdf((upper_bound - sample) / smoothing_sigma)
                      - normal_dbn.cdf((lower_bound - sample) / smoothing_sigma))
            sel_distr =  discrete_family(sample, weight_s)

        L, U = sel_distr.equal_tailed_interval(estimate,
                                               alpha=1-level)
        mle, _, _ = sel_distr.MLE(estimate)
        mle *= sigma**2; mle += estimate
        pval = sel_distr.cdf(-basept / sigma**2, estimate)
        pval = 2 * min(pval, 1-pval)
        L *= sigma**2; L += basept
        U *= sigma**2; U += basept

    else:
        warnings.warn('assuming data for selection is same as for inference -- using hard selection')
        
        lb = estimate - 20 * sigma
        ub = estimate + 20 * sigma

        if estimate < lower_bound or estimate > upper_bound:
            warn('Constraints not satisfied: returning [-np.inf, np.inf]')
            return -np.inf, np.inf, np.nan, np.nan
        
        def F(theta):
            
            Z = (estimate - theta) / sigma
            Z_L = (lower_bound - theta) / sigma
            Z_U = (upper_bound - theta) / sigma
            num = _norm_interval(Z_L, Z)
            den = _norm_interval(Z_L, Z_U)
            if Z_L > 0 and den < 1e-10:
                C = np.fabs(Z_L)
                D = Z-Z_L
                cdf = 1-np.exp(-C*D)
            elif Z_U < 0 and den < 1e-10:
                C = np.fabs(Z_U)
                D = Z_U-Z
                if C*D < 0:
                    raise ValueError
                cdf = np.exp(-C*D)
            else:
                cdf = num / den
            return cdf
            
        
        pval = F(0)
        pval = 2 * min(pval, 1-pval)

        lb = lower_bound - 20 * sigma
        ub = upper_bound + 20 * sigma

        alpha = 0.5 * (1 - level)
        L = find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        if np.isnan(L):
            L = -np.inf
        U = find_root(F, 0.5 * alpha, lb, ub)
        if np.isnan(U):
            U = np.inf

        mle = np.nan
        
    return L, U, mle, pval, sel_distr

def _split_inference(glmnet_obj,
                     X,
                     Df, # "Y" argument of sklearn
                     lambda_val,
                     proportion=0.8,
                     level=0.9,
                     rng=None,
                     inactive=False):
    
    G = clone(glmnet_obj)
    G.lambda_values = [lambda_val]

    if rng is None:
        rng = np.random.default_rng()

    n, p = X.shape
    idx = rng.choice(n, int(proportion*n), replace=False)

    X_sel = X[idx]
    Df_sel = Df.iloc[idx]

    heldout = ~(Df.index.isin(idx))
    Df_split = Df.loc[heldout]
    X_split = X[heldout]
    
    G.fit(X_sel, Df_sel)
    intercept, coef = G.intercepts_[0], G.coefs_[0]

    ## Compute the gradient of the negative log-likelihood at LASSO solution
    # we normalize by weights.sum() as this is the objective
    # used in GLMNET so this scaled_grad actually satisfies our
    # KKT conditions
    
    scaled_grad, resid = compute_grad(G,
                                      intercept,
                                      coef,
                                      G.design_,
                                      Df_sel['Y'],
                                      scaled_output=True,
                                      norm_weights=True)

    ## Convert the LASSO solution to scaled coordinates

    raw_state = GLMState(intercept=intercept,
                         coef=coef)
    scaled_state = G.design_.raw_to_scaled(state=raw_state)

    W = G.get_data_arrays(X, Df)[-1]
    final_proportion = W[idx].sum() / W.sum()

    GNI = GLMNetInference(G,
                          (X, Df),
                          lambda_val,
                          scaled_state,
                          scaled_grad,
                          final_proportion,
                          inactive=inactive)

    return GNI, (X_split, Df_split)

def _score_inference(score,
                     cov_score,
                     lambda_val,
                     proportion=0.8,
                     level=0.9,
                     chol_cov=None,
                     perturbation=None,
                     penalty_factor=None,
                     rng=None):

    # perturbation should be N(0, cov_score) roughly

    # this is the X for the LASSO problem
    # X.T @ X = S = cov_score

    if chol_cov is None:
        chol_cov = X = np.linalg.cholesky(cov_score).T 

    if rng is None:
        rng = np.random.default_rng()

    # shorthand
    Z = score # X.T @ Y in OLS case
    X = chol_cov 
    p = X.shape[1]
    
    if perturbation is None:
        perturbation = X.T @ rng.standard_normal(p) # X is square here...

    # this is the Y of the LASSO problem

    Y = scipy.linalg.solve_triangular(chol_cov.T, Z, lower=True)
    W = np.ones(X.shape[0])
    Df = pd.DataFrame({'response':Y, 'weight':W})
    
    GN = GLMNet(response_id='response',
                weight_id='weight',
                fit_intercept=False,
                standardize=False,
                lambda_values=np.array([lambda_val / p]),
                penalty_factor=penalty_factor)

    Z_sel = Z + np.sqrt((1 - proportion) / proportion) * perturbation
    Y_sel = scipy.linalg.solve_triangular(chol_cov.T, Z_sel, lower=True)
    X_sel = X
    W_sel = np.ones(X.shape[0]) * proportion
    Df_sel = pd.DataFrame({'response':Y_sel, 'weight':W_sel})
    GN.fit(X_sel, Df_sel)
    
    state = GN.state_
    logl_score = state.logl_score(GN._family,
                                  Y_sel,
                                  W_sel / W_sel.sum())
    D = _get_design(X,
                    W,
                    standardize=GN.standardize,
                    intercept=GN.fit_intercept)

    score = (D.T @ logl_score)[1:]

    GNI = GLMNetInference(GN,
                          (X, Df),
                          lambda_val / p,
                          state,
                          score,
                          proportion,
                          dispersion=1,
                          inactive=False)

    indep_est = Z - np.sqrt(proportion / (1 - proportion)) * perturbation
    indep_cov = (1 + proportion / (1 - proportion)) * cov_score

    if GNI.active_set_.shape[0] > 0:
        return GNI, (indep_est, indep_cov)
    else:
        return None, (np.nan, np.nan)

def _resampler_inference(sample,
                         lambda_val=None,
                         lam_frac=1,
                         proportion=0.8,
                         random_idx=None,
                         penalty_factor=None,
                         rng=None,
                         estimate=None,
                         standardize=True,
                         inactive=False):

    if estimate is None:
        estimate = sample.mean(0)

    centered = sample - estimate[None,:]
    B, p = centered.shape

    if random_idx is None:
        if rng is None:
            rng = np.random.default_rng()
        random_idx = rng.choice(B, 1)
        
    prec_score = centered.T @ centered / B
    cov_score = np.linalg.inv(prec_score)

    centered_scores = centered @ cov_score
    scaling = centered_scores.std(0)
    score = cov_score @ estimate

    if penalty_factor is None:
        penalty_factor = np.ones(p)
    if standardize:
        penalty_factor *= scaling
        
    # pick a lam
    if lambda_val is None:
        max_scores = np.fabs(centered_scores).max(1)
        lambda_val = lam_frac * np.median(max_scores)

    # pick a pseudo-Gaussian perturbation
    perturbation = centered_scores[random_idx].reshape((p,))
    
    (GNI,
     (perturbation,
      cov)) = _score_inference(score=score,
                               cov_score=cov_score,
                               lambda_val=lambda_val,
                               proportion=proportion,
                               perturbation=perturbation,
                               penalty_factor=penalty_factor,
                               rng=rng)

    # perturbation is on the score scale,
    # convert it to the original coords

    if GNI is not None:
        return (GNI, (prec_score @ perturbation,
                      prec_score @ cov @ prec_score))
    else:
        return (GNI, (perturbation, cov))

def _coef_inference(coef,
                    cov_coef,
                    lambda_val,
                    proportion=0.8,
                    penalty_factor=None,
                    perturbation=None,
                    rng=None,
                    standardize=True,
                    inactive=False):

    cov_score = np.linalg.inv(cov_coef)
    score = cov_score @ coef
    scaling = np.diag(cov_score)

    # perturbation would be on beta/coef scale,
    # convert to score scale
    if perturbation is not None:
        perturbation = cov_score @ perturbation 

    p = cov_score.shape[0]
    
    if penalty_factor is None:
        penalty_factor = np.ones(p)
    if standardize:
        penalty_factor *= scaling
        
    (GNI,
     (perturbation,
      cov)) = _score_inference(score=score,
                               cov_score=cov_score,
                               lambda_val=lambda_val,
                               proportion=proportion,
                               perturbation=perturbation,
                               penalty_factor=penalty_factor,
                               rng=rng)

    # perturbation is on the score scale,
    # convert it to the original coords

    if GNI is not None:
        return (GNI, (cov_coef @ perturbation,
                      cov_coef @ cov @ cov_coef))
    else:
        return (GNI, (perturbation, cov))
    
def _simple_score_inference(beta,
                            beta_cov,
                            active=None,
                            level=0.9):
    q = normal_dbn.ppf(1 - (1 - level) / 2)
    prec = np.linalg.inv(beta_cov)
    score = prec @ beta

    if active is not None:
        Q_a = prec[np.ix_(active, active)]
        C_a = np.linalg.inv(Q_a)
        beta_sel = (C_a @ score[active])
    else:
        C_a = beta_cov
        beta_sel = beta
        
    _sd = np.sqrt(np.diag(C_a))
    _Z = beta_sel / _sd
    _df = pd.DataFrame({'pval':2 * normal_dbn.sf(np.fabs(_Z))},
                        index=active)
    _df['upper'] = beta_sel + q * _sd
    _df['lower'] = beta_sel - q * _sd
    _df['estimate'] = beta_sel
    _df['std err'] = _sd
    return _df
