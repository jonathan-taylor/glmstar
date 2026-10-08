import warnings

from typing import Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

from .fastnet import FastNetMixin
from ..cox import (CoxFamily,
                   CoxFamilySpec,
                   CoxNetIRLS,
                   CoxState,
                   RegCoxLM,
                   cox_survfit)

from .._utils import (_get_data,
                      _jerr_coxnet)

from .._coxnet import coxnet as _dense
from .._coxnet import spcoxnet as _sparse

"""
Implements the CoxNet path algorithm for Cox proportional hazards models.
Provides the CoxNet estimator class using the FastNetMixin base, backed by
glmnetpp's C++ Cox path (which uses coxdev), as in R's glmnet.
"""

@dataclass
class CoxNet(FastNetMixin):
    """CoxNet estimator for Cox regression using the FastNet path algorithm.

    This class implements the regularization path for Cox proportional
    hazards models using coordinate descent, with Breslow or Efron
    tie-breaking, right-censored or start-stop data, and optional strata.

    The response `y` passed to `fit` must be a DataFrame whose columns are
    named by `family` (see `CoxFamily`), along with any columns named by
    `weight_id` and `offset_id`.

    Parameters
    ----------
    family : CoxFamily, default=CoxFamily()
        Column names for the survival data and tie-breaking method.
    lambda_min_ratio : float, optional
        Minimum lambda ratio.
    nlambda : int, default=100
        Number of lambda values.
    df_max : int, optional
        Maximum degrees of freedom.
    control : FastNetControl, optional
        Control parameters for the solver.

    Attributes
    ----------
    coefs_ : ndarray of shape (n_lambda, n_features)
        Fitted coefficients across the path.
    intercepts_ : ndarray of shape (n_lambda,)
        Fitted intercepts across the path (always 0 for Cox models).
    lambda_values_ : ndarray of shape (n_lambda,)
        The sequence of lambda values used.
    """

    family: CoxFamily = field(default_factory=CoxFamily)
    fit_intercept: Literal[False] = False
    # single lambda fits for get_fixed_lambda, as in CoxNetIRLS
    regularized_estimator: BaseEstimator = RegCoxLM

    _dense = _dense
    _sparse = _sparse
    _jerr_message = staticmethod(_jerr_coxnet)

    # predictions are the linear predictor (risk score), as in CoxNetIRLS
    predict = CoxNetIRLS.predict

    def fit(self,
            X,
            y,
            sample_weight=None, # ignored
            interpolation_grid=None):
        """Fit the Cox regularization path.

        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix.
        y : pd.DataFrame
            Survival data, with optional weight and offset columns.
        sample_weight : array-like, optional
            Sample weights (ignored; use `weight_id`).
        interpolation_grid : array-like, optional
            Grid for coefficient interpolation.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(y, pd.DataFrame):
            raise ValueError('CoxNet requires y to be a DataFrame of survival data')
        if self.fit_intercept:
            raise ValueError('Cox models do not have an intercept; fit_intercept must be False')

        self._family = self._finalize_family(y)
        self._survival_data = self._get_survival_data(y)

        super().fit(X,
                    y,
                    interpolation_grid=interpolation_grid)
        self.state_ = CoxState(self.coefs_[-1])
        return self

    def _fixed_lambda_family(self):
        return self.family

    def survfit(self,
                X,
                y,
                lambda_val=None,
                newX=None,
                new_offset=None,
                new_strata=None,
                tie_breaking=None):
        """Survival curves at one or more points on the path.

        The analogue of R's ``survfit.coxnet``: the baseline hazard is
        estimated with the coefficients held fixed at their values at
        `lambda_val`, so the training data must be passed again.

        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix used in `fit`.
        y : pd.DataFrame
            Survival data used in `fit`, with any weight, offset or strata columns.
        lambda_val : float or array-like, optional
            Value(s) of lambda; coefficients are interpolated linearly between
            path values, as in `predict`. Defaults to the whole path.
        newX : array-like or sparse matrix, optional
            Features of the subjects to compute curves for. If None, one curve
            (per stratum) is computed at the mean linear predictor, as R does.
        new_offset : array-like, optional
            Offsets for `newX`; required if the model was fit with an offset.
        new_strata : array-like, optional
            Strata labels for `newX`; required if the model is stratified.
        tie_breaking : {'efron', 'breslow'}, optional
            Hazard estimate for tied event times. Defaults to the family's
            `tie_breaking`. R's ``survfit.coxnet`` always uses 'efron'.

        Returns
        -------
        CoxSurvivalCurves or list of CoxSurvivalCurves
            A list if `lambda_val` has more than one value.
        """
        check_is_fitted(self, ["coefs_"])

        _, _, _, offset, weight = self.get_data_arrays(X, y, check=False)
        surv_data = self._get_survival_data(y)
        strata = None
        if self.family.strata_id is not None:
            strata = np.asarray(y[self.family.strata_id])
        if offset is None:
            offset = np.zeros(X.shape[0])

        if newX is not None:
            if self.offset_id is not None and new_offset is None:
                raise ValueError('new_offset is required for a model fit with an offset')
            if strata is not None and new_strata is None:
                raise ValueError('new_strata is required for a stratified model')
            if new_offset is None:
                new_offset = np.zeros(newX.shape[0])

        if tie_breaking is None:
            tie_breaking = self.family.tie_breaking

        if lambda_val is None:
            lambda_val = self.lambda_values_
        lambda_val = np.asarray(lambda_val, float)
        coefs, _ = self.interpolate_coefs(np.atleast_1d(lambda_val))

        offset = np.asarray(offset, float).reshape(-1)
        curves = []
        for beta in coefs:
            eta = np.asarray(X @ beta).reshape(-1)
            lp = eta + offset
            # R's default curve is at coxph's `means`, which are weighted for
            # right-censored data but unweighted for (start, stop] data; the
            # offset enters at its weighted mean
            if self.family.start_id is None:
                center = np.average(lp, weights=weight)
            else:
                center = eta.mean() + np.average(offset, weights=weight)
            new_lp = None
            if newX is not None:
                new_lp = (np.asarray(newX @ beta).reshape(-1) +
                          np.asarray(new_offset, float).reshape(-1))
            curves.append(cox_survfit(lp,
                                      surv_data['stop'],
                                      surv_data['status'],
                                      start=surv_data['start'],
                                      strata=strata,
                                      sample_weight=weight,
                                      new_linear_predictor=new_lp,
                                      new_strata=new_strata,
                                      tie_breaking=tie_breaking,
                                      center=center))

        return curves[0] if lambda_val.ndim == 0 else curves

    def get_data_arrays(self,
                        X,
                        y,
                        check=True):
        """Prepare and validate data arrays for Cox regression.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : pd.DataFrame
            Survival data.
        check : bool, default=True
            Whether to check input validity.

        Returns
        -------
        tuple
            Tuple of (X, y, response, offset, weight).
        """
        return _get_data(self,
                         X,
                         y,
                         offset_id=self.offset_id,
                         response_id=self.response_id,
                         weight_id=self.weight_id,
                         check=check,
                         multi_output=True)

    def _finalize_family(self,
                         response):
        return CoxFamilySpec(event_data=response,
                             tie_breaking=self.family.tie_breaking,
                             event_id=self.family.event_id,
                             status_id=self.family.status_id,
                             start_id=self.family.start_id,
                             strata_id=self.family.strata_id)

    def _get_survival_data(self, y):
        """Extract and validate (start, stop, status, strata) as in R's coxnet."""

        family = self.family
        for col in [family.event_id, family.status_id, family.start_id, family.strata_id]:
            if col is not None and col not in y.columns:
                raise ValueError(f'expecting column "{col}" in survival data')

        n = y.shape[0]
        # np.array copies: with pandas >= 3 (copy-on-write) np.asarray of a column
        # is a read-only view, which the C++ binding does not accept
        stop = np.array(y[family.event_id], float)
        status = np.asarray(y[family.status_id])
        if not np.all(np.isin(status, [0, 1])):
            raise ValueError('status should be binary (0=censored, 1=event)')
        status = status.astype(np.int32)

        if family.start_id is not None:
            start = np.array(y[family.start_id], float)
        else:
            start = np.zeros(n)

        if family.strata_id is not None:
            # integer labels; empty means a single stratum
            strata = pd.factorize(y[family.strata_id], sort=True)[0].astype(np.int32)
        else:
            strata = np.zeros(0, np.int32)

        if np.any(stop <= 0):
            raise ValueError('non-positive event times encountered; not permitted for Cox family')
        if np.any(start < 0):
            raise ValueError('negative start times encountered; not permitted for Cox family')
        if np.any(start >= stop):
            raise ValueError('start time must be less than stop time')
        if np.all(status == 0):
            raise ValueError(_jerr_coxnet(8888, self.control.maxit)['msg'])

        return {'start': start,
                'stop': stop,
                'status': status,
                'strata': strata,
                'efron': family.tie_breaking == 'efron'}

    def _wrapper_args(self,
                      design,
                      response,
                      sample_weight,
                      offset,
                      exclude=[]):
        """Prepare arguments for the C++ backend wrapper for Cox regression.

        Parameters
        ----------
        design : object
            Design matrix and related info.
        response : array-like
            Response array (unused; survival data is taken from `fit`).
        sample_weight : array-like
            Sample weights.
        offset : array-like
            Offset array.
        exclude : list, optional
            Indices to exclude from penalization.

        Returns
        -------
        dict
            Arguments for the backend solver.
        """
        if not np.any(np.asarray(sample_weight) > 0):
            raise ValueError(_jerr_coxnet(9999, self.control.maxit)['msg'])

        n_samples = design.X.shape[0]
        if offset is None:
            offset = np.zeros(n_samples)

        _args = super()._wrapper_args(design,
                                      response,
                                      sample_weight,
                                      offset,
                                      exclude=exclude)

        # Cox has no response vector or intercept: survival data replaces y
        del(_args['y'])
        del(_args['intr'])
        _args.update(**self._survival_data)

        _args['g'] = np.asarray(offset, float).reshape(-1).copy()
        _args['w'] = _args['w'].copy()

        return _args
