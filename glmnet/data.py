"""
glmnet.data
------------
Synthetic dataset generation utilities for GLMNet models.

This module provides a function to generate synthetic datasets for various
GLMNet estimator classes (LogNet, GaussNet, MultiGaussNet, MultiClassNet, etc.),
similar to sklearn's make_regression, but with support for different response types
and signal-to-noise ratio (SNR) control.
"""

import numpy as np
from scipy.special import expit, softmax
from numpy.random import default_rng
import pandas as pd


def make_dataset(estimator_class, n_samples=100, n_features=20, n_informative=10, noise=1.0, snr=None,
                 coef=None, random_state=None, bias=None, n_targets=None, n_classes=None, **kwargs):
    """
    Generate a random regression or classification problem for GLMNet estimator classes.

    Parameters
    ----------
    estimator_class : type
        The GLMNet estimator class (e.g., LogNet, GaussNet, MultiGaussNet, MultiClassNet, FishNet).
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=20
        The total number of features.
    n_informative : int, default=10
        The number of informative features.
    noise : float, default=1.0
        Standard deviation of the Gaussian noise added to the output (for regression).
    snr : float or None, default=None
        Desired signal-to-noise ratio. If set, noise will be scaled to achieve this SNR.
    coef : array-like, default=None
        The coefficients to use. If None, random coefficients are generated.
        For multi-output, should be (n_features, n_targets) or (n_features, n_classes).
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation.
    bias : float, array-like, or None, default=None
        The bias (intercept) term in the underlying linear model. If None, a random value is generated.
        For multi-output, can be array-like.
    n_targets : int or None, default=None
        Number of targets for multi-output regression (e.g., MultiGaussNet). If None, defaults to 1.
    n_classes : int or None, default=None
        Number of classes for multiclass classification (e.g., MultiClassNet). If None, defaults to 3.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray
        The output targets (regression, binary, multiclass, or count).
    coef : ndarray
        The underlying true coefficients used to generate the data.
    intercept : float or ndarray
        The intercept (bias) used in the data generation.

    Examples
    --------
    >>> from glmnet.paths import LogNet, GaussNet, MultiGaussNet, MultiClassNet
    >>> X, y, coef, intercept = make_dataset(LogNet, n_samples=100, n_features=10, snr=5)
    >>> X.shape, y.shape, coef.shape, np.shape(intercept)
    ((100, 10), (100,), (10,), ())
    >>> X, y, coef, intercept = make_dataset(MultiGaussNet, n_samples=100, n_features=10, n_targets=3, snr=5)
    >>> X.shape, y.shape, coef.shape, intercept.shape
    ((100, 10), (100, 3), (10, 3), (3,))
    >>> X, y, coef, intercept = make_dataset(MultiClassNet, n_samples=100, n_features=10, n_classes=4, snr=2)
    >>> np.unique(y)
    array([0, 1, 2, 3])
    """
    rng = default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))

    # Ensure n_informative does not exceed n_features
    n_informative = min(n_informative, n_features)

    # Determine family/type from estimator class name
    name = estimator_class.__name__.lower()
    if 'multiclass' in name:
        family = 'multiclass'
    elif 'multigauss' in name:
        family = 'multigaussian'
    elif 'gauss' in name:
        family = 'gaussian'
    elif 'lognet' in name:
        family = 'binomial'
    elif 'fishnet' in name:
        family = 'poisson'
    else:
        raise ValueError(f"Unknown estimator class: {estimator_class}")

    if family == 'gaussian':
        # Standard regression
        if coef is None:
            coef = np.zeros(n_features)
            coef[:n_informative] = rng.normal(0, 1, size=n_informative)
            rng.shuffle(coef)
        if bias is None:
            intercept = rng.normal(0, 1)
        else:
            intercept = bias
        lin_pred = X @ coef + intercept
        if snr is not None:
            signal_var = np.var(lin_pred)
            noise_var = signal_var / snr
            noise = np.sqrt(noise_var)
        y = lin_pred + rng.normal(0, noise, size=n_samples)
    elif family == 'multigaussian':
        # Multi-output regression
        if n_targets is None:
            n_targets = 2
        if coef is None:
            coef = np.zeros((n_features, n_targets))
            for j in range(n_targets):
                coef[:n_informative, j] = rng.normal(0, 1, size=n_informative)
                rng.shuffle(coef[:, j])
        if bias is None:
            intercept = rng.normal(0, 1, size=n_targets)
        else:
            intercept = np.broadcast_to(bias, (n_targets,))
        lin_pred = X @ coef + intercept
        if snr is not None:
            signal_var = np.var(lin_pred, axis=0)
            noise_var = signal_var / snr
            noise = np.sqrt(noise_var)
        y = lin_pred + rng.normal(0, noise, size=(n_samples, n_targets))
    elif family == 'binomial':
        # Binary classification
        if coef is None:
            coef = np.zeros(n_features)
            coef[:n_informative] = rng.normal(0, 1, size=n_informative)
            rng.shuffle(coef)
        if bias is None:
            intercept = rng.normal(0, 1)
        else:
            intercept = bias
        lin_pred = X @ coef + intercept
        p = expit(lin_pred)
        if snr is not None:
            var_signal = np.var(lin_pred)
            var_noise = var_signal / snr
            scale = np.sqrt(var_signal / (var_signal + var_noise))
            lin_pred = lin_pred * scale
            p = expit(lin_pred)
        y = rng.binomial(1, p, size=n_samples)
    elif family == 'multiclass':
        # Multiclass classification
        if n_classes is None:
            n_classes = 3
        if coef is None:
            coef = np.zeros((n_features, n_classes))
            for j in range(n_classes):
                coef[:n_informative, j] = rng.normal(0, 1, size=n_informative)
                rng.shuffle(coef[:, j])
        if bias is None:
            intercept = rng.normal(0, 1, size=n_classes)
        else:
            intercept = np.broadcast_to(bias, (n_classes,))
        lin_pred = X @ coef + intercept
        logits = lin_pred
        if snr is not None:
            var_signal = np.var(logits, axis=0)
            var_noise = var_signal / snr
            scale = np.sqrt(var_signal / (var_signal + var_noise))
            logits = logits * scale
        p = softmax(logits, axis=1)
        y = np.array([rng.choice(n_classes, p=p[i]) for i in range(n_samples)])
    elif family == 'poisson':
        # Poisson regression
        if coef is None:
            coef = np.zeros(n_features)
            coef[:n_informative] = rng.normal(0, 1, size=n_informative)
            rng.shuffle(coef)
        if bias is None:
            intercept = rng.normal(0, 1)
        else:
            intercept = bias
        lin_pred = X @ coef + intercept
        mu = np.exp(lin_pred)
        if snr is not None:
            var_signal = np.var(lin_pred)
            var_noise = var_signal / snr
            scale = np.sqrt(var_signal / (var_signal + var_noise))
            lin_pred = lin_pred * scale
            mu = np.exp(lin_pred)
        y = rng.poisson(mu)
    else:
        raise ValueError(f"Unknown family: {family}.")
    return X, y, coef, intercept


def make_survival(n_samples=100, n_features=20, n_informative=10, coef=None, random_state=None, bias=None,
                  snr=None, baseline_hazard=0.1, start_id=False, discretize=False, **kwargs):
    """
    Generate a random survival (time-to-event) dataset for CoxNet models.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=20
        The total number of features.
    n_informative : int, default=10
        The number of informative features.
    coef : array-like, default=None
        The coefficients to use. If None, random coefficients are generated.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation.
    bias : float or None, default=None
        (Ignored; included for API compatibility.)
    snr : float or None, default=None
        Desired signal-to-noise ratio for the linear predictor. If set, the scale of the linear predictor is adjusted.
    baseline_hazard : float, default=0.1
        The baseline hazard rate for event time simulation.
    start_id : bool, default=False
        If True, include a 'start' column for (start, stop] survival data.
    discretize : bool, default=False
        If True, discretize times to 2 significant digits to create ties in the data.
        This is useful for testing tie-breaking methods in Cox regression.
    **kwargs : dict
        Additional keyword arguments (ignored).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : pandas.DataFrame
        The output DataFrame with columns: 'event', 'status', and optionally 'start'.
    coef : ndarray
        The underlying true coefficients used to generate the data.

    Examples
    --------
    >>> X, y, coef = make_survival(n_samples=100, n_features=10, start_id=True)
    >>> y.columns
    Index(['start', 'event', 'status'], dtype='object')
    >>> X.shape, y.shape, coef.shape
    ((100, 10), (100, 3), (10,))
    
    >>> # Generate data with ties for testing tie-breaking methods
    >>> X, y, coef = make_survival(n_samples=100, n_features=10, discretize=True)
    >>> len(np.unique(y['event'])) < len(y['event'])  # Should have ties
    True
    """
    rng = default_rng(random_state)
    n_informative = min(n_informative, n_features)
    X = rng.standard_normal((n_samples, n_features))
    if coef is None:
        coef = np.zeros(n_features)
        coef[:n_informative] = rng.normal(0, 1, size=n_informative)
        rng.shuffle(coef)
    lin_pred = X @ coef
    if snr is not None:
        var_signal = np.var(lin_pred)
        var_noise = var_signal / snr
        scale = np.sqrt(var_signal / (var_signal + var_noise))
        lin_pred = lin_pred * scale
    # Simulate event times
    U = rng.uniform(0, 1, size=n_samples)
    duration = -np.log(U) / (baseline_hazard * np.exp(lin_pred))
    # Random censoring
    censor_time = rng.exponential(duration.mean(), size=n_samples)
    status = (duration <= censor_time).astype(int)
    observed_time = np.minimum(duration, censor_time)
    
    # Discretize times if requested
    if discretize:
        # Round to 2 significant digits to create ties
        observed_time = np.round(observed_time, decimals=1) + 0.2
        if start_id:
            # Also discretize start times
            start_times = np.zeros(n_samples)
            start_times = np.round(start_times, decimals=1)
    
    data = {'event': observed_time, 'status': status}
    if start_id:
        # Simulate start times that are strictly less than event times
        # Use a fraction of the event times to ensure they're reasonable
        start_times = observed_time * rng.uniform(0.1, 0.8, size=n_samples)
        # Ensure start times are strictly less than event times
        start_times = np.minimum(start_times, observed_time * 0.99)
        if discretize:
            start_times = np.round(start_times, decimals=1)
            # After discretization, ensure strict inequality
            start_times = np.maximum(np.minimum(start_times, observed_time - 0.1), 0)
        data = {'start': start_times, **data}
    y = pd.DataFrame(data)
    return X, y, coef 
