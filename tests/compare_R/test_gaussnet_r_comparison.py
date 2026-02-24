"""
Test GaussNet comparison with R glmnet using rpy2.

This module validates the GaussNet implementation (Gaussian/Least Squares regression with elastic net regularization)
by comparing its results against the reference implementation in R's `glmnet` package.

The tests cover:
- Comparison of unregularized GLM (Gaussian family) with R's `stats::glm`.
- Comparison of GaussNet (regularized Gaussian model) coefficients with R's `glmnet`.
- Comparison of cross-validation results (Mean Squared Error and SD) with R's `cv.glmnet`.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet import GaussNet, GLM, GLMNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector
from rpy2.robjects import numpy2ri

# Import R packages
stats = importr('stats')
glmnet = importr('glmnet')

# Check if rpy2 is available
try:
    import rpy2
    has_rpy2 = True
except ImportError:
    has_rpy2 = False




def numpy_to_r_matrix(X):
    """
    Convert a 2D numpy array to an R matrix.
    
    Ensures proper row/column major ordering during the conversion.
    
    Parameters
    ----------
    X : np.ndarray
        The 2D numpy array to convert.
        
    Returns
    -------
    ro.r.matrix
        The corresponding R matrix object.
    """
    return ro.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n, p = 103, 20
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    O = rng.standard_normal(n) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    Y = rng.standard_normal(n) * 5
    D = np.array([Y, O, W]).T
    Df = pd.DataFrame(D, columns=['response', 'offset', 'weight'])
    return X, Y, O, W, D, Df


def test_glm_comparison(sample_data, use_offset, use_weights):
    """Test GLM comparison with different alpha, offset, and weight combinations."""
    X, Y, O, W, D, Df = sample_data
    
    # Configure Python GLM
    glm_kwargs = {'family': sm.families.Gaussian()}
    if use_offset:
        glm_kwargs['offset_id'] = 'offset'
    if use_weights:
        glm_kwargs['weight_id'] = 'weight'
    
    G = GLM(response_id='response', **glm_kwargs)
    G.fit(X, Df)
    
    # Configure R glm
    r_data_dict = {'Y': FloatVector(Y)}
    r_kwargs = {'family': stats.gaussian}
    
    if use_offset:
        r_data_dict['O'] = FloatVector(O)
        r_kwargs['offset'] = FloatVector(O)
    if use_weights:
        W_numeric = W.astype(float)
        r_data_dict['W'] = FloatVector(W_numeric)
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    # Add predictor columns
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, **r_kwargs)
    r_coef = np.array(stats.coef(r_model))
    
    # Compare results
    rtol = 1e-5 if not use_weights else 1e-4  # Slightly more lenient with weights
    atol = 1e-5 if not use_weights else 1e-4
    assert np.allclose(G.coef_, r_coef[1:], rtol=rtol, atol=atol)
    assert np.allclose(G.intercept_, r_coef[0], rtol=rtol, atol=atol)


def test_glmnet_flex_comparison(sample_data, alpha, use_offset, use_weights):
    """Test GLMNet comparison with R glmnet."""
    X, Y, O, W, D, Df = sample_data
    
    # Configure Python GLMNet
    glmnet_kwargs = {'family': sm.families.Gaussian(), 'alpha': alpha}
    if use_offset:
        glmnet_kwargs['offset_id'] = 'offset'
    if use_weights:
        glmnet_kwargs['weight_id'] = 'weight'
    
    # standardize y like glmnet
    if alpha < 1:
        sample_weight = W
        y = Df['response']
        # standardize y like glmnet
        w_mean = (y * sample_weight).sum() / sample_weight.sum()
        w_std = np.sqrt(((y - w_mean)**2 * sample_weight).sum() / sample_weight.sum())
        y /= w_std
        Y /= w_std
        Df['response'] = y
        
    GN = GLMNet(response_id='response', **glmnet_kwargs)
    GN.fit(X, Df)
    
    # Configure R glmnet
    r_kwargs = {'family': 'gaussian', 'alpha': alpha}
    if use_offset:
        r_kwargs['offset'] = FloatVector(O)
    if use_weights:
        W_numeric = W.astype(float)
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gn = glmnet.glmnet(numpy_to_r_matrix(X), FloatVector(Y), **r_kwargs)
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    

    # Print lambda values from R object
    r_lambda = np.array(ro.r['as.numeric'](r_gn.rx2('lambda')))
    
    # Compare results (using index 30 as in original)
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(GN.lambda_values_, r_lambda)
    assert np.allclose(r_coef.T[30][1:], GN.coefs_[30], rtol=1e-3, atol=1e-3)


def test_gaussnet_comparison(sample_data, alpha, use_offset, use_weights):
    """Test GaussNet comparison with different alpha, offset, and weight combinations."""
    X, Y, O, W, D, Df = sample_data
    
    # Configure Python GaussNet
    gaussnet_kwargs = {'alpha': alpha}
    if use_offset:
        gaussnet_kwargs['offset_id'] = 'offset'
    if use_weights:
        gaussnet_kwargs['weight_id'] = 'weight'
    
    GN = GaussNet(response_id='response', **gaussnet_kwargs)
    GN.fit(X, Df)
    
    # Configure R glmnet
    r_kwargs = {'family': 'gaussian', 'alpha': alpha}
    if use_offset:
        r_kwargs['offset'] = FloatVector(O)
    if use_weights:
        W_numeric = W.astype(float)
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gn = glmnet.glmnet(numpy_to_r_matrix(X), FloatVector(Y), **r_kwargs)
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[:, 1:], GN.coefs_)
    assert np.allclose(r_coef[0], GN.intercepts_)


def test_cross_validation(sample_data, alpha, alignment, use_offset, use_weights):
    """Test cross-validation with different alpha, alignment, offset, weight, and grouped combinations."""
    X, Y, O, W, D, Df = sample_data
    
    # Configure Python GaussNet with CV
    gaussnet_kwargs = {'response_id': 'response', 'alpha': alpha}
    if use_offset:
        gaussnet_kwargs['offset_id'] = 'offset'
    if use_weights:
        gaussnet_kwargs['weight_id'] = 'weight'
    
    GN = GaussNet(**gaussnet_kwargs)
    GN.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN.cross_validation_path(X, Df, cv=cv, alignment=alignment)
    
    # Configure R cv.glmnet
    r_kwargs = {
        'foldid': IntVector(foldid.astype(int)),
        'family': 'gaussian',
        'alignment': alignment,
        'grouped': True,
        'alpha': alpha
    }
    if use_offset:
        r_kwargs['offset'] = FloatVector(O)
    if use_weights:
        W_numeric = W.astype(float)
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X), FloatVector(Y), **r_kwargs)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN.score_path_.scores['Mean Squared Error'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN.score_path_.scores['SD(Mean Squared Error)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3) 
