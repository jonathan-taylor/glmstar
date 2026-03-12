"""
Test MultiGaussNet comparison with R glmnet using rpy2.

This test file converts the original IPython R magic cells to use rpy2
for proper R integration in pytest.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet import MultiGaussNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# rpy2 imports
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector, IntVector
    from rpy2.robjects import numpy2ri
    
    # Import R packages
    glmnet = importr('glmnet')
    
    has_rpy2 = True
except ImportError:
    has_rpy2 = False


def numpy_to_r_matrix(X):
    """Convert numpy array to R matrix with proper row/column major ordering."""
    return ro.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n, p, q, nlambda = 103, 17, 3, 100
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    O = rng.standard_normal((n, q)) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    Y = rng.standard_normal((n, q)) * 5
    D = np.column_stack([Y, O, W])
    response_id = [f'response[{i}]' for i in range(q)]
    offset_id = [f'offset[{i}]' for i in range(q)]
    Df = pd.DataFrame(D, columns=response_id + offset_id + ['weight'])
    return X, Y, O, W, Df, response_id, offset_id, nlambda


def test_multigaussnet_comparison(sample_data, alpha, use_offset, use_weights):
    """Test MultiGaussNet comparison with different alpha, offset, and weight combinations."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Configure Python MultiGaussNet
    multigauss_kwargs = {'alpha': alpha, 'nlambda': nlambda}
    if use_offset:
        multigauss_kwargs['offset_id'] = offset_id
    if use_weights:
        multigauss_kwargs['weight_id'] = 'weight'
    
    GN = MultiGaussNet(response_id=response_id, **multigauss_kwargs)
    GN.fit(X, Df)
    
    # Configure R glmnet
    r_kwargs = {'family': 'mgaussian', 'alpha': alpha, 'nlambda': nlambda}
    if use_offset:
        r_kwargs['offset'] = numpy_to_r_matrix(O)
    if use_weights:
        W_numeric = W.astype(float)
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gn = glmnet.glmnet(numpy_to_r_matrix(X), numpy_to_r_matrix(Y), **r_kwargs)
    r_coef = ro.r.coef(r_gn)
    
    # Extract coefficients for each response
    C1 = np.array(ro.r['as.matrix'](r_coef.rx2('y1')))
    C2 = np.array(ro.r['as.matrix'](r_coef.rx2('y2')))
    C3 = np.array(ro.r['as.matrix'](r_coef.rx2('y3')))
    
    C = np.array([C1, C2, C3]).T
    
    # Compare results
    assert np.allclose(C[:, 1:], GN.coefs_)
    assert np.allclose(C[:, 0], GN.intercepts_)


def test_cross_validation(sample_data, alpha, alignment, use_offset, use_weights):
    """Test cross-validation with different alpha, alignment, offset, and weight combinations."""
    X, Y, O, W, Df, response_id, offset_id, nlambda = sample_data
    
    # Configure Python MultiGaussNet with CV
    multigauss_kwargs = {'response_id': response_id, 'alpha': alpha}
    if use_offset:
        multigauss_kwargs['offset_id'] = offset_id
    if use_weights:
        multigauss_kwargs['weight_id'] = 'weight'
    
    GN = MultiGaussNet(**multigauss_kwargs)
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
        'family': 'mgaussian',
        'alignment': alignment,
        'grouped': True,
        'alpha': alpha
    }
    if use_offset:
        r_kwargs['offset'] = numpy_to_r_matrix(O)
    if use_weights:
        W_numeric = W.astype(float)
        r_kwargs['weights'] = FloatVector(W_numeric)
    
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(X), numpy_to_r_matrix(Y), **r_kwargs)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results
    assert np.allclose(GN.score_path_.scores['Mean Squared Error'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN.score_path_.scores['SD(Mean Squared Error)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3) 