import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from glmnet import MultiClassNet
from glmnet.data import make_dataset

@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_dense_vs_sparse_path_multiclassnet(standardize, fit_intercept, n=100, p=10, n_classes=3):
    """
    Compare path fits for dense vs. sparse X for MultiClassNet.
    """
    # Use make_dataset for consistent test data
    X, y, coef, intercept = make_dataset(MultiClassNet, n_samples=n, n_features=p, 
                                        n_informative=3, n_classes=n_classes, snr=2.0, random_state=42)
    df = pd.DataFrame({'response': y})
    
    # Dense fit
    net_dense = MultiClassNet(response_id='response',
                             fit_intercept=fit_intercept,
                             standardize=standardize)
    net_dense.fit(X, df)
    
    # Sparse fit
    X_sparse = scipy.sparse.csc_array(X)
    net_sparse = MultiClassNet(response_id='response',
                              fit_intercept=fit_intercept,
                              standardize=standardize)
    net_sparse.fit(X_sparse, df)
    
    # Compare paths (for each lambda)
    for coef_dense, coef_sparse in zip(net_dense.coefs_, net_sparse.coefs_):
        assert np.allclose(coef_dense, coef_sparse, atol=1e-5, rtol=1e-4)
    assert np.allclose(net_dense.intercepts_, net_sparse.intercepts_, atol=1e-5) 