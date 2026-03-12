"""
Test MultiClassNet comparison with R glmnet using rpy2.

This test file converts the original IPython R magic cells to use rpy2
for proper R integration in pytest.
"""

import pytest
import string
import numpy as np
import pandas as pd
from glmnet import MultiClassNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm


def numpy_to_r_matrix(Rinfo, X):
    rpy = Rinfo['rpy']
    FloatVector = Rinfo['FloatVector']
    return rpy.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n, p, q, nlambda = 103, 17, 3, 100
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    O = rng.standard_normal((n, q)) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    labels = list(string.ascii_uppercase[:q])
    Y = rng.choice(labels, size=n)
    R = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape((-1,1)))
    response_id = ['response']
    offset_id = [f'offset[{i}]' for i in range(q)]
    Df = pd.DataFrame({'response':Y, 'weight':W})
    for i, l in enumerate(offset_id):
        Df[l] = O[:,i]
    return X, Y, O, W, R, Df, response_id, offset_id, nlambda


def test_multiclassnet_comparison(Rinfo, sample_data, use_offset, use_weights):
    """Test MultiClassNet comparison with various configurations."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    alpha = 1
    X, Y, O, W, R, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiClassNet
    kwargs = {'response_id': 'response', 'nlambda': nlambda, 'alpha': alpha}
    if use_offset:
        kwargs['offset_id'] = offset_id
    if use_weights:
        kwargs['weight_id'] = 'weight'
    
    GN2 = MultiClassNet(**kwargs)
    GN2.fit(X, Df)
    
    # R glmnet
    # Convert response to factor for multinomial regression
    response_factor = rpy.r.factor(rpy.StrVector(Df['response']))
    
    r_kwargs = {
        'x': numpy_to_r_matrix(Rinfo, X),
        'y': response_factor,
        'family': 'multinomial',
        'nlambda': nlambda,
        'alpha': alpha
    }
    
    if use_offset:
        r_kwargs['offset'] = numpy_to_r_matrix(Rinfo, O)
    if use_weights:
        r_kwargs['weights'] = FloatVector(W.astype(float))
    
    r_gn2 = glmnet.glmnet(**r_kwargs)
    r_coef = rpy.r.coef(r_gn2)
    
    # Extract coefficients for each class
    C1 = np.array(rpy.r['as.matrix'](r_coef.rx2('A')))
    C2 = np.array(rpy.r['as.matrix'](r_coef.rx2('B')))
    C3 = np.array(rpy.r['as.matrix'](r_coef.rx2('C')))
    
    C = np.array([C1, C2, C3]).T
    
    assert np.allclose(C[:, 1:], GN2.coefs_)
    assert np.allclose(C[:, 0], GN2.intercepts_)


def test_multiclassnet_cross_validation(Rinfo, sample_data, use_offset, use_weights, alignment):
    """Test MultiClassNet cross-validation with various configurations."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Y, O, W, R, Df, response_id, offset_id, nlambda = sample_data
    alpha = 1
    # Python MultiClassNet with CV
    kwargs = {'response_id': 'response', 'nlambda': nlambda, 'alpha': alpha}
    if use_offset:
        kwargs['offset_id'] = offset_id
    if use_weights:
        kwargs['weight_id'] = 'weight'
    
    GN3 = MultiClassNet(**kwargs)
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment=alignment)
    
    # R cv.glmnet
    # Convert response to factor for multinomial regression
    response_factor = rpy.r.factor(rpy.StrVector(Df['response']))
    
    r_foldid = IntVector(foldid.astype(int))
    r_kwargs = {
        'x': numpy_to_r_matrix(Rinfo, X),
        'y': response_factor,
        'foldid': r_foldid,
        'family': 'multinomial',
        'alignment': alignment,
        'nlambda': nlambda,
        'grouped': True
    }
    
    if use_offset:
        r_kwargs['offset'] = numpy_to_r_matrix(Rinfo, O)
    if use_weights:
        r_kwargs['weights'] = FloatVector(W.astype(float))
    
    r_gcv = glmnet.cv_glmnet(**r_kwargs)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN3.score_path_.scores['Multinomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_fraction_alignment(Rinfo, sample_data):
    """Test cross-validation with fraction alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Y, O, W, R, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiClassNet with CV
    GN3 = MultiClassNet(response_id='response', offset_id=offset_id, nlambda=nlambda)
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    # Convert response to factor for multinomial regression
    response_factor = rpy.r.factor(rpy.StrVector(Df['response']))
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             response_factor, offset=numpy_to_r_matrix(Rinfo, O), foldid=r_foldid,
                             family='multinomial', alignment='fraction', nlambda=nlambda, grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN3.score_path_.scores['Multinomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.score_path_.scores['SD(Multinomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_lambda_alignment(Rinfo, sample_data):
    """Test cross-validation with lambda alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Y, O, W, R, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiClassNet with CV
    GN3 = MultiClassNet(response_id='response', offset_id=offset_id, nlambda=nlambda)
    GN3.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    # Convert response to factor for multinomial regression
    response_factor = rpy.r.factor(rpy.StrVector(Df['response']))
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             response_factor, offset=numpy_to_r_matrix(Rinfo, O), foldid=r_foldid,
                             family='multinomial', alignment='lambda', nlambda=nlambda, grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN3.score_path_.scores['Multinomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.score_path_.scores['SD(Multinomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_with_weights_fraction(Rinfo, sample_data):
    """Test cross-validation with weights using fraction alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Y, O, W, R, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiClassNet with CV
    GN4 = MultiClassNet(response_id='response', offset_id=offset_id, weight_id='weight', nlambda=nlambda)
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    W_numeric = W.astype(float)
    # Convert response to factor for multinomial regression
    response_factor = rpy.r.factor(rpy.StrVector(Df['response']))
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             response_factor, offset=numpy_to_r_matrix(Rinfo, O), weights=FloatVector(W_numeric),
                             foldid=r_foldid, family='multinomial', alignment='fraction', nlambda=nlambda, grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN4.score_path_.scores['Multinomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.score_path_.scores['SD(Multinomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_with_weights_lambda(Rinfo, sample_data):
    """Test cross-validation with weights using lambda alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    rpy = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Y, O, W, R, Df, response_id, offset_id, nlambda = sample_data
    
    # Python MultiClassNet with CV
    GN4 = MultiClassNet(response_id='response', offset_id=offset_id, weight_id='weight', nlambda=nlambda)
    GN4.fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    W_numeric = W.astype(float)
    # Convert response to factor for multinomial regression
    response_factor = rpy.r.factor(rpy.StrVector(Df['response']))
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             response_factor, offset=numpy_to_r_matrix(Rinfo, O), weights=FloatVector(W_numeric),
                             foldid=r_foldid, family='multinomial', alignment='lambda', nlambda=nlambda, grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN4.score_path_.scores['Multinomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.score_path_.scores['SD(Multinomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3) 
