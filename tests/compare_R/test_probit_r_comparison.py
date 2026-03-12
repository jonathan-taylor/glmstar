"""
Test probit model comparison with R glmnet using rpy2.

This test file converts the original IPython R magic cells to use rpy2
for proper R integration in pytest.
"""

import pytest
import numpy as np
import pandas as pd
from glmnet.glm import BinomialGLM
from glmnet import GLM, GLMNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from glmnet.glm import GLMControl
from glmnet.glmnet import GLMNetControl


def numpy_to_r_matrix(Rinfo, X):
    """Convert numpy array to R matrix with proper row/column major ordering."""
    ro = Rinfo['rpy']
    FloatVector = Rinfo['FloatVector']
    return ro.r.matrix(FloatVector(X.T.flatten()), nrow=X.shape[0], ncol=X.shape[1])


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n, p = 103, 20
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    R = rng.choice(['D', 'C'], size=n) 
    O = rng.standard_normal(n) * 0.2
    W = rng.integers(2, 6, size=n)
    W[:20] = 3
    W = W / W.mean()
    L = LabelEncoder().fit(R)
    Y = R == L.classes_[1] if L.classes_ is not None else (R == 'C')
    D = np.array([Y, O, W]).T
    Df = pd.DataFrame(D, columns=['binary', 'offset', 'weight'])
    Df['response'] = R
    
    # Create probit family
    probit = sm.families.Binomial(link=sm.families.links.Probit())
    
    return X, Df, probit, Y, O, W


def test_glm_comparison(Rinfo, sample_data):
    """Test GLM comparison with R glm."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLM
    G2 = BinomialGLM(response_id='response',
                     offset_id='offset',
                     weight_id='weight', 
                     family=probit, 
                     summarize=True)
    G2.fit(X, Df)
    
    # R glm
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    W_numeric = W.astype(float)
    
    # Create individual columns for each predictor
    r_data_dict = {
        'Y': FloatVector(Y_numeric),
        'O': FloatVector(O_numeric),
        'W': FloatVector(W_numeric)
    }
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, 
                       weights=FloatVector(W_numeric), 
                       offset=FloatVector(O_numeric), 
                       family=stats.binomial(link='probit'))
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G2.coef_, r_coef[1:], rtol=1e-4, atol=1e-4)
    assert np.allclose(G2.intercept_, r_coef[0], rtol=1e-4, atol=1e-4)


def test_glm_no_weights_no_offset(Rinfo, sample_data):
    """Test GLM without weights or offset."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLM
    control = GLMControl(epsnr=1e-8)
    G4 = GLM(response_id='binary', family=probit, control=control)
    G4.fit(X, Df)
    
    # R glm
    Y_numeric = Y.astype(float)
    
    # Create individual columns for each predictor
    r_data_dict = {'Y': FloatVector(Y_numeric)}
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, 
                       family=stats.binomial(link='probit'),
                       control=stats.glm_control(epsilon=1e-10))
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G4.coef_, r_coef[1:], rtol=1e-4, atol=1e-4)
    assert np.allclose(G4.intercept_, r_coef[0], rtol=1e-4, atol=1e-4)


def test_glm_with_weights(Rinfo, sample_data):
    """Test GLM with weights."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLM
    control = GLMControl(epsnr=1e-8)
    G5 = GLM(response_id='binary', weight_id='weight', family=probit, control=control)
    G5.fit(X, Df)
    
    # R glm
    Y_numeric = Y.astype(float)
    W_numeric = W.astype(float)
    
    # Create individual columns for each predictor
    r_data_dict = {
        'Y': FloatVector(Y_numeric),
        'W': FloatVector(W_numeric)
    }
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, 
                       weights=FloatVector(W_numeric),
                       family=stats.binomial(link='probit'),
                       control=stats.glm_control(epsilon=1e-10))
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G5.coef_, r_coef[1:], rtol=1e-4, atol=1e-4)
    assert np.allclose(G5.intercept_, r_coef[0], rtol=1e-4, atol=1e-4)


def test_glm_with_offset(Rinfo, sample_data):
    """Test GLM with offset."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLM
    control = GLMControl(epsnr=1e-8)
    G6 = GLM(response_id='binary', offset_id='offset', family=probit, control=control)
    G6.fit(X, Df)
    
    # R glm
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    
    # Create individual columns for each predictor
    r_data_dict = {
        'Y': FloatVector(Y_numeric),
        'O': FloatVector(O_numeric)
    }
    for i in range(X.shape[1]):
        r_data_dict[f'X{i+1}'] = FloatVector(X[:, i])
    
    r_data = ro.r['data.frame'](**r_data_dict)
    
    # Create formula with individual predictors
    formula_str = 'Y ~ ' + ' + '.join([f'X{i+1}' for i in range(X.shape[1])])
    r_model = stats.glm(ro.Formula(formula_str), data=r_data, 
                       offset=FloatVector(O_numeric),
                       family=stats.binomial(link='probit'),
                       control=stats.glm_control(epsilon=1e-10))
    r_coef = np.array(stats.coef(r_model))
    
    assert np.allclose(G6.coef_, r_coef[1:], rtol=1e-4, atol=1e-4)
    assert np.allclose(G6.intercept_, r_coef[0], rtol=1e-4, atol=1e-4)


def test_glmnet_comparison(Rinfo, sample_data):
    """Test GLMNet comparison with R glmnet."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLMNet
    GNcontrol = GLMNetControl()
    GN = GLMNet(response_id='binary',
                offset_id='offset',
                weight_id='weight',
                family=probit,
                control=GNcontrol).fit(X, Df)
    
    # R glmnet
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    W_numeric = W.astype(float)
    
    # Create probit family in R
    r_probit_family = stats.binomial(link='probit')
    
    r_gn = glmnet.glmnet(numpy_to_r_matrix(Rinfo, X), 
                        FloatVector(Y_numeric), 
                        offset=FloatVector(O_numeric), 
                        weights=FloatVector(W_numeric), 
                        family=r_probit_family)
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn)))
    r_lambda = np.array(r_gn.rx2('lambda'))
    
    # Compare results (using index 10 as in original)
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[10][1:], GN.coefs_[10], rtol=1e-4, atol=1e-4)


def test_glmnet_no_offset(Rinfo, sample_data):
    """Test GLMNet without offset."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLMNet
    GNcontrol = GLMNetControl()
    GN2 = GLMNet(response_id='binary',
                 weight_id='weight',
                 family=probit,
                 control=GNcontrol).fit(X, Df)
    
    # R glmnet
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    W_numeric = W.astype(float)
    
    # Create probit family in R
    r_probit_family = stats.binomial(link='probit')
    
    r_gn2 = glmnet.glmnet(numpy_to_r_matrix(Rinfo, X), 
                          FloatVector(Y_numeric), 
                          weights=FloatVector(W_numeric), 
                          family=r_probit_family)
    r_coef = np.array(ro.r['as.matrix'](ro.r.coef(r_gn2)))
    r_lambda = np.array(r_gn2.rx2('lambda'))
    
    # Compare results (using first 50 as in original)
    if r_coef.ndim == 1:
        r_coef = r_coef.reshape(1, -1)
    
    assert np.allclose(r_coef.T[:50, 1:], GN2.coefs_[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(r_coef.T[:50, 0], GN2.intercepts_[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(r_lambda[:50], GN2.lambda_values_[:50])


def test_cross_validation_fraction_alignment(Rinfo, sample_data):
    """Test cross-validation with fraction alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLMNet with CV
    GNcontrol = GLMNetControl()
    GN3 = GLMNet(response_id='binary',
                 offset_id='offset', 
                 family=probit,
                 control=GNcontrol).fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    W_numeric = W.astype(float)
    
    # Create probit family in R
    r_probit_family = stats.binomial(link='probit')
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             FloatVector(Y_numeric), 
                             offset=FloatVector(O_numeric),
                             foldid=r_foldid,
                             family=r_probit_family,
                             alignment='fraction',
                             grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN3.score_path_.scores['Binomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.score_path_.scores['SD(Binomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_lambda_alignment(Rinfo, sample_data):
    """Test cross-validation with lambda alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLMNet with CV
    GNcontrol = GLMNetControl()
    GN3 = GLMNet(response_id='binary',
                 offset_id='offset',
                 family=probit,
                 control=GNcontrol).fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN3.cross_validation_path(X, Df, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    W_numeric = W.astype(float)
    
    # Create probit family in R
    r_probit_family = stats.binomial(link='probit')
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             FloatVector(Y_numeric), 
                             offset=FloatVector(O_numeric),
                             foldid=r_foldid,
                             family=r_probit_family,
                             alignment='lambda',
                             grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN3.score_path_.scores['Binomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN3.score_path_.scores['SD(Binomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_with_weights_fraction(Rinfo, sample_data):
    """Test cross-validation with weights using fraction alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLMNet with CV
    GNcontrol = GLMNetControl()
    GN4 = GLMNet(response_id='binary',
                 offset_id='offset',
                 weight_id='weight',
                 family=probit,
                 control=GNcontrol).fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='fraction')
    
    # R cv.glmnet
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    W_numeric = W.astype(float)
    
    # Create probit family in R
    r_probit_family = stats.binomial(link='probit')
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             FloatVector(Y_numeric), 
                             offset=FloatVector(O_numeric),
                             weights=FloatVector(W_numeric),
                             foldid=r_foldid,
                             alignment='fraction',
                             family=r_probit_family,
                             grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN4.score_path_.scores['Binomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.score_path_.scores['SD(Binomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3)


def test_cross_validation_with_weights_lambda(Rinfo, sample_data):
    """Test cross-validation with weights using lambda alignment."""

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')
    ro = Rinfo['rpy']
    importr = Rinfo['importr']
    FloatVector = Rinfo['FloatVector']
    IntVector = Rinfo['IntVector']
    numpy2ri = Rinfo['numpy2ri']
    glmnet = importr('glmnet')
    stats = importr('stats')
    survival = importr('survival')
    base = importr('base')
    X, Df, probit, Y, O, W = sample_data
    
    # Python GLMNet with CV
    GNcontrol = GLMNetControl()
    GN4 = GLMNet(response_id='binary',
                 offset_id='offset',
                 weight_id='weight',
                 family=probit,
                 control=GNcontrol).fit(X, Df)
    
    # Create fold IDs
    cv = KFold(5, random_state=0, shuffle=True)
    foldid = np.empty(X.shape[0])
    for i, (train, test) in enumerate(cv.split(np.arange(X.shape[0]))):
        foldid[test] = i + 1
    
    # Use the correct cross-validation method
    predictions, scores = GN4.cross_validation_path(X, Df, cv=cv, alignment='lambda')
    
    # R cv.glmnet
    Y_numeric = Y.astype(float)
    O_numeric = O.astype(float)
    W_numeric = W.astype(float)
    
    # Create probit family in R
    r_probit_family = stats.binomial(link='probit')
    
    r_foldid = IntVector(foldid.astype(int))
    r_gcv = glmnet.cv_glmnet(numpy_to_r_matrix(Rinfo, X),
                             FloatVector(Y_numeric), 
                             offset=FloatVector(O_numeric),
                             weights=FloatVector(W_numeric),
                             foldid=r_foldid,
                             alignment='lambda',
                             family=r_probit_family,
                             grouped=True)
    
    r_cvm = np.array(r_gcv.rx2('cvm'))
    r_cvsd = np.array(r_gcv.rx2('cvsd'))
    
    # Compare results (using first 50 as in original)
    assert np.allclose(GN4.score_path_.scores['Binomial Deviance'].iloc[:50], r_cvm[:50], rtol=1e-3, atol=1e-3)
    assert np.allclose(GN4.score_path_.scores['SD(Binomial Deviance)'].iloc[:50], r_cvsd[:50], rtol=1e-3, atol=1e-3) 
