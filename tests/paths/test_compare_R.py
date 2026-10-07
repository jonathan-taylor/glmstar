"""
Compare the C++ path estimators with R's glmnet across families and options.

For each family (gaussian, binomial, poisson, multinomial, mgaussian) and each
configuration in a grid of options, the fitted path must agree with R's
glmnet: same number of lambdas, and lambda, coefficients, intercepts and
fraction of deviance explained to 1e-8. Cross-validation (with fixed folds)
must agree with R's cv.glmnet for every measure at every lambda.

With --test-size=small one simulated dataset is used per case; otherwise three.
"""
import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from sklearn.model_selection import PredefinedSplit

from glmnet import GaussNet, LogNet, FishNet, MultiClassNet, MultiGaussNet
from glmnet.scoring import (binomial_deviance_scorer,
                            multinomial_deviance_scorer,
                            BinomialDevianceScorer,
                            MultinomialDevianceScorer,
                            PROB_MIN)

ESTIMATORS = {'gaussian': GaussNet,
              'binomial': LogNet,
              'poisson': FishNet,
              'multinomial': MultiClassNet,
              'mgaussian': MultiGaussNet}
MULTI = ['multinomial', 'mgaussian']
TOL = 1e-8

# configurations: options passed to both the Python estimator and R's glmnet

PATH_CONFIGS = [dict(),
                dict(weights='positive'),
                dict(weights='zeros'),
                dict(offset=True),
                dict(alpha=0.5),
                dict(alpha=0.05),
                dict(df_max=3, nvars=50),
                dict(exclude=[1, 2]),
                dict(penalty_factor=True),
                dict(lower_limits=0),
                dict(upper_limits=0.1),
                dict(lambda_values=True),
                dict(nlambda=20),
                dict(lambda_min_ratio=0.05),
                dict(nobs=50, nvars=200),
                dict(sparse=True),
                dict(sparse=True, weights='positive', alpha=0.5),
                dict(sparse=True, nobs=50, nvars=200),
                dict(standardize=False),
                dict(fit_intercept=False)]

FAMILY_PATH_CONFIGS = {'gaussian': [dict(covariance=False),
                                    dict(covariance=False, nobs=50, nvars=200)],
                       'binomial': [dict(modified_newton=True)],
                       'poisson': [],
                       'multinomial': [dict(grouped=True),
                                       dict(grouped=True, alpha=0.5),
                                       dict(grouped=True, sparse=True)],
                       'mgaussian': [dict(standardize_response=True)]}

CV_CONFIGS = [dict(),
              dict(weights='positive'),
              dict(offset=True),
              dict(alignment='fraction'),
              dict(weights='positive', offset=True, alignment='fraction'),
              dict(standardize=False),
              dict(fit_intercept=False),
              dict(nobs=60, nvars=150)]

# Python CV score name -> R type.measure
CV_MEASURES = {'gaussian': {'Mean Squared Error': 'mse',
                            'Mean Absolute Error': 'mae',
                            'Gaussian Deviance': 'deviance'},
               'binomial': {'Binomial Deviance': 'deviance',
                            'Accuracy': 'class',
                            'AUC': 'auc',
                            'Mean Squared Error': 'mse',
                            'Mean Absolute Error': 'mae'},
               'poisson': {'Poisson Deviance': 'deviance',
                           'Mean Squared Error': 'mse',
                           'Mean Absolute Error': 'mae'},
               'multinomial': {'Multinomial Deviance': 'deviance',
                               'Misclassification Error': 'class',
                               'Mean Squared Error': 'mse',
                               'Mean Absolute Error': 'mae'},
               'mgaussian': {'Mean Squared Error': 'mse',
                             'Mean Absolute Error': 'mae'}}

def _label(config):
    return '-'.join(f'{k}={v}' for k, v in config.items()) or 'default'

def _path_cases():
    for family in ESTIMATORS:
        for config in PATH_CONFIGS + FAMILY_PATH_CONFIGS[family]:
            yield pytest.param(family, config, id=f'{family}-{_label(config)}')

def _cv_cases():
    for family in ESTIMATORS:
        for config in CV_CONFIGS:
            yield pytest.param(family, config, id=f'{family}-{_label(config)}')

def _seeds(request):
    return [0] if request.config.getoption('test_size') == 'small' else [0, 1, 2]

# data

def simulate(family, nobs, nvars, rng):
    X = rng.standard_normal((nobs, nvars))
    beta = np.zeros(nvars)
    beta[:4] = [1, -1, 0.5, 0.5]
    eta = 0.6 * X @ beta
    if family == 'gaussian':
        y = eta + rng.standard_normal(nobs)
    elif family == 'binomial':
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    elif family == 'poisson':
        y = rng.poisson(np.exp(eta + 0.5)).astype(float)
    elif family == 'multinomial':
        P = np.exp(np.column_stack([eta, -eta, 0 * eta]))
        P /= P.sum(1, keepdims=True)
        y = np.array([rng.choice(3, p=p_) for p_ in P]).astype(float)
    else:
        y = np.column_stack([eta, -eta]) + rng.standard_normal((nobs, 2))
    return X, y

def setup(Rinfo,
          family,
          seed,
          nobs=100,
          nvars=10,
          weights=None,
          offset=False,
          alpha=None,
          df_max=None,
          exclude=None,
          penalty_factor=False,
          lower_limits=None,
          upper_limits=None,
          lambda_values=False,
          nlambda=None,
          lambda_min_ratio=None,
          sparse=False,
          standardize=True,
          fit_intercept=True,
          covariance=None,
          modified_newton=False,
          grouped=False,
          standardize_response=False,
          alignment=None):
    """Simulate data; return (X, D, estimator kwargs, R glmnet argument string).

    Variables X, y and any of W, O, PF, EX, LAM are assigned in R.
    """
    rpy = Rinfo['rpy']
    rng = np.random.default_rng(seed)
    X, y = simulate(family, nobs, nvars, rng)

    if family == 'mgaussian':
        D = pd.DataFrame(y, columns=['y1', 'y2'])
        kw = {'response_id': ['y1', 'y2']}
    else:
        D = pd.DataFrame({'y': y})
        kw = {'response_id': 'y'}
    kw.update(standardize=standardize, fit_intercept=fit_intercept)
    R = {'family': f'"{family}"',
         'standardize': str(standardize).upper(),
         'intercept': str(fit_intercept).upper()}
    assign = {'X': X, 'y': y}

    if weights is not None:
        W = rng.uniform(0.5, 2, nobs)
        if weights == 'zeros':
            W[rng.choice(nobs, 10, replace=False)] = 0
        D['weight'] = W
        kw['weight_id'] = 'weight'
        assign['W'] = W
        R['weights'] = 'W'
    if offset:
        if family in MULTI:
            ncol = 2
            O = 0.3 * rng.standard_normal((nobs, ncol))
            if family == 'multinomial':
                O = np.column_stack([O, np.zeros(nobs)])
                ncol = 3
            cols = [f'offset{i}' for i in range(ncol)]
            for i, c in enumerate(cols):
                D[c] = O[:, i]
            kw['offset_id'] = cols
        else:
            O = 0.3 * rng.standard_normal(nobs)
            D['offset'] = O
            kw['offset_id'] = 'offset'
        assign['O'] = O
        R['offset'] = 'O'
    if alpha is not None:
        kw['alpha'] = alpha
        R['alpha'] = alpha
    if df_max is not None:
        kw['df_max'] = df_max
        R['dfmax'] = df_max
    if exclude is not None:
        kw['exclude'] = list(exclude)
        assign['EX'] = np.asarray(exclude) + 1.
        R['exclude'] = 'EX'
    if penalty_factor:
        PF = rng.uniform(0.2, 2, nvars)
        PF[0] = 0
        kw['penalty_factor'] = PF
        assign['PF'] = PF
        R['penalty.factor'] = 'PF'
    if lower_limits is not None:
        kw['lower_limits'] = lower_limits
        R['lower.limits'] = lower_limits
    if upper_limits is not None:
        kw['upper_limits'] = upper_limits
        R['upper.limits'] = upper_limits
    if lambda_values:
        lam = np.exp(np.linspace(np.log(0.3), np.log(0.003), 25))
        kw['lambda_values'] = lam
        assign['LAM'] = lam
        R['lambda'] = 'LAM'
    if nlambda is not None:
        kw['nlambda'] = nlambda
        R['nlambda'] = nlambda
    if lambda_min_ratio is not None:
        kw['lambda_min_ratio'] = lambda_min_ratio
        R['lambda.min.ratio'] = lambda_min_ratio
    if covariance is not None:
        kw['covariance'] = covariance
        R['type.gaussian'] = '"covariance"' if covariance else '"naive"'
    if modified_newton:
        kw['modified_newton'] = True
        R['type.logistic'] = '"modified.Newton"'
    if grouped:
        kw['grouped'] = True
        R['type.multinomial'] = '"grouped"'
    if standardize_response:
        kw['standardize_response'] = True
        R['standardize.response'] = 'TRUE'
    if alignment is not None:
        R['alignment'] = f'"{alignment}"'

    with Rinfo['np_cv_rules'].context():
        for k, v in assign.items():
            rpy.r.assign(k, v)
    for k, v in assign.items():
        # 1-d numpy arrays arrive as 1-d R arrays (with a dim attribute)
        if np.ndim(v) == 1:
            rpy.r(f'{k} <- as.vector({k})')
    rpy.r('suppressMessages(library(glmnet)); X <- as.matrix(X)')
    if sparse:
        rpy.r('X <- Matrix::Matrix(X, sparse=TRUE)')
        X = scipy.sparse.csc_matrix(X)

    return X, D, kw, ', '.join(f'{k}={v}' for k, v in R.items())

def _get(Rinfo, expr):
    with Rinfo['np_cv_rules'].context():
        return np.asarray(Rinfo['rpy'].r(expr))

def _rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1)

# tests

@pytest.mark.parametrize('family, config', list(_path_cases()))
def test_path(Rinfo, request, family, config):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    for seed in _seeds(request):
        X, D, kw, R_args = setup(Rinfo, family, seed, **config)
        L = ESTIMATORS[family](**kw).fit(X, D)

        Rinfo['rpy'].r(f'G <- glmnet(X, y, {R_args})')
        R_lambda = _get(Rinfo, 'G$lambda')
        R_dev = _get(Rinfo, 'G$dev.ratio')
        if family in MULTI:
            K = int(_get(Rinfo, 'length(G$beta)')[0])
            R_coefs = np.stack([_get(Rinfo, f'as.matrix(G$beta[[{k+1}]])').T for k in range(K)], axis=-1)
            R_intercepts = _get(Rinfo, 'as.matrix(G$a0)').T
        else:
            R_coefs = _get(Rinfo, 'as.matrix(G$beta)').T
            R_intercepts = _get(Rinfo, 'G$a0')

        assert L.lambda_values_.shape == R_lambda.shape, f'seed {seed}: path length differs'
        assert np.allclose(L.lambda_values_, R_lambda, rtol=TOL, atol=0), f'seed {seed}: lambda'
        assert _rel(L.coefs_, R_coefs) < TOL, f'seed {seed}: coefficients'
        assert _rel(np.asarray(L.intercepts_), R_intercepts) < TOL, f'seed {seed}: intercepts'
        assert np.allclose(L.summary_['Fraction Deviance Explained'], R_dev, rtol=0, atol=TOL), \
            f'seed {seed}: fraction deviance explained'

@pytest.mark.parametrize('family, config', list(_cv_cases()))
def test_cross_validation(Rinfo, request, family, config):

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    config = dict(config)
    alignment = config.pop('alignment', 'lambda')

    for seed in _seeds(request):
        X, D, kw, R_args = setup(Rinfo, family, seed, alignment=alignment, **config)
        n = X.shape[0]
        foldid = np.random.default_rng(seed).permutation(np.arange(n) % 5 + 1)
        with Rinfo['np_cv_rules'].context():
            Rinfo['rpy'].r.assign('foldid', foldid.astype(float))
        Rinfo['rpy'].r('foldid <- as.integer(as.vector(foldid))')

        L = ESTIMATORS[family](**kw).fit(X, D)
        L.cross_validation_path(X, D, cv=PredefinedSplit(foldid), alignment=alignment)
        scores = L.score_path_.scores

        for name, measure in CV_MEASURES[family].items():
            Rinfo['rpy'].r(f'CV <- cv.glmnet(X, y, type.measure="{measure}", foldid=foldid, {R_args})')
            R_cvm, R_cvsd = _get(Rinfo, 'CV$cvm'), _get(Rinfo, 'CV$cvsd')
            cvm, cvsd = scores[name].values, scores[f'SD({name})'].values
            if name == 'Accuracy':
                cvm = 1 - cvm         # R reports misclassification error
            if family == 'binomial' and measure in ['mse', 'mae']:
                # R sums over both classes, giving twice Python's (y - p)^2 and |y - p|
                R_cvm, R_cvsd = R_cvm / 2, R_cvsd / 2

            assert cvm.shape == R_cvm.shape, f'seed {seed}, {name}: number of lambdas'
            assert np.allclose(cvm, R_cvm, rtol=TOL, atol=1e-12), f'seed {seed}, {name}: mean'
            assert np.allclose(cvsd, R_cvsd, rtol=TOL, atol=1e-12), f'seed {seed}, {name}: SD'

# probability clamping in the deviance scorers

def _saturated(family, nobs=60, nvars=150, seed=0):
    rng = np.random.default_rng(seed)
    X, y = simulate(family, nobs, nvars, rng)
    D = pd.DataFrame({'y': y})
    foldid = rng.permutation(np.arange(nobs) % 5 + 1)
    return X, D, foldid

@pytest.mark.parametrize('family, make_scorer, cls',
                         [('binomial', binomial_deviance_scorer, BinomialDevianceScorer),
                          ('multinomial', multinomial_deviance_scorer, MultinomialDevianceScorer)])
def test_deviance_clamping(family, make_scorer, cls):

    default = make_scorer()
    assert isinstance(default, cls)
    assert default.prob_min == PROB_MIN
    assert make_scorer() == default            # deduplicated against the default scorers
    unclamped = make_scorer(prob_min=None)
    assert unclamped.name.endswith('(Unclamped)')
    assert make_scorer(prob_min=1e-3).name != default.name

    # p > n: the end of the path is saturated, with probabilities beyond 1e-5
    X, D, foldid = _saturated(family)
    L = ESTIMATORS[family](response_id='y').fit(X, D)
    L.cross_validation_path(X, D, cv=PredefinedSplit(foldid), scorers=[unclamped])
    clamped_scores = L.score_path_.scores[default.name].values
    unclamped_scores = L.score_path_.scores[unclamped.name].values

    # clamping can only lower the deviance; it should matter at the saturated end
    assert np.all(clamped_scores <= unclamped_scores + 1e-12)
    assert np.any(unclamped_scores - clamped_scores > 1e-6)
    # and not at the start of the path
    assert np.allclose(clamped_scores[:10], unclamped_scores[:10])

def test_deviance_scorer_values():
    # binomial: matches the deviance formula, with and without clamping
    y = np.array([0., 1., 1., 0., 1.])
    p = np.array([1e-8, 0.9, 1 - 1e-9, 0.3, 0.5])
    w = np.array([1., 2., 1., 1., 0.5])
    split = np.arange(5)

    def binom_dev(p):
        return np.average(-2 * (y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w)

    val, wsum = binomial_deviance_scorer(prob_min=None).score_fn(split, y, p, w)
    assert np.isclose(val, binom_dev(p)) and wsum == w.sum()
    val, _ = binomial_deviance_scorer().score_fn(split, y, p, w)
    assert np.isclose(val, binom_dev(np.clip(p, PROB_MIN, 1 - PROB_MIN)))

    # multinomial: one-hot y, clamped without renormalizing (as R does)
    Y = np.eye(3)[[0, 2, 1, 1]]
    P = np.array([[1 - 2e-7, 1e-7, 1e-7], [0.2, 0.3, 0.5], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1]])
    wm = np.array([1., 1., 2., 1.])
    val, _ = multinomial_deviance_scorer().score_fn(np.arange(4), Y, P, wm)
    Pc = np.clip(P, PROB_MIN, 1 - PROB_MIN)
    assert np.isclose(val, np.average(-2 * np.log((Y * Pc).sum(1)), weights=wm))

def test_mgaussian_offset_shape():
    # as in R, a multi-response offset must have one column per response
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    D = pd.DataFrame(rng.standard_normal((50, 2)), columns=['y1', 'y2'])
    D['offset'] = rng.standard_normal(50)
    with pytest.raises(ValueError, match='same shape as the response'):
        MultiGaussNet(response_id=['y1', 'y2'], offset_id='offset').fit(X, D)

# binomial responses given as proportions

@pytest.mark.parametrize('form', ['proportion', 'trials_successes'])
@pytest.mark.parametrize('extra_weights', [False, True])
def test_lognet_proportions(Rinfo, request, form, extra_weights):
    # proportions with observation weights (e.g. numbers of trials) correspond
    # to R's two-column matrix of proportions with weights

    if not Rinfo.get('has_rpy2'):
        pytest.skip('requires rpy2')

    rpy = Rinfo['rpy']
    for seed in _seeds(request):
        rng = np.random.default_rng(seed)
        nobs = 120
        X = rng.standard_normal((nobs, 8))
        trials = rng.integers(1, 12, nobs).astype(float)
        successes = rng.binomial(trials.astype(int), 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1]))))
        prop = successes / trials
        W = trials * (rng.uniform(0.5, 2, nobs) if extra_weights else 1)
        foldid = rng.permutation(np.arange(nobs) % 5 + 1)

        if form == 'proportion':
            D = pd.DataFrame({'prop': prop, 'weight': W})
            kw = dict(response_id='prop', weight_id='weight')
        else:
            # (trials, successes): the weight multiplies the number of trials
            D = pd.DataFrame({'trials': trials, 'successes': successes, 'weight': W / trials})
            kw = dict(response_id=['trials', 'successes'], weight_id='weight')

        L = LogNet(**kw).fit(X, D)
        L.cross_validation_path(X, D, cv=PredefinedSplit(foldid))
        scores = L.score_path_.scores

        with Rinfo['np_cv_rules'].context():
            for k, v in dict(X=X, P=np.column_stack([1 - prop, prop]), W=W, foldid=foldid.astype(float)).items():
                rpy.r.assign(k, v)
        rpy.r('suppressMessages(library(glmnet)); W <- as.vector(W); foldid <- as.integer(as.vector(foldid))')
        rpy.r('G <- glmnet(X, P, family="binomial", weights=W)')

        R_lambda = _get(Rinfo, 'G$lambda')
        assert L.lambda_values_.shape == R_lambda.shape
        assert np.allclose(L.lambda_values_, R_lambda, rtol=TOL, atol=0)
        assert _rel(L.coefs_, _get(Rinfo, 'as.matrix(G$beta)').T) < TOL
        assert _rel(L.intercepts_, _get(Rinfo, 'G$a0')) < TOL
        assert np.allclose(L.summary_['Fraction Deviance Explained'], _get(Rinfo, 'G$dev.ratio'), rtol=0, atol=TOL)

        for name, measure in CV_MEASURES['binomial'].items():
            rpy.r(f'CV <- cv.glmnet(X, P, family="binomial", weights=W, foldid=foldid, type.measure="{measure}")')
            R_cvm, R_cvsd = _get(Rinfo, 'CV$cvm'), _get(Rinfo, 'CV$cvsd')
            cvm, cvsd = scores[name].values, scores[f'SD({name})'].values
            if name == 'Accuracy':
                cvm = 1 - cvm
            if measure in ['mse', 'mae']:
                R_cvm, R_cvsd = R_cvm / 2, R_cvsd / 2
            assert np.allclose(cvm, R_cvm, rtol=TOL, atol=1e-12), f'seed {seed}, {name}: mean'
            assert np.allclose(cvsd, R_cvsd, rtol=TOL, atol=1e-12), f'seed {seed}, {name}: SD'

def test_lognet_response_forms():
    rng = np.random.default_rng(0)
    nobs = 80
    X = rng.standard_normal((nobs, 5))
    trials = rng.integers(1, 10, nobs).astype(float)
    successes = rng.binomial(trials.astype(int), 0.4)

    # proportions + weights and (trials, successes) give the same path
    L1 = LogNet(response_id='prop', weight_id='w').fit(X, pd.DataFrame({'prop': successes / trials, 'w': trials}))
    L2 = LogNet(response_id=['t', 's']).fit(X, pd.DataFrame({'t': trials, 's': successes}))
    assert np.allclose(L1.coefs_, L2.coefs_, rtol=1e-12, atol=1e-14)
    assert np.allclose(L1.intercepts_, L2.intercepts_, rtol=1e-12, atol=1e-14)
    assert list(L1.classes_) == [0, 1]

    # numeric 0/1 responses are still labels; other two-valued responses too
    y = rng.integers(0, 2, nobs)
    L3 = LogNet(response_id='y').fit(X, pd.DataFrame({'y': y.astype(float)}))
    L4 = LogNet(response_id='y').fit(X, pd.DataFrame({'y': np.where(y == 1, 'b', 'a')}))
    assert np.allclose(L3.coefs_, L4.coefs_)
    assert list(L4.classes_) == ['a', 'b']

    # values outside [0, 1] with more than two levels are not proportions
    with pytest.raises(ValueError, match='binary'):
        LogNet(response_id='y').fit(X, pd.DataFrame({'y': rng.integers(0, 3, nobs) * 0.75}))

# failed folds in scoring

def test_failed_fold_scores_are_missing():
    from glmnet.scoring import Scorer
    from glmnet.scorer import PathScorer
    from glmnet.family import GLMFamilySpec

    def picky(y, yhat, sample_weight):
        if y.shape[0] == 4:              # fails only on the first fold
            raise ValueError('cannot score this fold')
        return np.average((y - yhat)**2, weights=sample_weight)

    scorer = Scorer(name='Picky', score=picky, maximize=False)
    y = np.arange(10.)
    shift = np.where(y < 7, 1., 3.)      # squared errors 1 (fold 2), 9 (fold 3) at the first lambda
    predictions = np.column_stack([y + shift, y + 2 * shift])
    splits = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
    with pytest.warns(UserWarning, match='failed on fold 0'):
        scores = PathScorer(data=(y, y),
                            predictions=predictions,
                            sample_weight=np.ones(10),
                            splits=splits,
                            index=np.array([1., 0.5]),
                            family=GLMFamilySpec()).compute_scores(scorers=[scorer])[0]
    # the failed fold is excluded: the mean is over the two remaining folds
    assert np.allclose(scores['Picky'], [(1 + 9) / 2, (4 + 36) / 2])
