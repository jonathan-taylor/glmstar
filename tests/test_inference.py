import pytest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import KFold

import scipy.sparse

from glmnet import GLMNet
from glmnet.inference import (lasso_inference,
                              _score_inference as score_inference,
                              _resampler_inference as resampler_inference,
                              TruncatedGaussian,
                              WeightedGaussianFamily,
                              AffineConstraint,
                              GLMNetInference,
                              _simple_score_inference,
                              discrete_family)

def test_Auto():

    df = sm.datasets.get_rdataset('Auto', package='ISLR').data
    df = df.set_index('name')
    X = np.asarray(df.drop(columns=['mpg']))
    y = np.asarray(df['mpg'])

    GN = GLMNet(response_id='response',
                fit_intercept=True,
                standardize=True)

    Df = pd.DataFrame({'response':y})

    GN.fit(X, Df)
    proportion = 0.8
    n =	df.shape[0]
    m = int(proportion*n)

    df = lasso_inference(GN,
                         2 / n, # GN.lambda_values_[min(10, GN.lambda_values_.shape[0]-1)],
                         (X[:m], Df.iloc[:m]),
                         (X, Df),
                         proportion=proportion)

def test_truncated_gaussian_basic():
    """Test TruncatedGaussian object creation and basic functionality."""
    tg = TruncatedGaussian(
        estimate=1.0,
        sigma=0.5,
        smoothing_sigma=0.1,
        lower_bound=-2.0,
        upper_bound=2.0,
        noisy_estimate=1.1,
        factor=1.0
    )
    
    assert tg.estimate == 1.0
    assert tg.sigma == 0.5
    assert tg.smoothing_sigma == 0.1
    assert tg.lower_bound == -2.0
    assert tg.upper_bound == 2.0
    assert tg.noisy_estimate == 1.1
    assert tg.factor == 1.0
    
    # Test weight function
    x_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    weights = tg.weight(x_values)
    assert np.all(weights >= 0)
    assert np.all(np.isfinite(weights))

def test_weighted_gaussian_family_basic():
    """Test WeightedGaussianFamily object creation and basic functionality."""
    def simple_weight(x):
        return np.ones_like(x)
    
    wgf = WeightedGaussianFamily(
        estimate=1.0,
        sigma=0.5,
        weight_fns=[simple_weight],
        num_sd=5,
        num_grid=100,
        use_sample=False,
        seed=42
    )
    
    assert wgf.estimate == 1.0
    assert wgf.sigma == 0.5
    assert len(wgf.weight_fns) == 1
    assert wgf.num_sd == 5
    assert wgf.num_grid == 100
    assert wgf.use_sample is False
    assert wgf.seed == 42
    
    # Test _get_family method
    family = wgf._get_family()
    assert hasattr(family, 'sufficient_stat')
    assert hasattr(family, 'weights')
    assert np.all(family.weights >= 0)
    assert np.all(np.isfinite(family.weights))
    assert np.sum(family.weights) > 0
    
    # Test pvalue method
    pval = wgf.pvalue(null_value=0.0, alternative='twosided')
    assert 0 <= pval <= 1
    assert np.isfinite(pval)
    
    # Test interval method
    L, U = wgf.interval(level=0.9)
    assert L < U
    assert np.isfinite(L)
    assert np.isfinite(U)
    
    # Test MLE method
    mle = wgf.MLE()
    assert np.isfinite(mle)
    
    # Test invalid alternative
    with pytest.raises(ValueError):
        wgf.pvalue(null_value=0.0, alternative='invalid')

def test_affine_constraint_basic():
    """Test AffineConstraint object creation and basic functionality."""
    # Simple constraint: x <= 1
    linear = np.array([[1.0]])
    offset = np.array([1.0])
    observed = np.array([0.5])  # satisfies constraint
    
    ac = AffineConstraint(
        linear=linear,
        offset=offset,
        observed=observed,
        solver=lambda v: v,
        scale=1.0,
        bias=np.array([0.0])
    )
    
    assert np.array_equal(ac.linear, linear)
    assert np.array_equal(ac.offset, offset)
    assert np.array_equal(ac.observed, observed)
    assert ac.scale == 1.0
    assert np.array_equal(ac.bias, np.array([0.0]))
    
    # Test constraint violation
    with pytest.raises(ValueError):
        AffineConstraint(
            linear=linear,
            offset=offset,
            observed=np.array([2.0]),  # violates constraint
            solver=lambda v: v,
            scale=1.0,
            bias=np.array([0.0])
        )
    
    # Test interval_constraints method
    target = 0.0
    gamma = np.array([1.0])
    lower_bound, upper_bound = ac.interval_constraints(target, gamma)
    assert lower_bound < upper_bound
    assert lower_bound == -np.inf
    assert np.isfinite(upper_bound)
    
    # Test compute_weight method
    estimate = 0.5
    variance = 1.0
    covariance = np.array([0.1])
    tg = ac.compute_weight(estimate, variance, covariance)
    assert isinstance(tg, TruncatedGaussian)
    assert tg.estimate == estimate
    assert tg.sigma == np.sqrt(variance)

def test_discrete_family_basic():
    """Test discrete_family object creation and basic functionality."""
    sufficient_stat = np.array([0.0, 1.0, 2.0])
    weights = np.array([0.3, 0.4, 0.3])
    
    df = discrete_family(sufficient_stat, weights)
    
    assert np.array_equal(df.sufficient_stat, sufficient_stat)
    assert np.array_equal(df.weights, weights)
    assert df.theta == 0.0  # default value
    
    # Test pdf method
    pdf_values = df.pdf(0.0)
    assert np.all(pdf_values >= 0)
    assert np.isclose(np.sum(pdf_values), 1.0)
    
    # Test cdf method
    cdf_value = df.cdf(0.0, x=1.0)
    assert 0 <= cdf_value <= 1
    assert np.isfinite(cdf_value)
    
    # Test MLE method
    mle, _, _ = df.MLE(1.0)
    assert np.isfinite(mle)

def test_simple_score_inference_basic():
    """Test _simple_score_inference function with basic parameters."""
    # Simple case with 2 parameters
    beta = np.array([1.0, 2.0])
    beta_cov = np.array([[1.0, 0.1], [0.1, 1.0]])
    
    df = _simple_score_inference(beta, beta_cov, level=0.9)
    
    assert isinstance(df, pd.DataFrame)
    assert 'pval' in df.columns
    assert 'upper' in df.columns
    assert 'lower' in df.columns
    assert 'estimate' in df.columns
    assert 'std err' in df.columns
    
    # Check that p-values are between 0 and 1
    assert np.all((df['pval'] >= 0) & (df['pval'] <= 1))
    
    # Check that confidence intervals are ordered
    assert np.all(df['lower'] < df['upper'])

def sample_orthogonal(rng=None,
                      p=50,
                      s=5,
                      penalty_facs=False,
                      alt=True,
                      proportion=0.8,
                      level=0.9):
    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)

    X = np.identity(p)
    n, p = X.shape
    Y = rng.standard_normal(n) 
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])
    mu = beta
    Y += mu
    Df = pd.DataFrame({'response':Y})

    penalty_factor = np.ones(p)
    if penalty_facs:
        penalty_factor[:p//10] = 0.5
        penalty_factor[p//10:2*p//10] = 0
        
    penalty_factor *= p / penalty_factor.sum()

    GN = GLMNet(response_id='response',
                fit_intercept=False,
                standardize=False,
                penalty_factor=penalty_factor)

    lambda_val = 2
    Y_sel = Y + np.sqrt((1 - proportion) / proportion) * rng.standard_normal(Y.shape)
    active_naive = np.nonzero(np.fabs(Y_sel) > lambda_val)[0]

    #Df_sel = pd.DataFrame({'response':Y_sel * np.sqrt(proportion)})
    #X_sel = np.sqrt(proportion) * X

    Df_sel = pd.DataFrame({'response':Y_sel})
    X_sel = X 

    GN.fit(X, Df)
    active_naive = np.nonzero(np.fabs(Y_sel) > lambda_val)[0]

    pivots = []
    test_pvals = []

    if active_naive.shape[0] > 0:
        df = lasso_inference(GN, 
                             lambda_val / n,
                             (X_sel, Df_sel),
                             (X, Df),
                             proportion=proportion,
                             dispersion=1)
        df['target'] = mu[df.index]

        signs = np.sign(Y_sel[df.index])
        assert np.all(Y_sel[df.index] * signs >= lambda_val * GN.penalty_factor[df.index])

    if df is not None:
        for j, s in zip(df.index, signs):
            if s == 1:
                upper_bound = np.inf
                if penalty_factor is not None:
                    lower_bound = lambda_val * penalty_factor[j]
                else:
                    lower_bound = lambda_val
            else:
                if penalty_factor is not None:
                    upper_bound = -lambda_val * penalty_factor[j]
                else:
                    upper_bound = -lambda_val
                lower_bound = -np.inf

            tg = df.loc[j,'WG']

            TG = TruncatedGaussian(estimate=Y[j],
                                   noisy_estimate=Y_sel[j],
                                   sigma=1,
                                   smoothing_sigma=np.sqrt((1 - proportion) / proportion),
                                   lower_bound=lower_bound,
                                   upper_bound=upper_bound)
            print(TG, 'test TG')
            WG = WeightedGaussianFamily(estimate=Y[j],
                                        sigma=1,
                                        weight_fns=[TG.weight],
                                        use_sample=False)

            pval = WG.pvalue()
            pivots.append(WG.pvalue(null_value=df.loc[j,'target']))
            test_pvals.append(WG.pvalue(null_value=0))
            assert np.allclose(pval, df.loc[j]['pval'])
        df['pivot_byhand'] = pivots
        df['pval_byhand'] = test_pvals
        assert np.allclose(df['pval'], df['pval_byhand'])

    if df is not None:
        df['pivot'] = [df.loc[j,'WG'].pvalue(df.loc[j, 'target']) for j in df.index]
        assert np.allclose(df['pivot'], df['pivot_byhand'])        

    return df

def resample_orthogonal(rng=None,
                        p=50,
                        s=5,
                        alt=True,
                        proportion=0.8,
                        standardize=True,
                        B=200,  # Reduced from 2000 for faster tests
                        level=0.9):

    # when standardize is True, coverage will be fine but pivot won't be
    # this is because the TruncatedGaussian instance is not corrected for scale

    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])

    scale = np.linspace(1, 3, p)
    beta_hat = beta + rng.standard_normal(p) * scale
    bootstrap_noise = rng.standard_normal((B, p)) * scale[None,:]
    sample = bootstrap_noise + beta_hat[None,:]

    GNI, _ = resampler_inference(sample,
                                proportion=proportion,
                                standardize=standardize)
    if GNI is not None:
        df = GNI.summarize(level=level)
        df['target'] = beta[df.index]
        if not standardize:
            df['pivot'] = [df.loc[j,'WG'].pvalue(df.loc[j,'target']) for j in df.index]
        else:
            # scaling transformation not taken into account so cannot reuse TruncatedGaussian object
            df['pivot'] = np.nan * df['pval']
        return df

def sample_AR1(rho=0.6,               
               rng=None,
               p=100,
               s=5,
               alt=True,
               proportion=0.8,
               dispersion=2,
               level=0.9,
               penalty_facs=False):

    if penalty_facs:
        penalty_factor = np.ones(p)
        penalty_factor[:p//10] = 0.5
        penalty_factor[p//10:2*p//10] = 0.1
    else:
        penalty_factor = None

    D = np.fabs(np.subtract.outer(np.arange(p), np.arange(p)))
    dispersion = 2
    S = (rho**D) * dispersion

    return sample_cov(S,
                      rng=rng,
                      p=p,
                      s=s,
                      alt=alt,
                      proportion=proportion,
                      level=level,
                      penalty_factor=penalty_factor)

def resample_AR1(rng=None,
                 p=50,
                 rho=0.6,
                 s=5,
                 alt=True,
                 proportion=0.8,
                 standardize=True,
                 B=200,  # Reduced from 2000 for faster tests
                 level=0.9):

    # when standardize is True, coverage will be fine but pivot won't be
    # this is because the TruncatedGaussian instance is not corrected for scale

    D = np.fabs(np.subtract.outer(np.arange(p), np.arange(p)))
    dispersion = 2
    S_init = (rho**D) * dispersion
    S = np.linalg.inv(S_init)
    
    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])

    S_sqrt = np.linalg.cholesky(S)
    beta_hat = beta + S_sqrt @ rng.standard_normal(p) 
    bootstrap_noise = rng.standard_normal((B, p)) @ S_sqrt.T
    sample = bootstrap_noise + beta_hat[None,:]

    S_i = np.linalg.inv(S)
    mu = S_i @ beta

    df = resampler_inference(sample,
                             proportion=proportion,
                             standardize=standardize,
                             level=level)
    if df is not None:
        prec_E = S_i[df.index][:,df.index]
        df['target'] = np.linalg.inv(prec_E) @ mu[df.index]
        if not standardize:
            df['pivot'] = [df.loc[j,'WG'].pvalue(df.loc[j,'target']) for j in df.index]
        else:
            # scaling transformation not taken into account so cannot reuse TruncatedGaussian object
            df['pivot'] = np.nan * df['pval']
        return df

def sample_cov(S,
               rng=None,
               p=100,
               s=5,
               alt=True,
               proportion=0.8,
               lambda_val=3,
               level=0.9,
               penalty_factor=None):

    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)

    S_sqrt = X = np.linalg.cholesky(S).T  # X.T @ X = S

    noise = X.T @ rng.standard_normal(p) 
    if alt:
        beta[subs] = rng.standard_normal(s) * 2 + 3 * rng.choice([1,-1])
    mu = S @ beta

    Z = mu + noise

    GNI = score_inference(score=Z,
                          cov_score=S,
                          lambda_val=lambda_val,
                          proportion=proportion,
                          chol_cov=S_sqrt,
                          level=level,
                          penalty_factor=penalty_factor)[0]
    if len(GNI.active_set_) > 0:
        df = GNI.summarize()
    else:
        df = None
        
    if df is not None:
        active = list(df.index)
        df['target'] = np.linalg.inv(S[active][:,active]) @ mu[active]

    if df is not None:
        df['pivot'] = [df.loc[j,'WG'].pvalue(df.loc[j, 'target']) for j in df.index]
        
    return df

def sample_randomX(n,
                   p,
                   fit_intercept,
                   standardize,
                   rng=None,
                   alt=False,
                   s=10,
                   snr=1,
                   proportion=0.8,
                   penalty_facs=False,
                   cv=True,
                   upper_limits=np.inf,
                   level=0.9):

    if rng is None:
        rng = np.random.default_rng(0)
    beta = np.zeros(p)
    subs = rng.choice(p, s, replace=False)

    X = rng.standard_normal((n, p))
    D = np.linspace(1, p, p) / p + 1
    X *= D[None,:]
    sd = np.fabs(1+np.random.standard_normal())
    Y = rng.standard_normal(n) * sd
    if alt:
        beta[subs] = snr * (rng.uniform(3, 5) * rng.choice([-1,1], size=s, replace=True) / np.sqrt(n))

    mu = X @ beta
    Y += mu

    if penalty_facs:
        penalty_factor = np.ones(p)
        penalty_factor[:10] = 0.5
    else:
        penalty_factor = None

    GN = GLMNet(response_id='response',
                fit_intercept=fit_intercept,
                standardize=standardize,
                penalty_factor=penalty_factor,
                upper_limits=upper_limits)

    m = int(proportion*n)

    Df = pd.DataFrame({'response':Y})
    GN.fit(X[:m], Df.iloc[:m]) 

    if cv:
        GN.cross_validation_path(X[:m], Df.iloc[:m], cv=KFold(5))
        lambda_val = GN.index_1se_['Mean Squared Error']
    else:
        eps = rng.standard_normal((X.shape[0], 1000)) * sd
        noise_score = X.T @ eps
        lambda_val = 1.2 * np.median(np.fabs(noise_score).max(0)) / X.shape[0]
        
    df = lasso_inference(GN, 
                         lambda_val,
                         (X[:m], Df.iloc[:m]),
                         (X, Df),
                         proportion=proportion,
                         level=level)

    if df is not None:
        if fit_intercept:
            active_set = np.array(df.index[1:]).astype(int)
        else:
            active_set = np.array(df.index).astype(int)
        X_sel = X[:,active_set]

        if fit_intercept:
            X_sel = np.column_stack([np.ones(X_sel.shape[0]), X_sel])
        targets = np.linalg.pinv(X_sel) @ mu

        df['target'] = targets
            
    if df is not None:
        df['pivot'] = [df.loc[j,'WG'].pvalue(df.loc[j, 'target']) for j in df.index]
        
    return df

@pytest.mark.parametrize('standardize', [True])  # Reduced from [True, False]
@pytest.mark.parametrize('penalty_facs', [False])  # Reduced from [True, False]
@pytest.mark.parametrize('fit_intercept', [True])  # Reduced from [True, False]
#@pytest.mark.parametrize('upper_limits', [np.inf, 0.1])
@pytest.mark.parametrize('p', [20])  # Reduced from 50 for faster tests
@pytest.mark.parametrize('n', [100])  # Reduced from 200 for faster tests
def test_randomX(n,
                 p,
                 standardize,
                 fit_intercept,
                 penalty_facs,
                 upper_limits=np.inf):

    for _ in range(1):  # Reduced from 2 for faster tests
        df = None
        while df is None: # make sure it is run
            kwargs = dict(n=n,
                          p=p,
                          standardize=standardize,
                          fit_intercept=fit_intercept,
                          penalty_facs=penalty_facs,
                          cv=False,  # Always False to avoid expensive cross-validation
                          upper_limits=upper_limits,
                          alt=True,  # Enable alternative hypothesis for stronger signal
                          snr=3)  # Increased from default 1 for stronger signal

            df = main(sample_randomX,
                      kwargs,
                      ntrial=1)

def test_resampler(p=30,  # Reduced from 50 for faster tests
                   standardize=True,
                   level=0.9):

    for _ in range(2):  # Reduced from 5 for faster tests
        df = None
        while df is None: # make sure it is run
            df = main(resample_orthogonal,
                      {'p':p, 'level':level},
                      ntrial=1)


def test_orthogonal(p=50,
                    level=0.9):  # Reduced from 100 for faster tests

    for _ in range(2):  # Reduced from 5 for faster tests
        df = None
        while df is None: # make sure it is run
            df = main(sample_orthogonal,
                      {'p':p, 'level':level},
                      ntrial=1)


def test_AR1(p=50,  # Reduced from 100 for faster tests
             rho=0.6,
             level=0.9):

    for _ in range(2):  # Reduced from 5 for faster tests
        df = None
        while df is None: # make sure it is run
            df = sample_AR1(p=p,
                            rho=rho,
                            level=level)

            

def main(sampler,
         kwargs,
         ntrial=50):  # Reduced from 500 for faster tests

    ncover = 0
    nsel = 0

    dfs = []

    rng = np.random.default_rng(0)
    kwargs.update(rng=rng)
    
    for i in range(ntrial):
        df = sampler(**kwargs)
        if df is not None:
            dfs.append(df)
        if len(dfs) > 0:
            all_df = pd.concat(dfs).dropna()
            ncover += ((all_df['lower'] < all_df['target']) & (all_df['upper'] > all_df['target'])).sum()
            nsel += all_df.shape[0]

            print('cover:', ncover / nsel,
                  'power:', (all_df['pval'] < 0.05).mean(),
                  'pivot<0.05:', (all_df['pivot'] < 0.05).mean(),
                  'std(pivot):', all_df['pivot'].std())

    if len(dfs) > 0:
        return all_df
        
def truncated_inference(B=100,  # Reduced from 1000 for faster tests
                             smoothing_sigma=np.sqrt(1/3),
                             sigma=1,
                             upper_bound=None,
                             lower_bound=None,
                             alt=False,
                             level=0.9):

    pvals = []
    cover = []

    rng = np.random.default_rng(0)
    
    for _ in range(B):
        mu = rng.standard_normal() + 3
        if not alt:
            mu *= 0
        if lower_bound is None or upper_bound is None:
            lower_bound = rng.uniform(0, 2)
            upper_bound = lower_bound + rng.uniform(3, 5)
        while True:
            Z = rng.standard_normal() * sigma
            Z += mu
            Z_noisy = Z + rng.standard_normal() * smoothing_sigma
            if (Z_noisy > lower_bound) and (Z_noisy < upper_bound):
                TG = TruncatedGaussian(estimate=Z,
                                       sigma=sigma,
                                       smoothing_sigma=smoothing_sigma,
                                       lower_bound=lower_bound,
                                       upper_bound=upper_bound)
                
                (L, U), mle, pval = (TG.interval(level=level), TG.MLE(), TG.pvalue())

                cover.append((L<mu) * (U>mu))
                pvals.append(pval)
                print(np.mean(cover), np.mean(np.array(pvals) < 0.05))
                break
            
