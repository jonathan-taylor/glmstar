"""
The model a GAM specifies, in LaTeX (`GAM.model_latex()`, Jupyter's
`text/latex`): the family and link, the linear predictor and a line per
term, from the specification alone.
"""
import numpy as np
import pytest
import statsmodels.api as sm

from glmnet.cox import CoxFamily
from glmnet.gam import (GAM, GLMM, s, lo, re, linear, categorical, tensor_spline, gaussian_process,
                        correlated_random_effect, penalized_columns)
from glmnet.gam.terms import ColumnsTerm


def test_terms_and_family():
    G = GLMM(terms=[s('year', df=4), s('age'), categorical('education'), tensor_spline('hour', 'temp'),
                   gaussian_process('lon', 'lat'), re('maritl'), re(['batch', 'cask']),
                   re('subject', by='days'), correlated_random_effect('subject', slopes=['days']),
                   linear('x_1'), lo('z', span=0.3), penalized_columns(['a', 'b'], [np.eye(2)]),
                   ColumnsTerm(['c1', 'c2'])],
            family=sm.families.Poisson(), method='REML', offset_id='log_exposure', weight_id='w')
    tex = G.model_latex()
    assert tex.startswith(r'\begin{aligned}') and tex.endswith(r'\end{aligned}')
    for piece in [r'\text{Poisson}(\mu_i)', r'\log \mu_i = \eta_i', r'\beta_0',
                  r'f_{1}(\text{year}_i)', r'\text{df} = 4', r'\text{by REML}',
                  r'\alpha^{(3)}_{\text{education}[i]}', r'f_{4}(\text{hour}_i, \text{temp}_i)',
                  r'\text{Gaussian process', r'b^{(6)}_{\text{maritl}[i]}',
                  r'b^{(7)}_{\text{batch:cask}[i]}', r'b^{(8)}_{\text{subject}[i]}\, \text{days}_i',
                  r'\Sigma_{9} \text{ unstructured', r'\beta_{10}\, \text{x\_1}_i',
                  r'\text{loess, span 0.3, degree 1}', r'\gamma_{12}^T S \gamma_{12}',
                  r'\text{log\_exposure}_i', r'(prior weights)']:
        assert piece in tex, piece


def test_mimebundle():
    G = GAM(terms=[s('age')])
    bundle = G._repr_mimebundle_()
    assert set(bundle) == {'text/plain', 'text/latex'}
    assert bundle['text/latex'] == '$$' + G.model_latex() + '$$'
    # the default: Gaussian, identity link, fixed smoothing (df=4)
    assert r'N(\mu_i, \phi / w_i)' in bundle['text/latex']


@pytest.mark.parametrize('family, expected', [
    (sm.families.Binomial(), r'\operatorname{logit} \mu_i'),
    (sm.families.Gamma(sm.families.links.Log()), r'\text{Gamma}(\mu_i, \phi)'),
    (CoxFamily(event_id='time', status_id='status'), r'h_i(t) &= h_0(t)\, \exp(\eta_i)')])
def test_families(family, expected):
    tex = GAM(terms=[s('age', lam=0.01)], family=family).model_latex()
    assert expected in tex
    assert r'\lambda = 0.01' in tex
    if isinstance(family, CoxFamily):
        assert r'\beta_0' not in tex
