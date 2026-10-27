"""
The model a GAM specifies, in LaTeX: `model_latex(gam)`, shown by Jupyter
(`GAM._repr_mimebundle_`) so that a reader sees the model being fitted.
The response's distribution and link, the linear predictor as the sum of
the terms' contributions, and a line per term saying what it is (a smooth
and its penalty, a random effect and its distribution) and how its
smoothing parameter or variance is chosen. It needs only the
specification, not a fit.
"""
import numpy as np

from modelmatrix import ModelSpec, Feature, Column, Wildcard

from glmnet.cox import CoxFamily
from .terms import Smoother


def _tex(name):
    """A column's name as LaTeX text."""
    s = str(name)
    for a, b in [('\\', r'\textbackslash '), ('_', r'\_'), ('&', r'\&'), ('%', r'\%'),
                 ('$', r'\$'), ('#', r'\#'), ('{', r'\{'), ('}', r'\}')]:
        s = s.replace(a, b)
    return r'\text{' + s + '}'


def _offset_weight(gam):
    """
    The offset and the weights of a GAM in LaTeX (None if none): its
    `offset_id` and `weight_id`, or those of its outcome (a feature by its
    name), a binomial's trials (times any weights) its weights.
    """
    outcome = getattr(gam, 'outcome', None)
    if outcome is None:
        offset, weight = getattr(gam, 'offset_id', None), getattr(gam, 'weight_id', None)
        trials = None
    else:
        names = {key: getattr(value, 'name', value) for key, value in outcome.extra.items()}
        offset, weight = names.get('offset'), names.get('weight')
        trials = getattr(outcome.response, 'trials', None)
    tex = lambda name: None if name is None else rf'{_tex(name)}_i'
    weights = [w for w in [tex(trials), tex(weight)] if w is not None]
    return tex(offset), r' \cdot '.join(weights) if weights else None


def _columns(term):
    column = term.column
    return list(column) if isinstance(column, (list, tuple)) else [column]


def _group(term):
    """A random effect's grouping: its column, or the combination of its columns."""
    cols = _columns(term)
    return _tex(':'.join(str(c) for c in cols))


def _chosen(term, method, attr='lam'):
    """How a term's smoothing parameter is chosen: given, by df, or by the method."""
    df = getattr(term, 'df', None)
    lam = getattr(term, attr, None)
    if df is not None:
        return rf'\text{{df}} = {df:g}'
    if lam is not None:
        values = np.atleast_1d(lam)
        return r'\text{given}' if len(values) > 1 else rf'\lambda = {float(values[0]):.3g}'
    return rf'\text{{by {method}}}'


def _varying(term, j, method):
    """A varying-coefficient term (`varying(inner, by=...)`): the inner term's line, varied."""
    inner = term.inner
    contribution, description = _term(inner, j, method)
    by = _tex(term.by)
    f = rf'f_{{{j}}}'
    # the inner term's choice of smoothing replaced by the varying term's
    chosen_inner, chosen = _chosen(inner, method), _chosen(term, method)
    if description is not None and description.endswith(chosen_inner):
        description = description[:-len(chosen_inner)] + chosen
    if term.as_categorical:
        level = rf'{j},{by}[i]'
        contribution = contribution.replace(f + '(', rf'f_{{{level}}}(', 1)
        sharing = ('' if not hasattr(inner, 'lam') else
                   r'\text{, one smoothing parameter for all the levels}' if term.pooled_lam
                   else r'\text{, smoothing parameters per level}')
        head = rf'f_{{{j},l}}\ \text{{for each level }} l \text{{ of }} {by}\text{{, each summing to 0 over the data}}'
        if description is not None:
            description = description.replace(f + ':', head + r'\text{:}\ ', 1) + sharing
        return contribution, description
    contribution = rf'{by}_i\, {contribution}'
    if description is not None:
        description = description.replace(f + ':', rf'{f}\ \text{{(the coefficient of }} {by}\text{{)}}:', 1)
    return contribution, description


def _fixed_name(item):
    """The name of a fixed effect of `terms` (see `glmnet.gam.fixed`)."""
    if isinstance(item, ModelSpec):
        return ', '.join(_fixed_name(t) for t in item.terms)
    if isinstance(item, Wildcard):
        return 'the other columns'
    if isinstance(item, (Feature, Column)):
        return item.name
    if isinstance(item, tuple):   # an interaction
        return ':'.join(_fixed_name(t) for t in item)
    return str(item)


def _term(term, j, method):
    """The term's contribution to the linear predictor, and its description."""
    if not isinstance(term, Smoother):   # a fixed effect: columns of a ModelSpec
        return rf'z^{{({j})T}}_i \gamma_{{{j}}}', \
            rf'z^{{({j})}}: ' + _tex(_fixed_name(term))
    if hasattr(term, 'inner') and term.inner is not None:
        return _varying(term, j, method)
    kind = type(term).__name__
    cols = _columns(term)
    x = lambda c: _tex(c) + '_i'
    f = rf'f_{{{j}}}'
    if kind == 'LinearSmoother':
        return rf'\beta_{{{j}}}\, {x(cols[0])}', None
    if kind in ('CategoricalSmoother', '_FactorTerm'):
        return rf'\alpha^{{({j})}}_{{{_tex(cols[0])}[i]}}', \
            rf'\alpha^{{({j})}}: \text{{an effect per level of }} {_tex(cols[0])}'
    if kind == 'ColumnsSmoother':
        cols = getattr(term, 'coef_names', None) or cols
        return rf'z^{{({j})T}}_i \gamma_{{{j}}}', \
            rf'z^{{({j})}}: \text{{the columns }} ' + ', '.join(_tex(c) for c in cols)
    if kind == 'SplineSmoother':
        return rf'{f}({x(cols[0])})', \
            (rf'{f}: \text{{cubic smoothing spline, penalty }} \lambda_{{{j}}}\textstyle\int {f}'
             rf"''(x)^2\,dx,\ " + _chosen(term, method))
    if kind == 'CubicRegressionSplineSmoother':
        return rf'{f}({x(cols[0])})', \
            (rf'{f}: \text{{cubic regression spline, {term.k} knots, penalty }} \lambda_{{{j}}}'
             rf"\textstyle\int {f}''(x)^2\,dx,\ " + _chosen(term, method))
    if kind == 'LoessSmoother':
        return rf'{f}({x(cols[0])})', \
            rf'{f}: \text{{loess, span {term.span:g}, degree {term.degree}}}'
    if kind == 'CyclicSplineSmoother':
        period = getattr(term, 'period', None)
        cycle = (rf'[{period[0]:g}, {period[1]:g})' if period is not None
                 else r"\text{the data's range}")
        return rf'{f}({x(cols[0])})', \
            (rf'{f}: \text{{cyclic cubic spline on }} {cycle}\text{{, penalty }} '
             rf"\lambda_{{{j}}}\textstyle\int {f}''(x)^2\,dx,\ " + _chosen(term, method))
    if kind == 'TensorSplineSmoother':
        args = ', '.join(x(c) for c in cols)
        return rf'{f}({args})', \
            (rf'{f}: \text{{tensor product of cubic regression splines, a penalty per '
             rf'column}},\ ' + _chosen(term, method))
    if kind == 'GaussianProcessSmoother':
        args = ', '.join(x(c) for c in cols)
        kernel = getattr(term, 'kernel_name', 'Gaussian process')
        return rf'{f}({args})', \
            (rf'{f}: \text{{Gaussian process, {kernel} correlation}},\ ' + _chosen(term, method))
    if kind == 'PenalizedColumnsSmoother':
        m = len(term.penalties)
        penalty = (rf'\lambda_{{{j}}}\, \gamma_{{{j}}}^T S \gamma_{{{j}}}' if m == 1 else
                   rf'\textstyle\sum_{{m=1}}^{{{m}}} \lambda_{{{j}m}}\, \gamma_{{{j}}}^T S_m \gamma_{{{j}}}')
        return rf'z^{{({j})T}}_i \gamma_{{{j}}}', \
            rf'\gamma_{{{j}}}: \text{{penalty }} {penalty},\ ' + _chosen(term, method)
    if kind == 'RandomEffectSmoother':
        g = _group(term)
        b = rf'b^{{({j})}}_{{{g}[i]}}'
        by = getattr(term, 'by', None)
        contribution = b if by is None else rf'{b}\, {x(by)}'
        return contribution, \
            (rf'b^{{({j})}}_g \sim N(0, \sigma_{{{j}}}^2) \text{{ independently over the groups of }} '
             rf'{g},\ ' + _chosen(term, method).replace(r'\lambda', r'\phi/\sigma^2'))
    if kind == 'CorrelatedRandomEffectSmoother':
        g = _group(term)
        effects = (['1'] if term.intercept else []) + list(term.slopes)
        parts = []
        for m, e in enumerate(effects):
            b = rf'b^{{({j})}}_{{{m},{g}[i]}}'
            parts.append(b if e == '1' else rf'{b}\, {x(e)}')
        structure = {'unstructured': 'unstructured', 'compound_symmetry': 'compound symmetric',
                     'diagonal': 'diagonal'}[getattr(term, 'covariance', 'unstructured')]
        given = getattr(term, 'precision', None) is not None
        q = len(effects)
        vector = (', '.join(rf'b^{{({j})}}_{{{m},g}}' for m in range(q)) if q <= 3 else
                  rf'b^{{({j})}}_{{0,g}}, \ldots, b^{{({j})}}_{{{q - 1},g}}')
        return '(' + ' + '.join(parts) + ')', \
            (rf'({vector}) \sim '
             rf'N(0, \Sigma_{{{j}}}),\ \Sigma_{{{j}}} \text{{ {structure}, }} ' +
             (r'\text{given}' if given else rf'\text{{by {method}}}'))
    args = ', '.join(x(c) for c in cols)
    return rf'{f}({args})', None


_FAMILIES = {'Gaussian': r'N(\mu_i, \phi / w_i)', 'Binomial': r'\text{Binomial}(\mu_i)',
             'Poisson': r'\text{Poisson}(\mu_i)', 'Gamma': r'\text{Gamma}(\mu_i, \phi)',
             'InverseGaussian': r'\text{IG}(\mu_i, \phi)', 'NegativeBinomial':
             r'\text{NegBin}(\mu_i)', 'Tweedie': r'\text{Tweedie}(\mu_i, \phi)'}
_LINKS = {'Identity': r'\mu_i', 'Log': r'\log \mu_i', 'Logit': r'\operatorname{logit} \mu_i',
          'Probit': r'\Phi^{-1}(\mu_i)', 'CLogLog': r'\log(-\log(1 - \mu_i))',
          'InversePower': r'1/\mu_i', 'InverseSquared': r'1/\mu_i^2', 'Sqrt': r'\sqrt{\mu_i}'}


def model_latex(gam):
    """
    The model `gam` specifies, as a LaTeX `aligned` block (without the
    surrounding `$$`): the response's distribution and link, the linear
    predictor `eta_i` and a line per term.
    """
    method = getattr(gam, 'method', None) or 'fixed'
    method = method if isinstance(method, str) else getattr(method, 'name', type(method).__name__)
    if method == 'fixed':
        method = 'fixed smoothing'
    family = gam.family
    cox = isinstance(family, CoxFamily)
    lines = []
    if cox:
        lines.append(r'h_i(t) &= h_0(t)\, \exp(\eta_i)')
    else:
        base = getattr(family, 'base', family)
        fam = type(base).__name__
        link = type(base.link).__name__ if hasattr(base, 'link') else 'Identity'
        lines.append(rf'y_i &\sim {_FAMILIES.get(fam, r"\text{" + fam + r"}(\mu_i)")}, \qquad '
                     rf'{_LINKS.get(link, r"g(\mu_i)")} = \eta_i')
    contributions, descriptions = [] if cox else [r'\beta_0'], []
    for j, term in enumerate(gam.terms, 1):
        contribution, description = _term(term, j, method)
        contributions.append(contribution)
        if description is not None:
            descriptions.append(description)
    offset, weight = _offset_weight(gam)
    if offset is not None:
        contributions.append(offset)
    # (a few terms a line)
    per_line = 3
    rows = [contributions[i:i + per_line] for i in range(0, len(contributions), per_line)]
    lines.append(r'\eta_i &= ' + ' + '.join(rows[0]))
    lines.extend(r'&\quad + ' + ' + '.join(row) for row in rows[1:])
    if weight is not None and not cox:
        lines.append(rf'w_i &= {weight} \text{{ (prior weights)}}')
    body = r' \\ '.join(lines)
    if descriptions:
        body += r' \\[4pt] ' + r' \\ '.join('&' + d for d in descriptions)
    return r'\begin{aligned} ' + body + r' \end{aligned}'
