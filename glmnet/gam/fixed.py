"""
Fixed effects of a GAM from modelmatrix: the items of `terms` that are not
smoothers, e.g. `GAM([s('x'), 'z', contrast('c', 'sum'), poly('w', 2),
('z', 'c')])`, mgcv's `y ~ s(x) + z + c + poly(w, 2) + z:c`.

The items that are not smoothers or ModelSpecs (a column, an interaction,
a modelmatrix `Feature` such as `poly()` or `contrast()`, a `wildcard`)
make one `ModelSpec`, in their order, so that a wildcard leaves out the
terms already given; a `ModelSpec` item is its own. Each term of a fitted
ModelSpec is a `ColumnsSmoother` of its columns, at the place of its item
in `terms`, named by the term (`z`, `poly(w, degree=2)`, `z:c`), its
coefficients by the ModelSpec's names of its columns (`c[b]`, ...). The
columns enter the GAM's data under names of their own, which no column of
the data has.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from modelmatrix import ModelSpec, Feature, Column, Wildcard

from .terms import (Smoother, ColumnsSmoother, LinearSmoother, SplineSmoother,
                    LoessSmoother, CategoricalSmoother, RandomEffectSmoother,
                    CorrelatedRandomEffectSmoother, _column_label)
from .tensor import TensorSplineSmoother
from .varying import VaryingSmoother, CubicRegressionSplineSmoother

# the names of the fixed effects' columns in the GAM's data
_PREFIX = '\x00fixed'

# smoothers whose fit spans a linear function of their columns, unpenalized:
# with such a column as a fixed effect too, the model is not identified
_LINEAR_SPAN = (SplineSmoother, CubicRegressionSplineSmoother, LoessSmoother,
                TensorSplineSmoother, LinearSmoother)


def expand_terms(items, X):
    """
    The terms of a GAM, its fixed effects expanded, the fitted ModelSpecs
    of these and the data with their columns.

    Returns
    -------
    terms : list of Smoother
    specs : list of fitted ModelSpec
    X : the data, with the columns of the specs (`X` if there are none)
    """
    items = list(items)
    plain = [item for item in items if not isinstance(item, (Smoother, ModelSpec))]
    specs, places = [], []   # places[k]: the positions (in items) of spec k's items
    if plain:
        specs.append(ModelSpec(plain))   # (its terms are the items: see _place)
        places.append([j for j, item in enumerate(items)
                       if not isinstance(item, (Smoother, ModelSpec))])
    for j, item in enumerate(items):
        if isinstance(item, ModelSpec):
            if item.intercept:
                raise ValueError('a ModelSpec of a GAM has no intercept (the GAM fits it): '
                                 'ModelSpec(..., intercept=False)')
            specs.append(clone(item))
            places.append([j])
    features = _fit_features([item for item in items if isinstance(item, Smoother)], X)
    if not specs and not features:
        return items, [], X
    data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    for spec in specs:
        spec.fit(data)
    specs = specs + list(features.values())

    # the terms of each item, at its place
    expanded = {j: [item] for j, item in enumerate(items) if isinstance(item, Smoother)}
    for k, (spec, place) in enumerate(zip(specs, places)):
        for j, spec_terms in _place(spec, [items[i] for i in place], place).items():
            expanded[j] = [_columns_smoother(spec, k, t) for t in spec_terms]
    terms = [term for j in sorted(expanded) for term in expanded[j]]
    _check_identified(terms, specs[:len(specs) - len(features)])
    return terms, specs, augment(data, specs)


def augment(X, specs):
    """
    The data `X` with the columns of the fitted ModelSpecs `specs`: those of
    the fixed effects, and the column of each feature a smoother uses.
    """
    if not specs:
        return X
    data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    blocks = [data]
    for k, spec in enumerate(specs):
        D = spec.transform(data)
        feature = getattr(spec, '_smoother_feature', None)
        names = ([feature_column(feature)] if feature is not None else
                 [_private(k, c) for c in spec.get_feature_names_out()])
        blocks.append(pd.DataFrame(np.asarray(D, float), index=data.index, columns=names))
    return pd.concat(blocks, axis=1)


def feature_column(feature):
    """The name, in the GAM's data, of the column of a feature a smoother uses."""
    return f'{_PREFIX}feature:{feature.name}'


def _smoother_columns(term):
    """
    The columns of a smoother, as a list, and those of them that are
    groupings (whose levels the term has), which cannot be features.
    """
    columns = term.columns
    columns = (list(columns) if isinstance(columns, (list, tuple))
               and not isinstance(columns, Feature) else [columns])
    if isinstance(term, CategoricalSmoother):
        groups = [term.column]
    elif isinstance(term, (RandomEffectSmoother, CorrelatedRandomEffectSmoother)):
        column = term.column
        groups = (list(column) if isinstance(column, (list, tuple))
                  and not isinstance(column, Feature) else [column])
    elif isinstance(term, VaryingSmoother) and term.as_categorical:
        groups = [term.by]
    else:
        groups = []
    return columns, groups


def _fit_features(smoothers, X):
    """
    A fitted ModelSpec of each feature (a modelmatrix Feature, e.g. a log)
    a smoother uses as a column, by the feature's name: a single column.
    """
    features = {}
    data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    for term in smoothers:
        columns, groups = _smoother_columns(term)
        for column in columns:
            if not isinstance(column, Feature):
                continue
            if any(column is g or column == g for g in groups):
                raise ValueError(f'{term.name}: the feature {column.name!r} is a grouping '
                                 '(levels); a feature can only be a numeric column of a '
                                 'smoother')
            if column.name in features:
                if features[column.name]._smoother_feature != column:
                    raise ValueError(f'two different features have the name {column.name!r}: '
                                     'give them names of their own (name=...)')
                continue
            spec = ModelSpec([column]).fit(data)
            if len(spec.get_feature_names_out()) != 1:
                raise ValueError(f'{term.name}: the feature {column.name!r} has '
                                 f'{len(spec.get_feature_names_out())} columns; a smoother\'s '
                                 'feature should have one')
            spec._smoother_feature = column
            features[column.name] = spec
    return features


def _private(k, name):
    return f'{_PREFIX}{k}:{name}'


def _columns_smoother(spec, k, term):
    i = _index(spec.expanded_terms_, term)
    names = list(spec.column_names_[spec.expanded_terms_[i]])
    smoother = ColumnsSmoother([_private(k, c) for c in names], label=spec.terms_[i].name)
    smoother.coef_names = names
    return smoother


def _index(terms, term):
    # the position of a term (itself, else an equal one)
    for i, t in enumerate(terms):
        if t is term:
            return i
    return terms.index(term)


def _place(spec, items, place):
    """
    The expanded terms of a fitted spec by the position (in `terms`) of
    their item: a ModelSpec item has all its terms; in the spec of the
    other items, an item is its term, in their order, and a wildcard's
    terms are those up to the next item.
    """
    expanded = list(spec.expanded_terms_)
    if isinstance(items[0], ModelSpec):
        return {place[0]: expanded}
    out, p = {j: [] for j in place}, 0
    for n, (item, j) in enumerate(zip(items, place)):
        later = [i for i in items[n + 1:] if not isinstance(i, Wildcard)]
        if isinstance(item, Wildcard):
            while p < len(expanded) and not any(expanded[p] is i for i in later):
                out[j].append(expanded[p])
                p += 1
        elif p < len(expanded) and expanded[p] is item:
            out[j].append(expanded[p])
            p += 1
    if p != len(expanded):
        raise RuntimeError('the terms of the fixed effects could not be placed: '
                           f'{expanded[p:]!r}')
    return out


def _variables(term):
    """The columns of the data a main-effect term uses (none for an interaction)."""
    if isinstance(term, Feature):
        cols = set()
        for v in term.variables:
            cols |= _variables(v) if isinstance(v, (Feature, Column)) else {v}
        return cols
    if isinstance(term, Column):
        return {term.idx}
    if isinstance(term, tuple):   # an interaction
        return set()
    return {term}


def _smoothed_columns(term):
    """The columns whose linear function a smoother's fit spans (unpenalized)."""
    if isinstance(term, VaryingSmoother):
        if not term.as_categorical:
            return []
        term = term.inner
    if isinstance(term, _LINEAR_SPAN):
        column = term.column
        return (list(column) if isinstance(column, (list, tuple))
                and not isinstance(column, Feature) else [column])
    return []


def _key(column):
    # a column, or a feature by its name
    return ('feature', column.name) if isinstance(column, Feature) else column


def _check_identified(terms, specs):
    fixed = {}
    for spec in specs:
        for term, feature in zip(spec.expanded_terms_, spec.terms_):
            for column in _variables(term):
                fixed.setdefault(column, feature.name)
            if isinstance(term, Feature):
                fixed.setdefault(_key(term), feature.name)
    for term in terms:
        both = sorted((_column_label(c) for c in _smoothed_columns(term) if _key(c) in fixed),
                      key=str)
        if both:
            c = both[0]
            name = fixed.get(c, fixed.get(('feature', c)))
            raise ValueError(f'{term.name} and the fixed effect {name!r} both fit '
                             f'a linear function of {c!r}: the model is not identified. '
                             f'Leave {c!r} out of the fixed effects (e.g. '
                             f'wildcard(exclude=[..., {c!r}]))')
