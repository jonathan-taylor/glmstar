"""
The columns of `y` named by their role.

A data frame `y`, e.g. `Y = Outcome(...).fit_transform(D)` of modelmatrix,
names each column by its role:

- `offset`: the offset; for a multi-response model, `offset[r]` for each
  response `r` (modelmatrix's `offset={...}`), or `offset` for all;
- `weight`: the prior weights (of the observations);
- for a Cox model, `stop` and `status`, `start` for (start, stop] data,
  and `strata`;
- the response: every other column. Of a binomial, two columns are R's
  `cbind(successes, failures)` (modelmatrix's `Binomial(..., output='counts')`).

A series or an array `y` is the response alone. The deprecated ids
(`response_id`, `offset_id`, `weight_id`, and the `CoxFamily`'s
`event_id`, `status_id`, `start_id`, `strata_id`) name the columns of a
`y` of one's own instead: `_role_y` renames them to their roles, with a
FutureWarning.
"""

import sys
import warnings

import numpy as np
import pandas as pd

OFFSET, WEIGHT = 'offset', 'weight'
_RENAMED = '_glmstar_by_role'   # the attribute of a frame renamed from the ids'
START, STOP, STATUS, STRATA = 'start', 'stop', 'status', 'strata'
COX_ROLES = (START, STOP, STATUS, STRATA)


def _is_offset(column):
    return column == OFFSET or (isinstance(column, str) and column.startswith('offset['))


def response_columns(y, cox=False):
    """The columns of the response of a role-named frame `y`."""
    roles = set(COX_ROLES) if cox else set()
    return [c for c in y.columns if not (_is_offset(c) or c == WEIGHT or c in roles)]


def read_y(y):
    """
    The response, offset and weights of `y`: a role-named frame (see the
    module), or a series or array, the response alone.

    Returns
    -------
    response : np.ndarray
        Not squeezed.
    offset : np.ndarray or None
        One column, or one for each response (`offset[r]`, in the order of
        the response's columns).
    weight : np.ndarray
        The weights, or ones.
    """
    if not isinstance(y, pd.DataFrame):
        y = np.asarray(y)
        return y, None, np.ones(y.shape[0])
    if any(isinstance(c, str) and c.startswith('weight[') for c in y.columns):
        raise ValueError('weights of each response are not supported: the weights are of '
                         'the observations, a column weight')
    response = response_columns(y)
    offsets = [c for c in y.columns if _is_offset(c)]
    if OFFSET in offsets and len(offsets) > 1:
        raise ValueError(f'an offset for every response and one for each, {offsets}: '
                         'give one or the other')
    if OFFSET in offsets:
        offset = np.asarray(y[OFFSET])
    elif offsets:
        # in the order of the response's columns; of a multinomial's column of
        # labels, by its classes, in the order of its levels (modelmatrix's)
        names = [f'offset[{r}]' for r in response]
        if sorted(names) != sorted(offsets):
            names = offsets
        offset = np.asarray(y[names])
    else:
        offset = None
    weight = np.asarray(y[WEIGHT]) if WEIGHT in y.columns else np.ones(y.shape[0])
    return np.asarray(y[response]), offset, weight


def cox_ids(y):
    """The columns of the survival data of a role-named frame `y` (CoxFamilySpec's ids)."""
    columns = set(y.columns) if isinstance(y, pd.DataFrame) else set()
    return dict(event_id=STOP,
                status_id=STATUS,
                start_id=START if START in columns else None,
                strata_id=STRATA if STRATA in columns else None)


def legacy(estimator):
    """Does the estimator name the columns of `y` by the deprecated ids?"""
    if any(getattr(estimator, name, None) is not None
           for name in ['response_id', 'offset_id', 'weight_id']):
        return True
    family = getattr(estimator, 'family', None)
    return hasattr(family, 'event_id') and (
        family.event_id, family.status_id, family.start_id, family.strata_id) != (
            'event', 'status', None, None)


def _called_from_outside(depth=3):
    # is the method calling _role_y called from outside glmstar (not by a fit
    # of glmstar's own, e.g. a sub-estimator's, or a fit calling its parent's)?
    caller = sys._getframe(depth).f_globals.get('__name__', '')
    return not (caller == 'glmnet' or caller.startswith('glmnet.'))


def _role_y(estimator, y):
    """
    `y` named by role, for the estimator: with the deprecated ids, the
    columns they name renamed to their roles (a FutureWarning); for a Cox
    model, checked to be its survival data (and offset and weights).
    """
    cox = hasattr(getattr(estimator, 'family', None), 'event_id')   # a CoxFamily
    if not isinstance(y, pd.DataFrame):
        if cox or not (legacy(estimator) and np.ndim(y) == 2):
            return y
        y = pd.DataFrame(np.asarray(y))   # (the ids of an array: its columns' positions)
    if y.attrs.get(_RENAMED):
        return y
    outside = _called_from_outside()
    if cox:
        renamed = _cox_y(estimator, y, outside)
    elif legacy(estimator):
        renamed = _renamed(estimator, y, outside)
    else:
        return y
    if renamed is not y:
        # (so that a fit within the fit, or a fold, does not rename it again)
        renamed.attrs[_RENAMED] = True
    return renamed


def _id_columns(column):
    return list(column) if isinstance(column, (list, tuple)) else [column]


def _renamed(estimator, y, outside):
    """A frame of one's own, its columns named by the ids, named by role."""
    ids = {name: getattr(estimator, name, None) for name in ['response_id', 'offset_id', 'weight_id']}
    given = [name for name, value in ids.items() if value is not None]
    if outside and given:
        warnings.warn(f'{", ".join(given)} {"are" if len(given) > 1 else "is"} deprecated: '
                      'name the columns of y by their role, as modelmatrix\'s Outcome does, '
                      "e.g. Y = Outcome('y', offset='O', weight='w').fit_transform(D)",
                      FutureWarning, stacklevel=4)
    out = {}
    offset_cols = [] if ids['offset_id'] is None else _id_columns(ids['offset_id'])
    weight_cols = [] if ids['weight_id'] is None else _id_columns(ids['weight_id'])
    if ids['response_id'] is not None:
        response = _id_columns(ids['response_id'])
    else:
        response = [c for c in y.columns if c not in offset_cols + weight_cols]
    if (getattr(estimator, '_binomial_two_columns', False) and len(response) == 2):
        # glmstar's former (trials, successes): as R's cbind(successes, failures)
        trials, successes = y[response[0]].to_numpy(), y[response[1]].to_numpy()
        out[response[1]] = successes
        out[f'{response[0]}-{response[1]}'] = trials - successes
    else:
        for c in response:
            out[c] = y[c].to_numpy() if not isinstance(y[c].dtype, pd.CategoricalDtype) else y[c]
    if len(offset_cols) == 1:
        out[OFFSET] = y[offset_cols[0]].to_numpy()
    elif offset_cols:
        # one for each response, or (of a multinomial's labels) each class, in order
        keys = response if len(offset_cols) == len(response) else offset_cols
        for r, c in zip(keys, offset_cols):
            out[f'offset[{r}]'] = y[c].to_numpy()
    if weight_cols:
        out[WEIGHT] = y[weight_cols[0]].to_numpy()
    return pd.DataFrame(out, index=y.index)


def _cox_y(estimator, y, outside):
    """The survival data (and offset and weights) of a Cox model, named by role."""
    family = estimator.family
    offset_id, weight_id = (getattr(estimator, 'offset_id', None),
                            getattr(estimator, 'weight_id', None))
    if not legacy(estimator) and STOP in y.columns and STATUS in y.columns:
        # named by role
        unknown = [c for c in y.columns
                   if not (c in COX_ROLES or c == WEIGHT or _is_offset(c))]
        if unknown:
            raise ValueError(f'the y of a Cox model has the columns {unknown}, not of its '
                             f'roles (stop, status, start, strata, offset, weight): build it '
                             "with modelmatrix, e.g. Outcome(Surv('time', 'status'), "
                             "strata='inst').fit_transform(D)")
        return y
    # the columns the ids name (the CoxFamily's default: event and status)
    pick = lambda given, role: (given if given is not None and given in y.columns
                                else role if role in y.columns else None)
    columns = {STOP: pick(family.event_id, STOP), STATUS: pick(family.status_id, STATUS),
               START: pick(family.start_id, None) if family.start_id is not None else
               (START if START in y.columns and not legacy(estimator) else None),
               STRATA: pick(family.strata_id, None) if family.strata_id is not None else None,
               # (with the ids, only the columns they name: none by role)
               OFFSET: offset_id if offset_id in y.columns else None,
               WEIGHT: weight_id if weight_id in y.columns else None}
    if columns[STOP] is None or columns[STATUS] is None:
        raise ValueError(f'the y of a Cox model has no columns stop and status (nor '
                         f'{family.event_id!r} and {family.status_id!r}): build it with '
                         "modelmatrix, e.g. Outcome(Surv('time', 'status')).fit_transform(D)")
    if outside:
        warnings.warn("naming the survival data by the CoxFamily's event_id, status_id, "
                      'start_id and strata_id (or the columns event and status), or the '
                      'offset and weights by offset_id and weight_id, is deprecated: name the '
                      "columns of y by their role, as modelmatrix's Outcome does, e.g. "
                      "Y = Outcome(Surv('time', 'status'), strata='inst').fit_transform(D)",
                      FutureWarning, stacklevel=4)
    out = {}
    for role, column in columns.items():
        if column is not None:
            out[role] = y[column] if role == STRATA else y[column].to_numpy()
    return pd.DataFrame(out, index=y.index)


def broadcast_offset(offset, nresponse):
    """A multi-response model's offset `offset`, the same for every response."""
    if offset is not None and np.ndim(offset) == 1:
        return np.repeat(np.asarray(offset, float)[:, None], nresponse, axis=1)
    return offset
