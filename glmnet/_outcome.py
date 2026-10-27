"""
The columns of `y` named by an `Outcome` of modelmatrix.

An estimator given `outcome=Outcome(...)` fits it on the `y` of `fit`
(a copy, unless it is already fitted) and builds each `y` it is given
with it: a frame whose columns are named by their role (the response,
`offset`, `weight`, and for a Cox model `start`, `stop`, `status`,
`strata`). The estimator then reads these columns by their names, as it
reads the columns of `response_id`, `offset_id` and `weight_id`.
"""

from copy import deepcopy
import sys
import warnings

import numpy as np
import pandas as pd

from modelmatrix import Outcome, Response, Surv, Binomial, Multinomial

# the attribute of a frame built by an outcome: the fitted outcome, so that
# the frame is not built again, and an estimator fit to it (e.g. a clone, on
# a fold) takes that outcome
_BUILT = '_glmstar_outcome'


def _fit_outcome(estimator, y):
    """
    At `fit`: fit (a copy of) the estimator's outcome on `y`, as
    `outcome_`, and build `y` with it. Without an outcome, `y` as it is.
    """
    outcome = getattr(estimator, 'outcome', None)
    if outcome is None:
        estimator.__dict__.pop('outcome_', None)
        if _called_from_outside():
            _warn_ids(estimator, y)
        return y
    if _is_built(y):   # built by an enclosing fit
        estimator.outcome_ = y.attrs[_BUILT]
        return y
    if isinstance(outcome, Response) and not isinstance(outcome, Outcome):
        outcome = Outcome(outcome)   # a response alone, e.g. Surv('time', 'status')
    if not isinstance(outcome, Outcome):
        raise TypeError('outcome should be a modelmatrix Outcome or response (Surv, '
                        f'Binomial, Multinomial), got {type(outcome).__name__}')
    given = [name for name in ['response_id', 'offset_id', 'weight_id']
             if getattr(estimator, name, None) is not None]
    if given:
        raise ValueError(f'give the columns of y by outcome or by {", ".join(given)}, not both')
    family = getattr(estimator, 'family', None)
    cox = hasattr(family, 'event_id')   # a CoxFamily
    if cox and (family.event_id, family.status_id,
                family.start_id, family.strata_id) != ('event', 'status', None, None):
        raise ValueError('give the survival data by outcome (a Surv response, and '
                         'strata=...) or by the ids of the CoxFamily, not both')
    if isinstance(outcome.response, Surv) and not cox:
        raise ValueError(f'an Outcome of a Surv response is for a Cox model, not '
                         f'{type(estimator).__name__} with family {family!r}')
    outcome = deepcopy(outcome)
    if not hasattr(outcome, 'columns_'):   # not fitted
        outcome.fit(_as_frame(y))
    estimator.outcome_ = outcome
    return _outcome_y(estimator, y)


def _called_from_outside():
    # is the fit calling _fit_outcome called from outside glmstar (not a fit
    # of glmstar's own, e.g. a sub-estimator's, or a fit calling its parent's)?
    caller = sys._getframe(3).f_globals.get('__name__', '')
    return not (caller == 'glmnet' or caller.startswith('glmnet.'))


def _warn_ids(estimator, y):
    """The FutureWarnings of a fit without an outcome that names columns of `y`."""
    given = [name for name in ['response_id', 'offset_id', 'weight_id']
             if getattr(estimator, name, None) is not None]
    if given:
        warnings.warn(f'{", ".join(given)} {"are" if len(given) > 1 else "is"} deprecated: '
                      'give the columns of y by outcome=, a modelmatrix Outcome, e.g. '
                      "outcome=Outcome('y', offset='O', weight='w')",
                      FutureWarning, stacklevel=4)
    if hasattr(getattr(estimator, 'family', None), 'event_id'):   # a CoxFamily
        warnings.warn("the survival data by the CoxFamily's event_id, status_id, start_id and "
                      'strata_id are deprecated: give them by outcome=, e.g. '
                      "outcome=Surv('time', 'status') or outcome=Outcome(Surv('start', "
                      "'stop', 'status'), strata='inst', weight='w') (modelmatrix)",
                      FutureWarning, stacklevel=4)
    elif (getattr(estimator, 'response_id', None) is None and isinstance(y, pd.DataFrame)
          and y.shape[1] - sum(getattr(estimator, name, None) is not None
                               for name in ['offset_id', 'weight_id']) > 1):
        warnings.warn('the response taken as every column of y but the offset and weights is '
                      'deprecated: give the response by outcome=, e.g. '
                      "outcome=Outcome(['y1', 'y2']) or outcome=Binomial('successes', "
                      "trials='trials') (modelmatrix)", FutureWarning, stacklevel=4)


def _outcome_y(estimator, y):
    """
    `y` built by the fitted outcome of the estimator (once: a frame it
    built is left as it is). Without an outcome, `y` as it is.
    """
    if getattr(estimator, 'outcome', None) is None or y is None or _is_built(y):
        return y
    outcome = getattr(estimator, 'outcome_', None)
    if outcome is None:
        raise ValueError(f'this {type(estimator).__name__} has an outcome but is not '
                         'fitted: call fit first')
    Y = outcome.transform(_as_frame(y))
    binomial = outcome.response_
    if (isinstance(binomial, Binomial) and binomial.trials is not None
        and binomial.output == 'counts'):
        # (successes, failures) as R's cbind: the proportion of successes,
        # its trials multiplying the weights, as with output='proportion'
        s, n = binomial.successes, binomial.trials
        successes = Y.pop(s).to_numpy(float)
        trials = successes + Y.pop(f'{n}-{s}').to_numpy(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            Y.insert(0, f'{s}/{n}', successes / trials)
        Y['weight'] = trials * (Y['weight'].to_numpy(float) if 'weight' in Y.columns else 1)
    Y.attrs[_BUILT] = outcome
    return Y


def _is_built(y):
    return isinstance(y, pd.DataFrame) and y.attrs.get(_BUILT) is not None


def _as_frame(y):
    # an Outcome builds a frame from a frame (columns of an array: integers)
    if isinstance(y, pd.DataFrame):
        return y
    if isinstance(y, pd.Series):
        return y.to_frame()
    return pd.DataFrame(np.asarray(y))


def _response_columns(outcome):
    """The names of the columns of the response built by a fitted outcome."""
    response = outcome.response_
    if isinstance(response, Surv):
        return ['start', 'stop', 'status'] if response.start is not None else ['stop', 'status']
    if isinstance(response, Binomial) and response.trials is not None:
        # with output='counts' too, as _outcome_y builds it
        return [f'{response.successes}/{response.trials}']
    if isinstance(response, Multinomial):
        if isinstance(response.classes, list):
            return list(response.classes)
        if response.output == 'onehot':
            return [f'{response.classes}[{l}]' for l in response.levels_]
        return [response.classes]
    return list(response.columns)


def _outcome_ids(outcome):
    """
    The ids of the columns of `y` built by a fitted outcome. A
    multi-response model's offset is the column `offset`, the same for
    every response, or the columns `offset[...]`, one for each response
    in their order (an Outcome's `offset={...}`).
    """
    response = _response_columns(outcome)
    others = list(outcome.entries_)
    if isinstance(outcome.response_, Binomial) and outcome.response_.trials is not None:
        others.append('weight')   # the trials
    offsets = [name for name in others if name.startswith('offset[')]
    if offsets and 'offset' in others:
        raise ValueError(f'an offset for every response and one for each, {offsets}: '
                         'give one or the other')
    if any(name.startswith('weight[') for name in others):
        raise ValueError('weights of each response are not supported: the weights '
                         'are of the observations, weight=...')
    return dict(offset_id=offsets or ('offset' if 'offset' in others else None),
                weight_id='weight' if 'weight' in others else None,
                response_id=response[0] if len(response) == 1 else response)


def _broadcast_offset(offset, nresponse):
    """
    An Outcome's offset of a multi-response model: the column `offset`
    is the same for every response, the columns `offset[...]` are one
    for each.
    """
    if offset is not None and np.ndim(offset) == 1:
        return np.repeat(np.asarray(offset, float)[:, None], nresponse, axis=1)
    return offset


def _outcome_cox_ids(outcome):
    """The ids of the survival data in `y` built by a fitted outcome."""
    response = outcome.response_
    if not isinstance(response, Surv):
        raise ValueError(f'a Cox model needs an Outcome of a Surv response, got {response!r}')
    return dict(event_id='stop',
                status_id='status',
                start_id='start' if response.start is not None else None,
                strata_id='strata' if 'strata' in outcome.extra else None)


def _cox_ids(estimator):
    """
    The ids of the survival data in `y` of a Cox estimator: those of its
    `CoxFamily`, or with an outcome, those of the `y` it builds.
    """
    if getattr(estimator, 'outcome', None) is not None:
        return _outcome_cox_ids(estimator.outcome_)
    family = estimator.family
    return dict(event_id=family.event_id,
                status_id=family.status_id,
                start_id=family.start_id,
                strata_id=family.strata_id)


def _outcome_binomial(outcome, response, weight):
    """
    The labels, weights and classes of a binomial response built by an
    outcome, or None if its response is not a `Binomial`: 0/1 (or a
    proportion), weighted by any trials, the classes the levels of the
    `Binomial` (failure, success).
    """
    binomial = outcome.response_
    if not isinstance(binomial, Binomial):
        return None
    if binomial.trials is not None:   # a proportion, its trials in the weights
        return np.asfortranarray(np.asarray(response, float)), weight, np.array([0, 1])
    levels = list(binomial.levels_)
    if len(levels) == 2:
        failure = levels[0] if levels[1] == binomial.level_ else levels[1]
        classes = np.array([failure, binomial.level_])
    else:
        classes = np.array([0, 1])
    return np.asfortranarray(np.asarray(response, float)), weight, classes


def _outcome_multinomial(outcome, response, weight):
    """
    The one-hot response (proportions), weights and categories of a
    multinomial response built by an outcome, or None if its response is
    not a `Multinomial`: the categories are its levels.
    """
    multinomial = outcome.response_
    if not isinstance(multinomial, Multinomial):
        return None
    if isinstance(multinomial.classes, list) or multinomial.output == 'onehot':
        counts = np.asarray(response, float)
        trials = counts.sum(axis=1)
        if np.any(counts < 0) or np.any(trials <= 0):
            raise ValueError('the counts of a Multinomial should be nonnegative, '
                             'with a positive total')
        categories = (np.asarray(multinomial.classes) if isinstance(multinomial.classes, list)
                      else np.asarray(multinomial.levels_))
        return np.asfortranarray(counts / trials[:, None]), weight * trials, categories
    categories = np.asarray(multinomial.levels_)
    codes = pd.Index(categories).get_indexer(np.asarray(response, dtype=object))
    y_onehot = np.equal.outer(codes, np.arange(len(categories))).astype(float)
    return np.asfortranarray(y_onehot), weight, categories
