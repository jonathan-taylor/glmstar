"""
Assessing fitted paths on new data, as R's ``assess.glmnet``,
``confusion.glmnet`` and ``roc.glmnet``.
"""

import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
from statsmodels.genmod.families import family as sm_family
from statsmodels.genmod.families import links as sm_links

from ._utils import _get_data
from .cox import CoxFamilySpec, c_index
from .paths.multiclassnet import MultiClassNet
from .paths.multigaussnet import MultiGaussNet

# R's measures (glmnet.measures) and the score_path columns that compute them
_GAUSSIAN = {'mse': 'Mean Squared Error (Ungrouped)',
             'mae': 'Mean Absolute Error (Ungrouped)'}
_MEASURES = {
    'gaussian': _GAUSSIAN,
    'binomial': {'deviance': 'Binomial Deviance',
                 'class': 'Accuracy',
                 'auc': 'AUC',
                 'mse': 'Mean Squared Error (Ungrouped)',
                 'mae': 'Mean Absolute Error (Ungrouped)'},
    'multinomial': {'deviance': 'Multinomial Deviance',
                    'class': 'Misclassification Error',
                    'mse': 'Mean Squared Error',
                    'mae': 'Mean Absolute Error'},
    'mgaussian': {'mse': 'Mean Squared Error',
                  'mae': 'Mean Absolute Error'},
}

def _family_name(estimator):
    """R's name for the family of a fitted estimator."""
    family = estimator._family
    if isinstance(family, CoxFamilySpec):
        return 'cox'
    if isinstance(estimator, MultiClassNet):
        return 'multinomial'
    if isinstance(estimator, MultiGaussNet):
        return 'mgaussian'
    base = getattr(family, 'base', None)
    if isinstance(base, sm_family.Gaussian) and isinstance(base.link, sm_links.Identity):
        return 'gaussian'
    if isinstance(base, sm_family.Binomial) and isinstance(base.link, sm_links.Logit):
        return 'binomial'
    if isinstance(base, sm_family.Poisson) and isinstance(base.link, sm_links.Log):
        return 'poisson'
    return 'GLM'

def _link_predictions(estimator, X, y):
    """Linear predictor on the path (including any offset), raw response and weights."""
    check_is_fitted(estimator, ["coefs_"])
    _, _, response, offset, weight = _get_data(estimator,
                                               X,
                                               y,
                                               offset_id=estimator.offset_id,
                                               response_id=estimator.response_id,
                                               weight_id=estimator.weight_id,
                                               check=False,
                                               multi_output=True)
    nfit = estimator.lambda_values_.shape[0]
    link = estimator.predict(X, prediction_type='link')[:, :nfit]
    if offset is not None:
        offset = np.asarray(offset, float)
        if link.ndim == 3:
            link = link + offset.reshape((offset.shape[0], 1, -1))
        else:
            link = link + offset.reshape((-1, 1))
    if weight is None:
        weight = np.ones(link.shape[0])
    return link, response, np.asarray(weight, float)

def assess(estimator, X, y):
    """
    Performance measures of a fitted path on new data.

    The analogue of R's ``assess.glmnet``: for each lambda on the path, the
    measures R computes for the family (see ``glmnet.measures``), with R's
    names and conventions. These are `score_path` scores, except that for
    the binomial family "class" is the misclassification error and "mse" and
    "mae" are summed over the two classes, and for Cox "deviance" is the total
    (not average) partial likelihood deviance.

    Parameters
    ----------
    estimator : GLMNet
        A fitted path estimator.
    X : array-like or sparse matrix
        New feature matrix.
    y : array-like or pd.DataFrame
        New response, with any weight or offset columns used in the fit.

    Returns
    -------
    pd.DataFrame
        One row per lambda on the path, one column per measure.
    """
    family = _family_name(estimator)
    index = pd.Index(estimator.lambda_values_, name='lambda')

    if family == 'cox':
        link, _, weight = _link_predictions(estimator, X, y)
        fam = estimator._finalize_family(y)
        surv = estimator._get_survival_data(y)
        start = surv['start'] if estimator.family.start_id is not None else None
        # R's coxnet.deviance standardizes the weights to sum to n
        std_weight = weight * weight.shape[0] / weight.sum()
        deviance = [fam._coxdev(link[:, j], std_weight).deviance for j in range(link.shape[1])]
        C = c_index(link, surv['stop'], surv['status'], start=start, sample_weight=weight)
        return pd.DataFrame({'deviance': deviance, 'C': C}, index=index)

    scores = estimator.score_path(X, y).scores
    if family in _MEASURES:
        columns = _MEASURES[family]
    else:
        # poisson and other GLM families: R's measures are deviance, mse and mae
        deviance = [c for c in scores.columns if c.endswith('Deviance')]
        columns = dict(deviance=deviance[0], **_GAUSSIAN)

    value = pd.DataFrame({k: np.asarray(scores[v]) for k, v in columns.items()}, index=index)
    if family == 'binomial':
        value['class'] = 1 - value['class']
        # R uses a two column (failure, success) response for mse and mae
        value['mse'] *= 2
        value['mae'] *= 2
    return value

def _classes(estimator):
    return estimator.categories_ if isinstance(estimator, MultiClassNet) else estimator.classes_

def _true_labels(estimator, response):
    response = np.asarray(response)
    if response.ndim == 2 and response.shape[1] > 1:
        # indicator (or count) matrix: the class with the largest entry
        return np.asarray(_classes(estimator))[np.argmax(response, 1)]
    return response.reshape(-1)

def confusion(estimator, X, y):
    """
    Confusion matrices of a fitted classifier on new data.

    The analogue of R's ``confusion.glmnet``, for `LogNet`, `MultiClassNet`
    and other binomial fits: for each lambda on the path, the (unweighted)
    table of predicted against true classes. A binomial prediction is the
    second class when the linear predictor is positive.

    Parameters
    ----------
    estimator : GLMNet
        A fitted binomial or multinomial path estimator.
    X : array-like or sparse matrix
        New feature matrix.
    y : array-like or pd.DataFrame
        New response, with any weight or offset columns used in the fit.

    Returns
    -------
    list of pd.DataFrame
        One table per lambda, with rows "Predicted" and columns "True".
    """
    family = _family_name(estimator)
    if family not in ['binomial', 'multinomial']:
        raise ValueError('confusion is available only for binomial or multinomial fits')
    link, response, _ = _link_predictions(estimator, X, y)
    classes = np.asarray(_classes(estimator))
    true = pd.Categorical(_true_labels(estimator, response), categories=classes)

    tables = []
    for j in range(link.shape[1]):
        if family == 'binomial':
            predicted = classes[(link[:, j] > 0).astype(int)]
        else:
            predicted = classes[np.argmax(link[:, j], -1)]
        predicted = pd.Categorical(predicted, categories=classes)
        tables.append(pd.crosstab(pd.Series(predicted, name='Predicted'),
                                  pd.Series(true, name='True'),
                                  dropna=False))
    return tables

def roc(estimator, X, y):
    """
    ROC curves of a fitted binomial path on new data.

    The analogue of R's ``roc.glmnet``: for each lambda on the path, the
    (unweighted) false and true positive rates as the threshold on the
    linear predictor decreases, with tied predictions merged into one
    point. The positive class is the second of `classes_`.

    Parameters
    ----------
    estimator : GLMNet
        A fitted binomial path estimator.
    X : array-like or sparse matrix
        New feature matrix.
    y : array-like or pd.DataFrame
        New binary response, with any weight or offset columns used in the fit.

    Returns
    -------
    list of pd.DataFrame
        One table per lambda, with columns "FPR" and "TPR".
    """
    if _family_name(estimator) != 'binomial':
        raise ValueError('roc is available only for binomial fits')
    link, response, _ = _link_predictions(estimator, X, y)
    labels = _true_labels(estimator, response)
    classes = np.asarray(estimator.classes_)
    if not np.all(np.isin(labels, classes)):
        raise ValueError('roc needs a binary response with the classes of the fit')
    positive = (labels == classes[1]).astype(float)

    curves = []
    for j in range(link.shape[1]):
        # distinct predictions in decreasing order, with the number of
        # positives and negatives at each
        values, inverse = np.unique(-link[:, j], return_inverse=True)
        pos = np.bincount(inverse, weights=positive, minlength=values.shape[0])
        neg = np.bincount(inverse, weights=1 - positive, minlength=values.shape[0])
        curves.append(pd.DataFrame({'FPR': np.cumsum(neg) / neg.sum(),
                                    'TPR': np.cumsum(pos) / pos.sum()}))
    return curves
