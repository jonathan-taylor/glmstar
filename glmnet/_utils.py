import numpy as np
import pandas as pd

from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from dataclasses import fields
from ._frames import as_pandas
from ._outcome import (_outcome_y,
                       _outcome_ids,
                       _outcome_binomial,
                       _outcome_multinomial)

def _get_data(estimator,
              X,
              y,
              offset_id=None,
              weight_id=None,
              response_id=None,
              check=True,
              multi_output=True # always true below
              ):
    """
    Extracts response, offset, and weights from the input `y`.

    This function handles various input formats for `y` (DataFrame or array)
    and separates the response variable from any specified offset or weight columns.

    Parameters
    ----------
    estimator : object
        The estimator calling this function (for `check_X_y`).
    X : array-like
        The input data matrix.
    y : array-like or pd.DataFrame
        The target variable, which may contain response, offset, and weights.
    offset_id : str, int, or list, optional
        Column identifier for the offset in `y`.
    weight_id : str, int, or list, optional
        Column identifier for the weights in `y`.
    response_id : str, int, or list, optional
        Column identifier for the response in `y`.
    check : bool, default=True
        Whether to perform `sklearn.utils.check_X_y` validation.
    multi_output : bool, default=True
        Whether to allow multi-output for `check_X_y`.

    Returns
    -------
    X : array-like
        The input data matrix (possibly validated).
    y : array-like or pd.DataFrame
        The original target variable.
    response : np.ndarray
        The extracted response variable.
    offset : np.ndarray or None
        The extracted offset term, or None if not provided.
    weight : np.ndarray
        The extracted weights, or an array of ones if not provided.
    """
    X, y = as_pandas(X), as_pandas(y)
    y = _outcome_y(estimator, y)
    response, offset, weight = _extract_y(y,
                                          offset_id=offset_id,
                                          weight_id=weight_id,
                                          response_id=response_id)

    # Validate against Response
    if check:
        X, _ = check_X_y(X, response,
                         accept_sparse=['csc'],
                         multi_output=multi_output,
                         estimator=estimator)

    return X, y, np.squeeze(response), offset, weight

def _extract_y(y,
               offset_id=None,
               weight_id=None,
               response_id=None):
    """
    The response, offset and weights of `y`, from their columns.

    With `response_id` None, the response is every column of `y` other
    than the offset and weights (or `y` itself if it is 1-dimensional).

    Parameters
    ----------
    y : array-like or pd.DataFrame
        The target variable, which may contain response, offset, and weights.
    offset_id, weight_id, response_id : str, int, or list, optional
        Column identifiers in `y`.

    Returns
    -------
    response : np.ndarray
        The response, not squeezed.
    offset : np.ndarray or None
        The offset, or None if not provided.
    weight : np.ndarray
        The weights, or an array of ones if not provided.
    """
    y = as_pandas(y)
    is_df = isinstance(y, pd.DataFrame)

    def _as_list(val):
        """Helper to ensure column identifiers are lists for dropping/indexing."""
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
        return [val]

    # Extract Offset
    offset = None
    if offset_id is not None:
        offset = np.asarray(y.loc[:, offset_id] if is_df else y[:, offset_id])

    # Extract Weight
    weight = None
    if weight_id is not None:
        weight = np.asarray(y.loc[:, weight_id] if is_df else y[:, weight_id])

    # Extract Response
    if response_id is not None:
        response = np.asarray(y.loc[:, response_id] if is_df else y[:, response_id])
    else:
        if is_df:
            drop_cols = []
            if offset_id is not None:
                drop_cols.extend(_as_list(offset_id))
            if weight_id is not None:
                drop_cols.extend(_as_list(weight_id))
            response = np.asarray(y.drop(columns=drop_cols) if drop_cols else y)
        else:
            if y.ndim == 1:
                response = np.asarray(y)
            else:
                keep = np.ones(y.shape[1], dtype=bool)
                if offset_id is not None:
                    keep[_as_list(offset_id)] = False
                if weight_id is not None:
                    keep[_as_list(weight_id)] = False
                response = np.asarray(y[:, keep])

    # Default Weights
    if weight is None:
        weight = np.ones(y.shape[0])

    return response, offset, weight

def _y_ids(estimator):
    """
    The column ids of `y` an estimator reads: with an outcome, those of
    the `y` it builds.
    """
    if getattr(estimator, 'outcome', None) is not None:
        return _outcome_ids(estimator.outcome_)
    return dict(offset_id=estimator.offset_id,
                weight_id=estimator.weight_id,
                response_id=estimator.response_id)

def _sub_ids(estimator):
    """
    The columns of `y` of an estimator, for another it builds (e.g. at each
    point of a path, or `get_fixed_lambda`'s): its fitted outcome, used as it
    is, or its ids.
    """
    if getattr(estimator, 'outcome', None) is not None:
        return dict(outcome=estimator.outcome_)
    return dict(offset_id=estimator.offset_id,
                weight_id=estimator.weight_id,
                response_id=estimator.response_id)

def _forget_encoding(estimator):
    """Drop the classes of a previous fit, so that `fit` learns them again."""
    for attr in ('classes_', 'categories_'):
        estimator.__dict__.pop(attr, None)

def _binomial_response(estimator,
                       response,
                       weight,
                       proportion=False):
    """
    The labels (in [0, 1]), weights and classes of a binomial response.

    The response is a vector of labels of two classes, a 2-column array of
    (trials, successes), whose trials multiply the weights, or (with
    `proportion`) a vector of proportions. The classes of labels are
    learned when the estimator has none, i.e. at `fit`; otherwise the
    labels are coded by them, e.g. those of a fold being scored, which
    may not have both classes.

    Returns
    -------
    labels, weight, classes : np.ndarray
    """
    outcome = getattr(estimator, 'outcome_', None)
    if outcome is not None and getattr(estimator, 'outcome', None) is not None:
        coded = _outcome_binomial(outcome, response, weight)
        if coded is not None:
            return coded
    if response.ndim == 2 and response.shape[1] == 2:
        trials = response[:, 0]
        successes = response[:, 1]
        if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
            raise ValueError("For (trials, successes) input, trials must be > 0 and 0 <= successes <= trials.")
        return successes / trials, weight * trials, np.array([0, 1])
    if proportion and _is_proportion(response):
        return np.asfortranarray(np.asarray(response, float).ravel()), weight, np.array([0, 1])
    if response.ndim == 2 and response.shape[1] == 1:
        response = response.ravel()
    encoder = LabelEncoder()
    classes = getattr(estimator, 'classes_', None)
    if classes is None:
        encoder.fit(response)
        if len(encoder.classes_) > 2:
            raise ValueError(f"{type(estimator).__name__} expecting a binary classification problem.")
    else:
        encoder.classes_ = np.asarray(classes)
    return np.asfortranarray(encoder.transform(response)), weight, encoder.classes_

def _multinomial_response(estimator,
                          response,
                          weight):
    """
    The one-hot response (proportions), weights and categories of a
    multinomial response: a vector of labels, or a 2-column (or more)
    array of counts of each class, whose row sums multiply the weights.
    The categories of labels are learned when the estimator has none,
    i.e. at `fit`; otherwise the labels are coded by them.

    Returns
    -------
    y_onehot, weight, categories : np.ndarray
    """
    outcome = getattr(estimator, 'outcome_', None)
    if outcome is not None and getattr(estimator, 'outcome', None) is not None:
        coded = _outcome_multinomial(outcome, response, weight)
        if coded is not None:
            return coded
    if response.ndim == 2 and response.shape[1] > 1:
        trials = response.sum(axis=1)
        if np.any(response < 0) or np.any(trials <= 0):
            raise ValueError("Response matrix must contain non-negative counts and row sums must be > 0.")
        return (np.asfortranarray(response / trials[:, None]), weight * trials,
                np.arange(response.shape[1]))
    if response.ndim == 2 and response.shape[1] == 1:
        response = response.ravel()
    categories = getattr(estimator, 'categories_', None)
    encoder = OneHotEncoder(sparse_output=False,
                            categories='auto' if categories is None else [categories])
    y_onehot = encoder.fit_transform(np.asarray(response).reshape((-1, 1)))
    return np.asfortranarray(y_onehot), weight, encoder.categories_[0]

def _is_proportion(response):
    """A numeric response in [0, 1] with some value strictly between 0 and 1."""
    response = np.asarray(response)
    if response.ndim == 2 and response.shape[1] == 1:
        response = response.ravel()
    if response.ndim != 1 or not np.issubdtype(response.dtype, np.number):
        return False
    return bool(np.all((response >= 0) & (response <= 1)) and
                np.any((response > 0) & (response < 1)))

class UnboundedObjectiveError(ValueError):
    """
    The objective, with its linear term, is unbounded below: no solution exists.
    """


def _jerr_elnetfit(n, maxit, k=None, pmax=None):
    """
    Interprets error codes from `elnet` (C++ or Fortran) routines.

    Parameters
    ----------
    n : int
        The error code returned by the `elnet` routine.
    maxit : int
        Maximum number of iterations set for the `elnet` routine.
    k : int, optional
        The index of the lambda value for which convergence failed.
    pmax : int, optional
        Maximum number of variables ever nonzero (`nx`), used in the
        message when it is exceeded.

    Returns
    -------
    dict
        A dictionary containing the error code, a boolean indicating if it's a
        fatal error, and a descriptive message.
    """
    if k is None and -10000 < n < 0:
        k = -n # index of the lambda value that did not converge, as in R
    unbounded = False
    if n == 0:
        fatal = False
        msg = ''
    elif -50000 < n <= -40001:
        # only with a linear term
        fatal = False
        unbounded = True
        msg = ("The objective with the linear term is unbounded below " +
               f"(or did not converge after maxit={maxit} iterations) at the {-n-40000}-th lambda value")
    elif n > 0:
        # fatal error
        fatal = True
        msg =(f"Memory allocation error; contact package maintainer" if n < 7777 else
              "Unknown error")
    elif n < -10000:
        fatal = False
        msg = (f"Number of nonzero coefficients along the path exceeds pmax={pmax}" +
               f" at {-n-10000}-th lambda value; solutions for larger lambdas returned")
    else:
        fatal = False
        msg = (f"Convergence for {k}-th lambda value not reached after maxit={maxit}" +
               " iterations; solutions for larger lambdas returned")
    return {'n':n,
            'fatal':fatal,
            'unbounded':unbounded,
            'msg':f"Error code {n}:" + msg}

def _jerr_lognet(n, maxit, k=None, pmax=None):
    """
    Interprets error codes from the logistic and multinomial path routines
    (as in R's `jerr.lognet`).

    Parameters
    ----------
    n : int
        The error code returned by the `lognet` routine.
    maxit : int
        Maximum number of iterations set for the `lognet` routine.
    k : int, optional
        The index of the lambda value for which convergence failed.
    pmax : int, optional
        Maximum number of variables ever nonzero (`nx`).

    Returns
    -------
    dict
        A dictionary containing the error code, a boolean indicating if it's a
        fatal error, and a descriptive message.
    """
    if n < -20000:
        msg = (f"Max(p(1-p),1.0e-6 at {-n-20000}-th value of lambda;" +
               " solutions for larger values of lambda returned")
        return {'n':n,
                'fatal':False,
                'msg':f"Error code {n}:" + msg}
    if 8000 < n < 9000:
        msg = f"Null probability for class {n-8000} < 1.0e-5"
    elif 9000 < n < 10000:
        msg = f"Null probability for class {n-9000} > 1.0 - 1.0e-5"
    else:
        return _jerr_elnetfit(n, maxit, k, pmax=pmax)
    return {'n':n,
            'fatal':True,
            'msg':f"Error code {n}:" + msg}

def _jerr_coxnet(n, maxit, k=None, pmax=None):
    """
    Interprets error codes from the Cox path routine (as in R's `jerr.coxnet`).

    Parameters
    ----------
    n : int
        The error code returned by the `coxnet` routine.
    maxit : int
        Maximum number of iterations set for the `coxnet` routine.
    k : int, optional
        The index of the lambda value for which convergence failed.
    pmax : int, optional
        Maximum number of variables ever nonzero (`nx`).

    Returns
    -------
    dict
        A dictionary containing the error code, a boolean indicating if it's a
        fatal error, and a descriptive message.
    """
    if n > 0:
        if n == 8888:
            msg = "All observations censored - cannot proceed"
        elif n == 9999:
            msg = "No positive observation weights"
        elif n in [20000, 30000]:
            msg = "Initialization numerical error; probably too many censored observations"
        else:
            return _jerr_elnetfit(n, maxit, k, pmax=pmax)
        return {'n':n,
                'fatal':True,
                'msg':f"Error code {n}:" + msg}
    elif n <= -30000:
        msg = (f"Numerical error at {-n-30000}-th lambda value;" +
               " solutions for larger values of lambda returned")
        return {'n':n,
                'fatal':False,
                'msg':f"Error code {n}:" + msg}
    if k is None and -10000 < n < 0:
        k = -n # index of the lambda value that did not converge, as in R
    return _jerr_elnetfit(n, maxit, k, pmax=pmax)

def _parent_dataclass_from_child(cls,
                                 parent_dict,
                                 **modified_args):
    """
    Creates an instance of a dataclass from a dictionary, filtering keys.

    This utility function is useful when initializing a dataclass with arguments
    that might come from a larger dictionary, ensuring only relevant keys are used.

    Parameters
    ----------
    cls : type
        The dataclass type to instantiate.
    parent_dict : dict
        A dictionary containing potential arguments for the dataclass.
    **modified_args
        Additional arguments to update or override values from `parent_dict`.

    Returns
    -------
    object
        An instance of the specified dataclass.
    """
    _fields = [f.name for f in fields(cls)]
    _cls_args = {k:parent_dict[k] for k in parent_dict.keys() if k in _fields}
    _cls_args.update(**modified_args)
    return cls(**_cls_args)

# ===================================================================
# ARGUMENT DICTIONARY EXTRACTED FROM C++ WRAPPER FILES
# ===================================================================

# Using the dictionary structure you suggested for better organization
ALL_CPP_ARGS = {
    # From: elnet_point.cpp
    "elnet_point": [
        'alm0', 'almc', 'alpha', 'm', 'no', 'ni', 'x', 'r', 'xv', 'v',
        'intr', 'ju', 'vp', 'cl', 'nx', 'thr', 'maxit', 'a', 'aint', 'g',
        'ia', 'iy', 'iz', 'mm', 'nino', 'rsqc', 'nlp', 'jerr'
    ],
    "spelnet_point": [
        'alm0', 'almc', 'alpha', 'm', 'no', 'ni', 'x_data_array',
        'x_indices_array', 'x_indptr_array', 'xm', 'xs', 'r', 'xv', 'v',
        'intr', 'ju', 'vp', 'cl', 'nx', 'thr', 'maxit', 'a', 'aint', 'g',
        'ia', 'iy', 'iz', 'mm', 'nino', 'rsqc', 'nlp', 'jerr'
    ],

    # From: lognet.cpp
    "lognet": [
        'parm', 'ni', 'no', 'x', 'y', 'g', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'kopt', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "splognet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'g', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'kopt', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: gaussnet.cpp
    "gaussnet": [
        'ka', 'parm', 'ni', 'no', 'x', 'y', 'w', 'jd', 'vp', 'cl', 'ne',
        'nx', 'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spgaussnet": [
        'ka', 'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'intr', 'pb', 'lmu', 'a0', 'ca',
        'ia', 'nin', 'rsq', 'maxit', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    
    # From: multigaussnet.cpp
    "multigaussnet": [
        'parm', 'ni', 'no', 'x', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'jsd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spmultigaussnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam',
        'flmin', 'ulam', 'thr', 'isd', 'jsd', 'intr', 'maxit', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'rsq', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: fishnet.cpp
    "fishnet": [
        'parm', 'ni', 'no', 'x', 'y', 'w', 'g', 'jd', 'vp', 'cl', 'ne',
        'nx', 'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb',
        'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spfishnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'y', 'w', 'g', 'jd', 'vp', 'cl', 'ne', 'nx',
        'nlam', 'flmin', 'ulam', 'thr', 'isd', 'intr', 'maxit', 'pb', 'lmu',
        'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev', 'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],

    # From: coxnet.cpp
    "coxnet": [
        'parm', 'ni', 'no', 'x', 'start', 'stop', 'status', 'strata', 'efron',
        'g', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam', 'flmin', 'ulam', 'thr',
        'isd', 'maxit', 'pb', 'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev',
        'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ],
    "spcoxnet": [
        'parm', 'ni', 'no', 'x_data_array', 'x_indices_array',
        'x_indptr_array', 'start', 'stop', 'status', 'strata', 'efron',
        'g', 'w', 'jd', 'vp', 'cl', 'ne', 'nx', 'nlam', 'flmin', 'ulam', 'thr',
        'isd', 'maxit', 'pb', 'lmu', 'a0', 'ca', 'ia', 'nin', 'nulldev', 'dev',
        'alm', 'nlp', 'jerr',
        'fdev', 'eps', 'big', 'mnlam', 'devmax', 'pmin', 'exmx', 'itrace',
        'prec', 'mxit', 'epsnr', 'mxitnr'
    ]
}


# ===================================================================
# VALIDATION FUNCTION (Unchanged, it works with the new structure)
# ===================================================================

def _validate_cpp_args(
    args_dict: dict,
    function_name: str
) -> bool:
    """
    Compares a dictionary of arguments against a list of expected names.
    """
    expected_set = set(ALL_CPP_ARGS[function_name])
    actual_set = set(args_dict.keys())

    missing_args = expected_set - actual_set
    extra_args = actual_set - expected_set

    msg = ''
    if missing_args:
        msg = f"ERROR: The following arguments for {function_name} are MISSING: {sorted(list(missing_args))}, "
    if extra_args:
        msg += f"ERROR: The following EXTRA arguments for {function_name} were found: {sorted(list(extra_args))}"
    if msg:
        return msg



def _check_offset(offset, n, ncol=None):
    """
    Validate an offset passed to `predict` (R's `newoffset`).

    Parameters
    ----------
    offset : array-like
        Offset for `n` new observations: shape `(n,)`, or `(n, ncol)` for
        multi-response models (a vector is then used for every response).
    n : int
        Number of rows being predicted.
    ncol : int, optional
        Number of responses for multi-response models.

    Returns
    -------
    np.ndarray
        Offset of shape `(n,)`, or `(n, ncol)` if `ncol` is given.
    """
    offset = np.asarray(offset, dtype=float)
    if ncol is not None and offset.ndim == 1:
        offset = np.repeat(offset[:, None], ncol, axis=1)
    expected = (n,) if ncol is None else (n, ncol)
    if offset.ndim == 2 and ncol is None and offset.shape[1] == 1:
        offset = offset[:, 0]
    if offset.shape != expected:
        raise ValueError(f'offset should have shape {expected}, got {offset.shape}')
    return offset


def _check_predict_X(X):
    """
    The feature matrix passed to `predict`, as `fit` sees it: a pandas
    DataFrame is converted to a float array (as `check_X_y` does in
    `fit`); arrays and sparse matrices are returned unchanged.
    """
    if isinstance(X, pd.DataFrame):
        return np.asarray(X, dtype=float)
    return X
