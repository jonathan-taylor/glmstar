from itertools import product
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import warnings

from sklearn.model_selection import (cross_val_predict,
                                     check_cv,
                                     KFold)

from .glm import GLMFamilySpec

@dataclass
class PathScorer(object):
    """Scorer for path-based model evaluation.
    
    Parameters
    ----------
    data : tuple
        Tuple of (response, original_y) data.
    predictions : np.ndarray
        Array of predictions for all lambda values.
    family : GLMFamilySpec, optional
        GLM family specification.
    sample_weight : np.ndarray, optional
        Sample weights.
    splits : list, default_factory=list
        List of test splits for cross-validation.
    compute_std_error : bool, default=True
        Whether to compute standard errors.
    index : pd.Series, optional
        Index for the lambda values.
    complexity_order : str, default='increasing'
        Order of complexity ('increasing' or 'decreasing').
    """

    data: tuple
    predictions: np.ndarray
    family: GLMFamilySpec = field(default_factory=GLMFamilySpec)
    sample_weight: np.ndarray = None
    splits: list = field(default_factory=list)
    compute_std_error: bool = True
    index: pd.Series = None
    complexity_order: str = 'increasing'
    
    def __post_init__(self):
        """Initialize the PathScorer object."""
        if type(self.index) == np.ndarray:
            self.index = pd.Series(self.index,
                                   index=np.arange(self.index.shape[0]))

    def compute_scores(self,
                       scorers=[]):
        """Compute cross-validation scores.
        
        Parameters
        ----------
        scorers : list, default=[]
            List of scorers to use.
            
        Returns
        -------
        tuple
            Tuple of (cv_scores, index_best, index_1se).
        """

        self.scorers = list(set(scorers).union(self.family._default_scorers()))

        score_dict = self._get_scores(self.predictions,
                                      self.sample_weight,
                                      self.splits,
                                      self.scorers)
        df_dict = {}

        for scorer in score_dict.keys():
            val_W = score_dict[scorer]
            val = val_W[:,:,0]
            W = val_W[:,:,1]
            mask = ~np.isnan(val)
            count = mask.sum(1)
            # missing scores (failed folds) are excluded; NaN * 0 is NaN, so zero them first
            val = np.where(mask, val, 0)
            mean_ = np.sum(val * mask * W, 1) / np.sum(mask * W, 1)
            df_dict[scorer.name] = mean_

            if self.compute_std_error:
                resid = val - mean_[:,None]
                std_ = np.sqrt(np.sum(resid**2 * mask * W, 1) / (np.sum(mask * W, 1) * (count-1)))
                df_dict[f'SD({scorer.name})'] = std_
            else:
                std_ = None


        self.cv_scores_ = pd.DataFrame(df_dict,
                                       index=self.index)

        index_best_, index_1se_ = _tune(self.index,
                                        self.scorers,
                                        self.cv_scores_,
                                        complexity_order=self.complexity_order,
                                        compute_std_error=self.compute_std_error)

        return self.cv_scores_, index_best_, index_1se_
    
    def _get_scores(self,
                    predictions,
                    sample_weight,
                    test_splits,
                    scorers):
        """Get scores for all predictions and splits.
        
        Parameters
        ----------
        predictions : np.ndarray
            Array of predictions.
        sample_weight : np.ndarray
            Sample weights.
        test_splits : list
            List of test splits.
        scorers : list
            List of scorers.
            
        Returns
        -------
        dict
            Dictionary mapping scorers to score arrays.
        """

        response, original_y = self.data
                                      
        scores_ = []
        grouped_weights_ = [sample_weight[split].sum() for split in test_splits]

        final_scores = {}
        for cur_scorer in scorers:
            scores = []
            for i in np.arange(predictions.shape[1]):
                cur_scores = []
                for f, split in enumerate(test_splits):
                    try:
                        if cur_scorer.use_full_data:
                            y = original_y
                        else:
                            y = response
                        val, w = cur_scorer.score_fn(split,
                                                     y,
                                                     predictions[:,i],
                                                     sample_weight=sample_weight)
                    except ValueError as e:
                        warnings.warn(f'Scorer "{cur_scorer.name}" failed on fold {f}, lambda {i}: {e}')
                        # record a missing score for this fold (excluded when averaging)
                        # rather than reusing the previous fold's value
                        if cur_scorer.grouped:
                            val, w = np.nan, sample_weight[split].sum()
                        else:
                            val, w = np.full(len(split), np.nan), sample_weight[split]
                    cur_scores.append([val, w])

                if cur_scorer.grouped:
                    cur_scores = np.array(cur_scores)
                else:
                    cur_scores = np.hstack(cur_scores).T
                scores.append(cur_scores)
            scores = np.array(scores) # preds x splits x 2 (the 2 is for value, weight_sum)
            final_scores[cur_scorer] = scores

        return final_scores
    
def plot(cv_scores,
         index_best,
         index_1se,
         index=None,
         score=None,
         ax=None,
         capsize=3,
         legend=False,
         col_min='#909090',
         ls_min='--',
         col_1se='#909090',
         ls_1se='--',
         c='#c0c0c0',
         scatter_c='red',
         scatter_s=None,
         **plot_args):
    """Plot cross-validation scores.
    
    Parameters
    ----------
    cv_scores : pd.DataFrame
        Cross-validation scores.
    index_best : pd.Series
        Best lambda indices for each scorer.
    index_1se : pd.Series
        One standard error lambda indices for each scorer.
    index : pd.Series, optional
        Lambda values.
    score : str, optional
        Score to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    capsize : int, default=3
        Cap size for error bars.
    legend : bool, default=False
        Whether to show legend.
    col_min : str, default='#909090'
        Color for minimum line.
    ls_min : str, default='--'
        Line style for minimum line.
    col_1se : str, default='#909090'
        Color for one standard error line.
    ls_1se : str, default='--'
        Line style for one standard error line.
    c : str, default='#c0c0c0'
        Color for main line.
    scatter_c : str, default='red'
        Color for scatter points.
    scatter_s : int, optional
        Size of scatter points.
    **plot_args
        Additional plotting arguments.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """

    if score not in index_best:
        raise ValueError(f'Score "{score}" was not computed in the CV fit.')

    score_path = pd.DataFrame({score: cv_scores[score]}).reset_index()
    score_path[index.name] = index

    ax = score_path.plot.scatter(y=score,
                                 c=scatter_c,
                                 s=scatter_s,
                                 x=index.name,
                                 ax=ax,
                                 zorder=3,
                                 **plot_args)

    if f'SD({score})' in cv_scores.columns:
        have_std = True
        # use values as the index of cv_scores might be different
        score_path[f'SD({score})'] = cv_scores[f'SD({score})'].values
        score_path = score_path.set_index(index.name)
        ax = score_path.plot(y=score,
                             kind='line',
                             yerr=f'SD({score})',
                             capsize=capsize,
                             legend=legend,
                             c=c,
                             ax=ax,
                             **plot_args)
    else:
        score_path = score_path.set_index(index.name)
        ax = score_path.plot(y=score,
                             kind='line',
                             legend=legend,
                             c=c,
                             ax=ax,
                             **plot_args)

    
    ax.set_ylabel(score)
    
    # add the lines indicating best and 1se choice

    if index_best is not None:
        if score in index_best.index:
            _best_idx = list(cv_scores.index).index(index_best[score])
            ax.axvline(index[_best_idx],
                       c=col_min,
                       ls=ls_min,
                       label=r'Best')
        
    if index_1se is not None:
        if score in index_1se.index:
            _1se_idx = list(cv_scores.index).index(index_1se[score])
            ax.axvline(index[_1se_idx],
                       c=col_min,
                       ls=ls_min,
                       label=r'1SE')
        
    if legend:
        ax.legend()
    return ax

def _tune(index,
          scorers,
          cv_scores,
          complexity_order=None,
          compute_std_error=True):
    """Tune lambda selection based on cross-validation scores.
    
    Parameters
    ----------
    index : pd.Series
        Lambda values.
    scorers : list
        List of scorers.
    cv_scores : pd.DataFrame
        Cross-validation scores.
    complexity_order : str, optional
        Order of complexity ('increasing' or 'decreasing').
    compute_std_error : bool, default=True
        Whether standard errors were computed.
        
    Returns
    -------
    tuple
        Tuple of (index_best, index_1se).
    """

    if (complexity_order is not None
        and complexity_order not in ['increasing', 'decreasing']):
        raise ValueError("complexity order must be in ['increasing', 'decreasing']")

    index_best_ = []
    index_1se_ = []

    npath = cv_scores.shape[0]
    
    for i, scorer in enumerate(scorers):
        picker = {False:np.argmin, True:np.argmax}[scorer.maximize]

        if complexity_order == 'increasing':
            _mean = np.asarray(cv_scores[scorer.name])
        else:
            # in this case indices are taken from the last entry
            # must reshuffle indices below
            _mean = np.asarray(cv_scores[scorer.name].iloc[::-1])

        _best_idx = picker(_mean)
        index_best_.append((scorer.name, index[_best_idx]))

        if compute_std_error:
            _std = np.asarray(cv_scores[f'SD({scorer.name})'])
            if not scorer.maximize:
                _mean_1se = (_mean + _std)[_best_idx]

                if complexity_order is not None:
                    _1se_idx = max(np.nonzero((_mean <=
                                               _mean_1se))[0].min() - 1, 0)
                else:
                    _1se_idx = None

            else:
                _mean_1se = (_mean - _std)[_best_idx]                    
                if complexity_order is not None:
                    _1se_idx = max(np.nonzero((_mean >=
                                               _mean_1se))[0].min() - 1, 0)
                else:
                    _1se_idx = None

            if _1se_idx is not None:
                index_1se_.append((scorer.name, index[_1se_idx]))
            else:
                index_1se_.append((scorer.name, np.nan))

    index_best_ = pd.Series([v for _, v in index_best_],
                            index=[n for n, _ in index_best_],
                            name='index_best')

    if complexity_order == 'decreasing':
        index_best_ = npath - 1 - index_best_

    if len(index_1se_) > 0:
        index_1se_ = pd.Series([v for _, v in index_1se_],
                                index=[n for n, _ in index_1se_],
                               name='index_1se')
        if complexity_order == 'decreasing':
            index_1se_ = npath - 1 - index_1se_
    else:
        index_1se_ = None
        
    return index_best_, index_1se_

@dataclass
class ScorePath(object):
    """
    Container for cross-validation results along the regularization path.

    This class stores the results of cross-validation performed by GLMNet models.
    It provides access to cross-validated scores, standard errors, best/1se indices,
    lambda values, and other relevant information for model selection and diagnostics.

    Attributes
    ----------
    scores : pd.DataFrame
        DataFrame of cross-validated scores for each metric and lambda value.
        For example, scores['Mean Squared Error'] gives the mean CV MSE for each lambda.
    index_best : pd.Series
        Indices of the best lambda value for each score metric (e.g., minimum error).
    index_1se : pd.Series
        Indices of the lambda value within one standard error of the best for each metric.
    lambda_values : np.ndarray
        Array of lambda values used in the regularization path.
    norm : np.ndarray
        L1 norm (or other norm) of the coefficients at each lambda value.
    fracdev : np.ndarray
        Fraction of deviance explained at each lambda value.
    family : GLMFamilySpec
        The GLM family specification used for fitting.
    score : str or None
        The primary score metric (optional, used for plotting).

    Methods
    -------
    plot(...):
        Plot the cross-validation results for a given score metric.
    """

    scores: pd.DataFrame
    index_best: pd.Series
    index_1se: pd.Series
    lambda_values: np.ndarray
    norm: np.ndarray
    fracdev: np.ndarray
    family: GLMFamilySpec
    score: str | None = None

    def plot(self,
             score=None,
             xvar='-lambda',
             ax=None,
             capsize=3,
             legend=False,
             col_min='#909090',
             ls_min='--',
             col_1se='#909090',
             ls_1se='--',
             c='#c0c0c0',
             scatter_c='red',
             scatter_s=None,
             **plot_args):

        if xvar == 'lambda':
            self.index = pd.Series(np.log(self.lambda_values), name=r'$\log(\lambda)$')
        elif xvar == '-lambda':
            self.index = pd.Series(-np.log(self.lambda_values), name=r'$-\log(\lambda)$')
        elif xvar == 'norm':
            self.index = pd.Index(self.norm)
            self.index.name = r'$\|\beta(\lambda)\|$'
        elif xvar == 'dev':
            self.index = pd.Index(self.fracdev)
            self.index.name = 'Fraction Deviance Explained'
        else:
            raise ValueError("xvar should be in ['lambda', '-lambda', 'norm', 'dev']")

        if score is None:
            score = self.family._default_scorers()[0].name

        score = score or self.score
        return plot(self.scores,
                    self.index_best,
                    self.index_1se,
                    score=score,
                    index=self.index,
                    ax=ax,
                    capsize=capsize,
                    legend=legend,
                    col_min=col_min,
                    ls_min=ls_min,
                    col_1se=col_1se,
                    ls_1se=ls_1se,
                    c=c,
                    scatter_c=scatter_c,
                    scatter_s=scatter_s,
                    **plot_args)
