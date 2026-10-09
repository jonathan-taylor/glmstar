from dataclasses import dataclass
from typing import Optional

import numpy as np

from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             roc_auc_score,
                             average_precision_score)

@dataclass(frozen=True)
class Scorer(object):

    name: str
    score: callable=None
    maximize: bool=True
    use_full_data: bool=False
    grouped: bool=True
    normalize_weights: bool=True

    def score_fn(self,
                 split,
                 response,
                 predictions,
                 sample_weight):

        W = np.asarray(sample_weight)[split]
        W_sum = W.sum()
        if self.normalize_weights:
            W = W / W.mean()
        return (self.score(response[split],
                           predictions[split],
                           sample_weight=W), W_sum)

mse_scorer = Scorer(name='Mean Squared Error',
                    score=mean_squared_error,
                    maximize=False)
mae_scorer = Scorer(name='Mean Absolute Error',
                    score=mean_absolute_error,
                    maximize=False)

def _accuracy_score(y, yhat, sample_weight): # for binary data classifying at p=0.5, eta=0
    # y may be a proportion: the fraction y of the observation is a success,
    # as in R's cv.lognet(type.measure="class")
    y = np.asarray(y, float)
    correct = y * (yhat > 0.5) + (1 - y) * (yhat <= 0.5)
    return np.average(correct, weights=sample_weight)

def _expand_binary(y, yhat, sample_weight):
    """Expand proportions y into a failure row (weight w*(1-y)) and a success row
    (weight w*y) per observation, as in R's auc.mat. For 0/1 y this only adds
    rows with zero weight."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float)
    Y = np.concatenate([np.zeros_like(y), np.ones_like(y)])
    return Y, np.concatenate([yhat, yhat]), np.concatenate([w * (1 - y), w * y])

def _auc_score(y, yhat, sample_weight):
    Y, P, W = _expand_binary(y, yhat, sample_weight)
    return roc_auc_score(Y, P, sample_weight=W)

def _aucpr_score(y, yhat, sample_weight):
    Y, P, W = _expand_binary(y, yhat, sample_weight)
    return average_precision_score(Y, P, sample_weight=W)

accuracy_scorer = Scorer(name='Accuracy',
                         score=_accuracy_score,
                         maximize=True)
auc_scorer = Scorer('AUC',
                    score=_auc_score,
                    maximize=True)
aucpr_scorer = Scorer('AUC-PR',
                      score=_aucpr_score,
                      maximize=True)

# R's cv.lognet and cv.multnet clamp predicted probabilities to
# [PROB_MIN, 1 - PROB_MIN] before computing the deviance
PROB_MIN = 1e-5

def _clamped_name(name, prob_min):
    """Name of a deviance scorer, distinguishing non-default clamping."""
    if prob_min == PROB_MIN:
        return name
    if not prob_min:
        return f'{name} (Unclamped)'
    return f'{name} (prob_min={prob_min:g})'

def _xlogy_ratio(y, p):
    """y * log(y / p), with 0 * log(0) = 0."""
    y = np.asarray(y, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(y > 0, y * (np.log(np.where(y > 0, y, 1)) - np.log(p)), 0.)

@dataclass(frozen=True)
class BinomialDevianceScorer(Scorer):
    """Binomial deviance, with predicted probabilities clamped to
    [prob_min, 1 - prob_min] as in R's `cv.glmnet`.

    Use `binomial_deviance_scorer(prob_min=None)` for the unclamped deviance.
    """

    name: str = 'Binomial Deviance'
    maximize: bool = False
    prob_min: Optional[float] = PROB_MIN

    def score_fn(self,
                 split,
                 response,
                 predictions,
                 sample_weight):

        W = np.asarray(sample_weight)[split]
        y = np.asarray(response)[split]
        p = np.asarray(predictions)[split]
        if self.prob_min:
            p = np.clip(p, self.prob_min, 1 - self.prob_min)
        dev = 2 * (_xlogy_ratio(y, p) + _xlogy_ratio(1 - y, 1 - p))
        return np.average(dev, weights=W), W.sum()

def binomial_deviance_scorer(prob_min=PROB_MIN):
    """Binomial deviance scorer.

    Parameters
    ----------
    prob_min : float or None, default=1e-5
        Predicted probabilities are clamped to [prob_min, 1 - prob_min] before
        computing the deviance, as in R's `cv.glmnet`. None or 0 disables
        clamping. Non-default values give the scorer a distinct name, so it can
        be used alongside the default scorer.
    """
    return BinomialDevianceScorer(name=_clamped_name('Binomial Deviance', prob_min),
                                  prob_min=prob_min)

@dataclass(frozen=True)
class MultinomialDevianceScorer(Scorer):
    """Multinomial deviance, with predicted class probabilities clamped to
    [prob_min, 1 - prob_min] (without renormalizing) as in R's `cv.glmnet`.

    Use `multinomial_deviance_scorer(prob_min=None)` for the unclamped deviance.
    """

    name: str = 'Multinomial Deviance'
    maximize: bool = False
    prob_min: Optional[float] = PROB_MIN

    def score_fn(self,
                 split,
                 response,
                 predictions,
                 sample_weight):

        W = np.asarray(sample_weight)[split]
        y = np.asarray(response)[split]       # one-hot (or proportions), (n, K)
        p = np.asarray(predictions)[split]    # class probabilities, (n, K)
        if self.prob_min:
            p = np.clip(p, self.prob_min, 1 - self.prob_min)
        dev = 2 * _xlogy_ratio(y, p).sum(1)
        return np.average(dev, weights=W), W.sum()

def multinomial_deviance_scorer(prob_min=PROB_MIN):
    """Multinomial deviance scorer.

    Parameters
    ----------
    prob_min : float or None, default=1e-5
        Predicted probabilities are clamped to [prob_min, 1 - prob_min] before
        computing the deviance, as in R's `cv.glmnet`. None or 0 disables
        clamping. Non-default values give the scorer a distinct name, so it can
        be used alongside the default scorer.
    """
    return MultinomialDevianceScorer(name=_clamped_name('Multinomial Deviance', prob_min),
                                     prob_min=prob_min)

class UngroupedScorer(Scorer):

    grouped: bool=False
    score: callable=None

    def score_fn(self,
                 split,
                 response,
                 predictions,
                 sample_weight):
        return self.score(response[split], predictions[split]), sample_weight[split]

ungrouped_mse_scorer = UngroupedScorer(name="Mean Squared Error (Ungrouped)",
                                       score=lambda y, pred: (y-pred)**2,
                                       maximize=False,
                                       grouped=False)

ungrouped_mae_scorer = UngroupedScorer(name="Mean Absolute Error (Ungrouped)",
                                       score=lambda y, pred: np.fabs(y-pred),
                                       maximize=False,
                                       grouped=False)
