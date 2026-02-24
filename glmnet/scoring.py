from dataclasses import dataclass

import numpy as np

from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             accuracy_score,
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
    return accuracy_score(y,
                          yhat>0.5,
                          sample_weight=sample_weight,
                          normalize=True)

accuracy_scorer = Scorer(name='Accuracy',
                         score=_accuracy_score,
                         maximize=True)
auc_scorer = Scorer('AUC',
                    score=roc_auc_score,
                    maximize=True)
aucpr_scorer = Scorer('AUC-PR',
                      score=average_precision_score,
                      maximize=True)

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
