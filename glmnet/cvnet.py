from dataclasses import dataclass
from typing import Optional, Literal, Any

import numpy as np

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.utils.validation import check_is_fitted


@dataclass
class CVNet(BaseEstimator):
    """
    A path estimator tuned by cross-validation, as sklearn's `ElasticNetCV`
    or R's `cv.glmnet`: `fit` fits the path and runs its
    `cross_validation_path`, and `predict` predicts at the chosen lambda
    (and gamma, for a relaxed fit). As an sklearn estimator it can be
    cross-validated itself, e.g. by `sklearn.model_selection.cross_validate`
    (nested cross-validation).

    Parameters
    ----------
    estimator: GLMNet
        The path estimator (e.g. `GaussNet()`, `LogNet()`); it is cloned.
    cv: int, cross-validation generator or an iterable
        As for `cross_validation_path`. Default 10.
    which: str
        '1se' (the default, R's ``lambda.1se``) or 'best' (``lambda.min``).
    score: str, optional
        The score the choice is based on; see `GLMNet.cv_choice`.
    prediction_type: str, optional
        Passed to the estimator's `predict` (its default if None).

    Attributes
    ----------
    estimator_: GLMNet
        The fitted path, with its `score_path_`.
    lambda_: float
        The chosen lambda.
    gamma_: float
        The chosen gamma (1 unless the path is relaxed).
    """
    estimator: Any = None
    cv: Any = 10
    which: Literal['1se', 'best'] = '1se'
    score: Optional[str] = None
    prediction_type: Optional[str] = None

    def fit(self, X, y, **fit_params):
        """
        Fit the path to `(X, y)`, cross-validate it, and choose lambda.

        Returns
        -------
        self: object
            CVNet class instance.
        """
        if self.estimator is None:
            raise ValueError('CVNet needs a path estimator, e.g. CVNet(GaussNet())')
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        self.estimator_.cross_validation_path(X, y, cv=self.cv)
        self.lambda_, self.gamma_ = self.estimator_.cv_choice(which=self.which,
                                                              score=self.score)
        if hasattr(self.estimator_, 'classes_'):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X, **predict_args):
        """
        Predictions of the path at the chosen lambda (and gamma).
        """
        check_is_fitted(self, ['estimator_'])
        if self.prediction_type is not None:
            predict_args.setdefault('prediction_type', self.prediction_type)
        return self.estimator_.cv_predict(X,
                                          which=self.which,
                                          score=self.score,
                                          **predict_args)

    @property
    def coef_(self):
        """Coefficients at the chosen lambda."""
        coefs, _ = self.estimator_.cv_coefs(which=self.which, score=self.score)
        return coefs

    @property
    def intercept_(self):
        """Intercept at the chosen lambda."""
        _, intercepts = self.estimator_.cv_coefs(which=self.which, score=self.score)
        return intercepts

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if self.estimator is not None:
            estimator_tags = self.estimator.__sklearn_tags__()
            tags.estimator_type = estimator_tags.estimator_type
        return tags
