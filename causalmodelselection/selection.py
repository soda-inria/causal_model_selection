from typing import Dict, Generator, List, Union
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.calibration import column_or_1d
from sklearn.model_selection import (
    BaseCrossValidator,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)


class CausalSelection(BaseEstimator):
    """Causal selection procedure for candidate g-estimators.

    CausalSelection implements a causal model selection procedure that scores
    candidate g-estimators based on their R-risk with a cross-validation
    procedure adapted to nuisances estimation.

    The covariates X should have the binary intervention/treatment as the first
    column of the nd.array.

    For details on this causal selection procedure see *Doutreligne, M., &
    Varoquaux, G. (2023). How to select predictive models for causal inference?.
    arXiv e-prints, arXiv-2302.*

    Args:
        a_estimator (BaseEstimator): Nuisance estimator for the propensity score
        :math:`\check e(x) = \mathbb P[A=1|X=x]`.

        m_estimator (BaseEstimator): Nuisance estimator for the mean outcome
        model :math:`\check m(x) = \mathbb E[Y|X=x]`.

        a_param_distributions (Dict, optional): Grid of parameters for
        propensity score hyper-parameter search. Defaults to None.

        m_param_distributions (Dict, optional): Grid of parameters for mean
        outcome model hyper-parameter search. Defaults to None.

        a_scoring (str, optional): Strategy to evaluate the performance of the
        propensity score hyper-parameter search. Defaults to "neg_brier_score".

        m_scoring (str, optional): Strategy to evaluate the performance of the
        mean outcome model hyper-parameter search. Defaults to
        "neg_mean_squared_error".

        n_iter (int, optional): Number of parameter settings that are sampled
        for nuisance estimator hyper-parameter search. n_iter trades off runtime
        vs quality of the solution.. Defaults to 10.

        cv (int, optional): cross-validation generator or an iterable,
        default=None Determines the cross-validation splitting strategy.
        Defaults to None.

        random_state (int, optional): Pseudo random number generator state.
        Defaults to 0.

        test_ratio (float, optional): Test ratio for the test set on which to
        evaluate the candidate estimators. Defaults to 0.1.

        n_jobs (int, optional): Number of jobs to run in parallel. ``None``
        means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors. Defaults to -1.

        strict_overlap (_type_, optional): Strict overlap parameter for
        propensity score clipping when evaluating R-risk. Defaults to 1e-10.

    Attributes:
        best_candidate_ (BaseEstimator): G-estimator with the smallest R-risk.

        best_score_ (float): Smaller R-risk corresponding to the best candidate
        g-estimator.

        fitted_candidates_ (List[BaseEstimator]): Candidate g-estimators fitted
        on the train set.

        r_risks_ (List[float]): R-risks of the candidate g-estimators computed
        on the test.

        a_model_rs_ (RandomizedSearchCV): Hyper-parameter search sklearn class
        for the propensity score model.

        m_model_rs_ (RandomizedSearchCV): Hyper-parameter search sklearn class
        for the mean outcome model.

        a_model_rs_results_ (RandomizedSearchCV): Fitted propensity score model
        hyper-parameter search.

        m_model_rs_results_ (RandomizedSearchCV): Fitted mean outcome model
        hyper-parameter search.

        a_nuisance_estimators_ (List[BaseEstimator]): Propensity score
        models fitted by cross-validation on the train set.

        m_nuisance_estimators_ (List[BaseEstimator]): Mean outcome models
        fitted by cross-validation on the train set.
    """

    def __init__(
        self,
        a_estimator: BaseEstimator,
        m_estimator: BaseEstimator,
        a_param_distributions: Dict = None,
        m_param_distributions: Dict = None,
        a_scoring: str = "neg_brier_score",
        m_scoring: str = "neg_mean_squared_error",
        n_iter: int = 10,
        cv: Union[int, BaseCrossValidator, Generator] = None,
        random_state=0,
        test_ratio=0.1,
        n_jobs=-1,
        strict_overlap=1e-10,
    ) -> None:
        self.a_param_distributions = a_param_distributions
        self.m_param_distributions = m_param_distributions
        self.a_estimator = a_estimator
        self.m_estimator = m_estimator
        self.random_state = random_state
        if cv == 1:
            self.cv = dummy1Fold()
        elif isinstance(cv, int):
            self.cv = StratifiedKFold(
                n_splits=cv, random_state=self.random_state, shuffle=True
            )
        elif cv is None:
            self.cv = StratifiedKFold(
                n_splits=5, random_state=self.random_state, shuffle=True
            )
        else:
            self.cv = cv
        self.n_jobs = n_jobs
        self.strict_overlap = strict_overlap
        self.m_scoring = m_scoring
        self.a_scoring = a_scoring
        self.n_iter = n_iter
        self.test_ratio = test_ratio

    def fit(self, X, y, candidate_g_estimators: List[BaseEstimator]):
        """Fit the causal model selection procedure.

        Args:
            X (_type_): _description_

            y (_type_): _description_

            candidate_g_estimators (List[BaseEstimator]): Candidate g-estimators
            to be evaluated.

        Returns:
            _type_: _description_
        """

        _check_intervention_first_column(X)

        self.fitted_candidates_ = []
        self.r_risks_ = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_ratio, random_state=self.random_state
        )
        self._fit_nuisances(X_train, y_train)

        for g_estimator in candidate_g_estimators:
            g_estimator_ = clone(g_estimator)
            # train estimator on the train data
            g_estimator_.fit(X_train, y_train)
            self.fitted_candidates_.append(g_estimator_)
            # Evaluate candidates on the test data
            g_estimator_r_risk = self._evaluate_candidate(
                X_test, y_test, g_estimator_
            )
            self.r_risks_.append(g_estimator_r_risk)

        self.best_score_ = np.min(self.r_risks_)
        self.best_candidate_ = self.fitted_candidates_[np.argmin(self.r_risks_)]
        return self

    def _fit_nuisances(self, X, y):
        """Learn unknown nuisance :math:`(\\check e$, $\\check m)` necessary for model
        selection with R-risk.

        Args:
            X ([type]): [description]
            Y ([type]): [description]

        Returns:
            [type]: [description]
        """
        a, X_cov = get_treatment_and_covariates(X)

        # Find appropriate parameters for nuisance models
        if self.m_param_distributions is not None:
            self.m_model_rs_ = RandomizedSearchCV(
                estimator=self.m_estimator,
                param_distributions=self.m_param_distributions,
                scoring=self.m_scoring,
                n_iter=self.n_iter,
                random_state=self.random_state,
                cv=None,
            )
            self.m_model_rs_results_ = self.m_model_rs_.fit(X_cov, y)
            m_best_estimator = clone(self.m_model_rs_results_.best_estimator_)
        else:
            m_best_estimator = clone(self.m_estimator)
        if self.a_param_distributions is not None:
            self.a_model_rs_ = RandomizedSearchCV(
                estimator=self.a_estimator,
                param_distributions=self.a_param_distributions,
                scoring=self.a_scoring,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                cv=None,
            )
            self.a_model_rs_results_ = self.a_model_rs_.fit(X_cov, a)
            a_best_estimator = clone(self.a_model_rs_results_.best_estimator_)
        else:
            a_best_estimator = clone(self.a_estimator)
        # Refit best nuisance estimators with CV
        splitter_m = self.cv.split(X_cov, a)
        self.m_nuisance_estimators_cv_ = cross_validate(
            m_best_estimator,
            X_cov,
            y,
            cv=splitter_m,
            return_estimator=True,
            scoring="neg_mean_squared_error",
        )

        splitter_a = self.cv.split(X_cov, a)
        self.a_nuisance_estimators_cv_ = cross_validate(
            a_best_estimator,
            X_cov,
            a,
            cv=splitter_a,
            return_estimator=True,
            scoring="neg_brier_score",
        )
        self.a_nuisance_estimators_ = self.a_nuisance_estimators_cv_[
            "estimator"
        ]
        self.m_nuisance_estimators_ = self.m_nuisance_estimators_cv_[
            "estimator"
        ]
        return self

    def _evaluate_candidate(self, X, y, g_estimator: BaseEstimator) -> float:
        """Evaluate candidate estimator with R-risk."""
        check_is_fitted(g_estimator)

        a, X_cov = get_treatment_and_covariates(X)
        # TODO: Should use bagging instead of useless CV predictions on new data.
        hat_e = cross_val_predict_from_fitted(
            estimators=self.a_nuisance_estimators_,
            X=X_cov,
            A=None,
            cv=None,
            method="predict_proba",
        )[:, 1]
        if self.strict_overlap is not None:
            hat_e[hat_e <= 0.5] = hat_e[hat_e <= 0.5] + self.strict_overlap
            hat_e[hat_e > 0.5] = hat_e[hat_e > 0.5] - self.strict_overlap
        hat_m = cross_val_predict_from_fitted(
            estimators=self.m_nuisance_estimators_,
            X=X_cov,
            A=None,
            method="predict",
            cv=self.cv,
        )
        hat_y_0 = g_estimator.predict(
            np.column_stack((np.zeros(X.shape[0]) * 0.0, X_cov))
        )
        hat_y_1 = g_estimator.predict(
            np.column_stack((np.ones(X.shape[0]) * 1.0, X_cov))
        )
        hat_tau = hat_y_1 - hat_y_0
        r_risk_ = r_risk(y, a, hat_m, hat_e, hat_tau)
        return r_risk_


## External util functions ##
def cross_val_predict_from_fitted(
    estimators: List[BaseEstimator],
    X: np.array,
    A=None,
    cv=None,
    method: str = "predict",
) -> np.array:
    """Compute cross-validation predictions from a list of fitted estimators.
    If no treatment A is provided, then the predictions of each estimator are
    bagged by mean-average.

    Args:
        estimators (List): List of fitted estimators.

        X (np.array): Predictors

        splitter ([type], optional): splitter, should yield the same number of
        splits as the number of estimator. Defaults to None.

        method (str, optional): estimator method for prediction : ["predict",
        "predict_proba"]. Defaults to "predict".

    Returns:
        hat_Y (np.array): predictions

    """
    hat_Y = []
    indices = []
    if A is None:
        iterator = dummy1Fold().split(X)
    elif hasattr(cv, "split"):
        iterator = cv.split(X, A)
    for i, (train_ix, test_ix) in enumerate(iterator):
        estimator = estimators[i]
        func = getattr(estimator, method)
        hat_Y.append(func(X[test_ix]))
        indices.append(test_ix)

    if A is None:
        # Average in case of no CV (ie. leftout)
        hat_Y = np.mean(hat_Y, axis=0)
    else:
        indices = np.argsort(np.concatenate(indices, axis=0))
        hat_Y = np.concatenate(hat_Y, axis=0)[indices]
    return hat_Y


def get_treatment_and_covariates(X):
    """Split treatment and covariates from full covariate matrix $X=[a, X_cov]$.

    Require that the first column of $X$ is the treatment indicator.

    Args:
        X (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    a = X[:, 0]
    X_cov = X[:, 1:]
    a = column_or_1d(a, warn=True)
    if not np.array_equal(a, a.astype(bool)):
        raise ValueError(
            "First column of covariates should contains the treatment indicator as binary values."
        )
    return a, X_cov


def _check_intervention_first_column(X):
    """Check that the first column of $X$ is the treatment indicator.

    Args:
        X (_type_): _description_
    """
    a = X[:, 0]
    a = column_or_1d(a, warn=True)
    if not np.array_equal(a, 1.0 * a.astype(bool)):
        raise ValueError(
            "First column of covariates should contains the treatment indicator as binary values."
        )


class dummy1Fold:
    """Dummy splitter for no CV."""

    def __init__(self) -> None:
        pass

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        yield indices, indices


def r_risk(y, a, hat_m, hat_e, hat_tau):
    return np.mean(((y - hat_m) - (a - hat_e) * hat_tau) ** 2)
