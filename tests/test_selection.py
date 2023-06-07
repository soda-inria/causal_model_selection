import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from causalmodelselection.selection import get_treatment_and_covariates
from causalmodelselection.selection import CausalSelection


def test_get_treatment_and_covariates(dummy_factuals):
    a, X_cov, _ = dummy_factuals
    XX_bad = np.column_stack((X_cov, a))
    expected_warn_msg = "First column of covariates should contains the treatment indicator as binary values."
    with pytest.raises(ValueError, match=expected_warn_msg):
        get_treatment_and_covariates(XX_bad)
    XX_good = np.column_stack((a, X_cov))
    a_retrieved, X_cov_retrieved = get_treatment_and_covariates(XX_good)
    assert np.array_equal(a_retrieved, a)
    assert np.array_equal(X_cov, X_cov_retrieved)


def test_causal_selection_fit(dummy_factuals):
    a_estimator = LogisticRegression()
    m_estimator = Ridge()

    causal_selector = CausalSelection(
        a_estimator=a_estimator, m_estimator=m_estimator, cv=2, test_ratio=0.33
    )
    candidate_g_estimators = [
        Ridge(alpha=10, random_state=42),
        Ridge(alpha=1, random_state=42),
    ]
    (a, X, y) = dummy_factuals
    a_X = np.insert(X, 0, np.array(a), axis=1)
    causal_selector.fit(a_X, y, candidate_g_estimators=candidate_g_estimators)

    expected_r_risks = np.array([0.22283, 0.218055])
    np.testing.assert_array_almost_equal(
        np.array(causal_selector.r_risks_), expected_r_risks, decimal=5
    )
