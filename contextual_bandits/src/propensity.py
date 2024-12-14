import numpy as np
from sklearn.linear_model import LogisticRegression


def train_propensity_model(X, A):
    """
    Train a logistic regression model to estimate p(action=1 | features).
    """
    prop_model = LogisticRegression(max_iter=1000)
    prop_model.fit(X, A)
    return prop_model


def compute_weights(prop_model, X, A):
    """
    Compute inverse propensity weights.
    weight = 1/p for action=1
    weight = 1/(1-p) for action=0
    """
    prop = prop_model.predict_proba(X)[:, 1]
    weights = np.where(A == 1, 1.0 / prop, 1.0 / (1.0 - prop))
    return weights
