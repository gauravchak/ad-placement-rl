from sklearn.ensemble import RandomForestRegressor


def train_reward_models(X, A, R, weights=None):
    """
    Train separate reward models for each action.
    If weights are provided, they are used as sample weights.
    """
    model_ad = RandomForestRegressor(n_estimators=100, random_state=42)
    model_no_ad = RandomForestRegressor(n_estimators=100, random_state=42)

    if weights is None:
        # No weighting
        model_ad.fit(X[A == 1], R[A == 1])
        model_no_ad.fit(X[A == 0], R[A == 0])
    else:
        # Weighted training
        model_ad.fit(X[A == 1], R[A == 1], sample_weight=weights[A == 1])
        model_no_ad.fit(X[A == 0], R[A == 0], sample_weight=weights[A == 0])

    return model_ad, model_no_ad


def predict_policy(model_ad, model_no_ad, user_features):
    """
    Given trained models and user_features, return a binary policy decision:
    show_ad = 1 if predicted_reward_if_ad > predicted_reward_if_no_ad else 0
    """
    pred_if_ad, pred_if_no_ad = expected_reward(
        model_ad=model_ad,
        model_no_ad=model_no_ad,
        user_features=user_features)
    return (pred_if_ad > pred_if_no_ad).astype(int)


def expected_reward(model_ad, model_no_ad, user_features):
    """
    Compute the expected reward difference for analysis or evaluation.
    """
    pred_if_ad = model_ad.predict(user_features)
    pred_if_no_ad = model_no_ad.predict(user_features)
    return pred_if_ad, pred_if_no_ad
