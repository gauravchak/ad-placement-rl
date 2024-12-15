"""Main training code
"""

from data_utils import load_data, prepare_data
from propensity import train_propensity_model, compute_weights
from reward_models import train_reward_models, should_show_ad


def main():
    # Configuration
    file_path = "data/user_sessions.csv"  # User should update with your data path
    user_features_cols = ["feature1", "feature2", "feature3"]  # User shpould update with actual features

    # Load and prepare data
    df = load_data(file_path)
    X_train, A_train, R_train, X_val, A_val, R_val = prepare_data(
        df, user_features_cols
    )

    # Train propensity model
    prop_model = train_propensity_model(X_train, A_train)
    weights = compute_weights(prop_model, X_train, A_train)

    # Train reward models with and without weighting
    model_ad_unweighted, model_no_ad_unweighted = train_reward_models(
        X_train, A_train, R_train
    )
    model_ad_weighted, model_no_ad_weighted = train_reward_models(
        X_train, A_train, R_train, weights=weights
    )

    # Predict on validation set with unweighted policy
    policy_unweighted = should_show_ad(
        model_ad_unweighted, model_no_ad_unweighted, X_val
    )

    # Predict on validation set with weighted policy
    policy_weighted = should_show_ad(
        model_ad_weighted, model_no_ad_weighted, X_val
    )

    # Just print out some summary info (in real scenario, we'd evaluate properly)
    print("Unweighted Policy decisions (sample):", policy_unweighted[:10])
    print("Weighted Policy decisions (sample):", policy_weighted[:10])


if __name__ == "__main__":
    main()
