from reward_models import expected_reward


def evaluate_policy(model_ad, model_no_ad, X, A, R):
    """
    Evaluate how well the learned policy would perform on a hold-out set.
    This evaluation looks at predicted rewards vs. actual outcomes.
    """
    pred_if_ad, pred_if_no_ad = expected_reward(model_ad, model_no_ad, X)

    # Compare predicted actions to actual actions and rewards
    chosen_actions = (pred_if_ad > pred_if_no_ad).astype(int)
    actual_average_reward = R.mean()

    # If we can assume that the chosen_actions match historical actions (naive), 
    # we just print out the average reward. Real OPE is more complex.
    print("Average actual reward in dataset:", actual_average_reward)
    print(
        "Policy would choose 'ad' action for:", chosen_actions.mean(),
        "fraction of users."
    )
    # Add more thorough evaluation as needed.
