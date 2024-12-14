# Contextual Bandits Approach

This directory demonstrates learning a policy for deciding whether to show an ad at the start of a user session using a contextual bandit approach.

**Setup:**

- **Data**: Each logged session includes user features (`x`), a binary action (`a`=1 if an ad was shown, 0 otherwise), and a resulting reward (`r`).
- **Goal**: Learn a policy that, given `x`, chooses `a` to maximize expected reward.

**File structure**
data_utils.py: How data is prepared.
propensity.py: How propensity scores are computed.
reward_models.py: How reward predictions and policies are formed.
train_policy.py: How all pieces fit together to train a policy.
eval_policy.py: How to evaluate policies.

**Direct Method:**

1. Train two separate regressors:
   - \(\hat{r}_{\text{ad}}(x) = E[r|x,a=1]\)
   - \(\hat{r}_{\text{no\_ad}}(x) = E[r|x,a=0]\)
   
   Each is trained using only the data from sessions where that action was taken.

2. For a new user, predict both \(\hat{r}_{\text{ad}}(x)\) and \(\hat{r}_{\text{no\_ad}}(x)\). Choose the action that yields the higher predicted reward.

**Inverse Propensity Weighting (IPW):**
Historical actions may not be uniformly distributed. To reduce bias:

- Estimate the propensity \( p(a=1|x) \) via a logistic regression.
- Assign weights:
  \[
  w_i =
  \begin{cases}
  \frac{1}{p(a=1|x_i)} & \text{if } a_i=1 \\[6pt]
  \frac{1}{1 - p(a=1|x_i)} & \text{if } a_i=0
  \end{cases}
  \]

Use these weights when training the reward models.

**Pros & Cons:**

- **Pros**: Simple, interpretable, leverages standard regression tools.
- **Cons**: Relies on accurate weighting and reward models. Potentially high variance if some actions are rare.

**Future Directions:**

- **Doubly Robust Methods**: Combine reward modeling and IPW estimators more tightly to reduce bias and variance.
- **Contextual Bandit Algorithms**: Implement methods that adaptively learn policies with exploration.
- **Comparison with RL Methods**: Contrast this approach with policy-gradient method REINFORCE in `../reinforce_policy`
