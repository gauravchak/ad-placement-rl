# REINFORCE Policy Approach

This directory demonstrates using a simple reinforcement learning (RL) method (REINFORCE) to learn a policy for deciding whether to show an ad.

**Setup:**

- **Data**: Similar to the contextual bandit setting, we have logged sessions with user features (`x`), a binary action (`a`), and a scalar reward (`r`).
- **Goal**: Learn a stochastic policy \(\pi_\theta(a|x)\) that maximizes expected reward.

**REINFORCE Algorithm:**
Instead of fitting separate models for each action, REINFORCE optimizes parameters \(\theta\) directly to maximize expected reward:

- Define \(\pi_\theta(a=1|x) = \sigma(\text{mlp\_action}(x) - \text{mlp\_no\_action}(x))\).
- Given a batch of observations \((x_i, a_i, r_i)\), update \(\theta\) via:
  \[
  \nabla_\theta J(\theta) \propto \sum_i r_i \nabla_\theta \log \pi_\theta(a_i|x_i).
  \]
  
This uses the log probability of the chosen action, weighted by the observed reward.

**Pros & Cons:**

- **Pros**:  
  - Directly optimizes policy parameters towards higher reward.
  - Naturally produces a stochastic policy, facilitating exploration.
  
- **Cons**:  
  - High variance gradient estimates.
  - Requires careful tuning and may be less sample-efficient than regression-based approaches.
  - Sensitive to data distribution shifts (it’s an on-policy method, but here we’re applying it to logged data).

**Future Directions:**

- Incorporate variance-reduction techniques (e.g., baselines) to stabilize training.
- Explore off-policy corrections if the logged data policy differs from the learned policy.
- Compare performance to contextual bandit approaches and doubly robust estimators.
