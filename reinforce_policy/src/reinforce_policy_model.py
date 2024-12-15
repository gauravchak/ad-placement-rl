"""
This file implements a simple policy gradient approach (REINFORCE) to learn a binary decision policy using logged data.

In this single-step contextual bandit setting, we have:
- user_features: Contextual information about the user/session.
- action: A binary action taken historically (0 or 1).
- reward: A numerical reward observed after taking the action.

The model defines a parameterized stochastic policy π_θ(a|x), where:
    π_θ(a=1|x) = sigmoid(mlp_action(x) - mlp_no_action(x))

To train, we use the REINFORCE update rule: (Sutton & Barto, Chapter 13)
    ∇_θ J(θ) ∝ E[r * ∇_θ log π_θ(a|x)]

This code:
- Computes the log probability of the observed action.
- Multiplies it by the observed reward.
- Takes the negative mean (since we minimize loss = -J).

Note:
- REINFORCE is on-policy. Applying it directly to logged, offline data may
    result in biased estimates if the logging policy differs substantially
    from the current policy.
- This simple version may have high variance. Practical implementations often
    add a baseline or other variance-reduction techniques.
"""

import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    """
    A policy model that outputs the probability of taking action=1 given user features.
    This model defines a stochastic policy π(a=1|x) = sigmoid(score_action - score_no_action),
    where score_action and score_no_action are learned from the user features.

    Usage:
    ------
    At inference time, you can use the forward pass to get a probability for each instance:
        prob_action = model.forward(user_features)  # shape [B]

    To select an action stochastically (to allow for exploration), you can sample from a Bernoulli distribution:
        sampled_action = torch.bernoulli(prob_action)  # shape [B], values in {0,1}

    For example:
        prob_action = model(user_features)
        action = (torch.rand_like(prob_action) < prob_action).float()
    This will take action=1 with probability equal to prob_action and action=0 otherwise.

    At training time, you can use `train_forward` to compute the REINFORCE loss given (user_features, action, reward).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Simple MLPs for scoring each action
        # Adjust architecture as needed
        self.mlp_action = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.mlp_no_action = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_features: torch.Tensor) -> torch.Tensor:
        """
        Given user features [B, D], return the probability of taking action=1.

        Parameters:
            user_features: A [B, D] tensor representing the batch of user contexts.

        Returns:
            prob_action: A [B] tensor of probabilities, one per user, representing
                         π(a=1|x). Values are in (0,1).
        """
        a = self.mlp_action(user_features).squeeze(-1)      # [B]
        na = self.mlp_no_action(user_features).squeeze(-1)  # [B]
        prob_action = torch.sigmoid(a - na)                 # [B]
        return prob_action

    def train_forward(
            self,
            user_features: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the REINFORCE-style loss given a batch of transitions.

        Parameters:
            user_features: [B, D] tensor of user contexts.
            action: [B] tensor of actions taken (0 or 1).
            reward: [B] tensor of observed rewards.

        Returns:
            loss: A scalar tensor representing the REINFORCE objective.
                  The policy gradient update is derived from E[reward * log π(a|x)].
                  Here we minimize the negative objective (loss).
        """
        prob_action = self.forward(user_features)  # [B]

        # Compute log_prob of the chosen action
        log_prob_action = (
            action * torch.log(prob_action.clamp(min=1e-8)) +
            (1 - action) * torch.log((1 - prob_action).clamp(min=1e-8))
        )  # [B]

        # REINFORCE loss: - E[reward * log_prob(chosen_action)]
        loss = -(log_prob_action * reward).mean()  # [B]

        return loss
