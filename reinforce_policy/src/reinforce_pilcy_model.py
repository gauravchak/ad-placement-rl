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
        user_features: [B, D]
        action: [B] (0 or 1)
        reward: [B]

        Returns:
            loss: a scalar tensor representing the REINFORCE objective
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
