# Your task: Fill in the TODOs
import torch.nn as nn


class TinyPolicy(nn.Module):
    """A 1-layer network that outputs probability of 'heads'"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, 2)])
        pass

    def forward(self, x):
        # TODO: Return probability distribution over [heads, tails]
        pass


def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns from a list of rewards.

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor

    Returns:
        returns: List of discounted returns [G_0, G_1, ..., G_T]
        where G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """
    # TODO: Implement this
    # Hint: Work backwards from the end
    pass


def policy_gradient_loss(log_probs, returns):
    """
    The core REINFORCE loss: -log(π(a|s)) * G_t

    Args:
        log_probs: Log probabilities of actions taken
        returns: Discounted returns from compute_returns()

    Returns:
        loss: Scalar loss to minimize
    """
    # TODO: Implement this
    # Remember: PyTorch minimizes, so negate the objective
    pass


# TODO: Write training loop that:
# 1. Samples actions from policy
# 2. Gets rewards (1 for correct, 0 for wrong)
# 3. Computes returns
# 4. Updates policy
