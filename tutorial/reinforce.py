# Your task: Fill in the TODOs

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sympy.physics.control.control_plots import matplotlib


class TinyPolicy(nn.Module):
    """A 1-layer network that outputs probability of 'heads'"""

    def __init__(self, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, 2, bias)])

    def forward(self, x: torch.Tensor):
        # TODO: Return probability distribution over [heads, tails]
        for layer in self.layers:
            x = layer(x)
        return x.softmax(dim=-1)


def compute_returns(rewards: list[float], gamma: float = 0.99):
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

    # rtg stands for reward-to-go
    # Scratch work:
    # From G_{T} = r_T
    # G_{T-1} = r_{T-1} + gamma * G_T
    # G_{T-2} = r_{T-2} + gamma * G_{T-1}

    rtg: list[float] = [0] * len(rewards)
    for t in reversed(range(len(rewards))):
        if t == (len(rewards) - 1):
            rtg[t] = rewards[t]
        else:
            rtg[t] = rewards[t] + gamma * rtg[t + 1]
    return rtg


def policy_gradient_loss(
    log_probs: torch.Tensor, returns: torch.Tensor
) -> torch.Tensor:
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
    # logprobs: (B, S, 2)
    # returns: (B, S)
    loss = -(log_probs * returns).mean()
    return loss


@dataclass
class TrackingMetrics:
    returns: list[float]
    policy_loss: list[float]
    grad_norm: list[float]
    entropy: list[float]


@dataclass
class RunMetrics:
    return_cum_mean: list[float]
    return_cum_var: list[float]
    loss_cum_mean: list[float]
    loss_cum_var: list[float]
    grad_norm_cum_mean: list[float]
    grad_norm_cum_var: list[float]
    entropy_cum_mean: list[float]
    entropy_cum_var: list[float]


def calculate_final_metrics(
    metrics: TrackingMetrics, window_size: int = 100
) -> RunMetrics:
    def rolling_mean(x: list[float], window: int = window_size) -> np.ndarray:
        """Compute rolling mean with specified window size."""
        x_array = np.array(x)
        # Use cumsum trick for efficient rolling mean
        cumsum = np.cumsum(np.insert(x_array, 0, 0))
        # For indices < window, use expanding mean; otherwise rolling
        result = np.zeros(len(x_array))
        for i in range(len(x_array)):
            if i < window:
                result[i] = cumsum[i + 1] / (i + 1)
            else:
                result[i] = (cumsum[i + 1] - cumsum[i + 1 - window]) / window
        return result

    def rolling_var(x: list[float], window: int = window_size) -> np.ndarray:
        """Compute rolling variance with specified window size."""
        x_array = np.array(x)
        result = np.zeros(len(x_array))
        for i in range(len(x_array)):
            if i < 1:
                result[i] = 0.0
            else:
                win_start = max(0, i + 1 - window)
                window_data = x_array[win_start : i + 1]
                result[i] = np.var(window_data)
        return result

    return RunMetrics(
        return_cum_var=rolling_var(metrics.returns).tolist(),
        return_cum_mean=rolling_mean(metrics.returns).tolist(),
        loss_cum_var=rolling_var(metrics.policy_loss).tolist(),
        loss_cum_mean=rolling_mean(metrics.policy_loss).tolist(),
        grad_norm_cum_var=rolling_var(metrics.grad_norm).tolist(),
        grad_norm_cum_mean=rolling_mean(metrics.grad_norm).tolist(),
        entropy_cum_var=rolling_var(metrics.entropy).tolist(),
        entropy_cum_mean=rolling_mean(metrics.entropy).tolist(),
    )


def train_simple(
    x: torch.Tensor,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    seed: int = 24,
    use_baseline: bool = True,
) -> TrackingMetrics:
    torch.manual_seed(seed)
    # biased coin w/ 0.7 heads prob
    p_heads = 0.7
    env = torch.distributions.Categorical(probs=torch.tensor([1 - p_heads, p_heads]))
    returns, policy_losses, policy_entropy, grad_norms = [], [], [], []
    baseline = 0.0

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        action_probs: torch.Tensor = policy(x)
        action = torch.distributions.Categorical(probs=action_probs).sample()
        state = env.sample()
        logger.debug(f"Action: {action}", f"State: {state}")
        reward = 1 if action.item() == state.item() else -1
        logger.debug(f"Policy probs: {action_probs}")
        advantage = reward - baseline
        if use_baseline:
            baseline = baseline + (1 / (epoch + 1)) * (reward - baseline)
        log_probs = action_probs.squeeze(0)[action].log()
        loss = policy_gradient_loss(
            log_probs=log_probs,
            returns=torch.tensor(advantage),
        )
        logger.debug(f"Loss: {loss}")
        loss.backward()
        optimizer.step()
        # Update tracking metrics
        returns.append(reward)
        policy_losses.append(loss.item())
        # Entropy: H(π) = -Σ π(a) log π(a) over all actions
        entropy = -(action_probs * action_probs.log()).sum()
        policy_entropy.append(entropy.item())
        grads = [
            param.grad.detach().flatten()
            for param in policy.parameters()
            if param.grad is not None
        ]
        grad_norms.append(torch.cat(grads).norm().item())
    return TrackingMetrics(
        returns=returns,
        policy_loss=policy_losses,
        grad_norm=grad_norms,
        entropy=policy_entropy,
    )


if __name__ == "__main__":
    hidden_dim = 4
    dummy_x = torch.ones(1, hidden_dim)
    policy = TinyPolicy(hidden_dim)
    optimizer = torch.optim.SGD(params=policy.parameters(), lr=1e-2)
    # TODO: Track variance of losses, gradient norm
    metrics = train_simple(x=dummy_x, policy=policy, optimizer=optimizer, n_epochs=1000)

    # Plot and save returns
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(metrics.returns)
    matplotlib.pyplot.title("Returns over time")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Return")
    matplotlib.pyplot.savefig("returns_plot.png")

    # Plot and save policy loss
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(metrics.policy_loss)
    matplotlib.pyplot.title("Policy Loss over time")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.savefig("policy_loss_plot.png")

    # Plot and save gradient norms
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(metrics.grad_norm)
    matplotlib.pyplot.title("Gradient Norm over time")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Gradient Norm")
    matplotlib.pyplot.savefig("grad_norm_plot.png")

    # Plot and save entropy
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(metrics.entropy)
    matplotlib.pyplot.title("Policy Entropy over time")
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Entropy")
    matplotlib.pyplot.savefig("entropy_plot.png")

    logger.info(
        "Plots saved to: returns_plot.png, policy_loss_plot.png, grad_norm_plot.png, entropy_plot.png"
    )
    # TODO: Write training loop that:
    # 1. Samples actions from policy
    # 2. Gets rewards (1 for correct, 0 for wrong)
    # 3. Computes returns
    # 4. Updates policy
