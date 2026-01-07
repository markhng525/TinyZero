from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from torch.optim import Optimizer

from tutorial.ppo import entropy_from_logits, ppo_loss


class ActorCritic(nn.Module):
    """Shared backbone with policy (actor) and value (critic) heads."""

    def __init__(
        self,
        vocab_size: int = 10,
        hidden_dim: int = 64,
        n_layers: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared backbone
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias) for _ in range(n_layers)]
        )

        # Two heads
        self.lm_head = nn.Linear(hidden_dim, vocab_size)  # Policy head
        self.value_head = nn.Linear(hidden_dim, 1)  # Value head

    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Returns hidden states before the heads."""
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x  # Shape: (bs, seq_len, hidden_dim)

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for policy."""
        hidden = self.get_hidden_states(x)
        return self.lm_head(hidden)  # (bs, seq_len, vocab_size)

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns value estimates."""
        hidden = self.get_hidden_states(x)
        return self.value_head(hidden).squeeze(-1)  # (bs, seq_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns both policy logits and values."""
        hidden = self.get_hidden_states(x)
        logits = self.lm_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def reward_function(sequences: torch.Tensor) -> torch.Tensor:
    """
    Simple reward: +1 if sequence sums to 10, else 0
    sequences: (batch, seq_len) of integers
    """
    # Note: broadcast the rewards signal across the entire sequence
    return (sequences.sum(dim=-1) == 10).unsqueeze(-1).expand(-1, sequences.shape[-1])


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.1, lam: float = 0.95
) -> torch.Tensor:
    """
    Args:
    rewards (bs, output_len)
    values (bs, output_len)
    """
    output_len = rewards.shape[-1]
    # gae_{t} = delta_{t} + lam * gamma * gae_{t+1}
    # delta_t = reward_t + gamma * value_{t+1} - value_{t}
    # Terminal condition value_{T} = 0
    # => delta_{T - 1} = r_{T - 1} - value_{T - 1}
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(output_len)):
        value_next = values[:, t + 1] if t < output_len - 1 else 0.0
        delta = rewards[:, t] + gamma * value_next - values[:, t]
        gae = delta + lam * gamma * gae
        advantages[:, t] = gae

    return advantages


@dataclass
class TrackingMetrics:
    policy_loss: list[float]
    value_loss: list[float]
    kl_penalty: list[float]
    clip_frac: list[float]


def train_mini_ppo(
    sample: torch.Tensor,
    actor_critic: ActorCritic,
    reference: ActorCritic,
    optimizer: Optimizer,
    training_epochs: int = 5,
    ppo_epochs: int = 10,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> TrackingMetrics:
    metrics = TrackingMetrics(
        policy_loss=[], value_loss=[], kl_penalty=[], clip_frac=[]
    )

    for epoch in range(training_epochs):
        # ============== ROLLOUT PHASE ==============
        with torch.no_grad():
            logits: torch.Tensor
            values_old: torch.Tensor
            logits, values_old = actor_critic(sample)
            logprobs_all = logits.log_softmax(dim=-1)

            # Sample sequences from policy
            sequences = torch.distributions.Categorical(logits=logits).sample()

            # Get rewards and compute advantages
            rewards = reward_function(sequences=sequences).float()
            advantages = compute_gae(rewards=rewards, values=values_old)
            returns = (values_old + advantages).detach()

            # Cache old log probs for the actions taken
            old_log_probs = logprobs_all.gather(
                dim=-1, index=sequences.unsqueeze(-1)
            ).squeeze(-1)

        # Detach advantages for PPO epochs
        advantages = advantages.detach()

        # ============== PPO UPDATE PHASE ==============
        for ppo_epoch in range(ppo_epochs):
            optimizer.zero_grad()

            # Forward pass
            logits_new, values_new = actor_critic(sample)
            logprobs_new = (
                logits_new.log_softmax(dim=-1)
                .gather(dim=-1, index=sequences.unsqueeze(-1))
                .squeeze(-1)
            )

            # Policy loss (PPO clipped)
            policy_loss, clipfrac = ppo_loss(
                old_log_probs=old_log_probs,
                new_log_probs=logprobs_new,
                advantages=advantages,
            )

            # Value function loss
            value_loss = nn.functional.mse_loss(values_new, returns)

            # KL penalty (for tracking only)
            with torch.inference_mode():
                logits_ref = reference.forward_policy(sample)
                logprobs_ref = (
                    logits_ref.log_softmax(dim=-1)
                    .gather(dim=-1, index=sequences.unsqueeze(-1))
                    .squeeze(-1)
                )
                kl: torch.Tensor = (logprobs_new - logprobs_ref).mean()

            # Entropy bonus (for exploration)
            entropy = entropy_from_logits(logits_new).mean()

            # Total loss
            total_loss = (
                policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            )

            total_loss.backward()
            optimizer.step()

            # Track metrics
            metrics.policy_loss.append(policy_loss.item())
            metrics.value_loss.append(value_loss.item())
            metrics.kl_penalty.append(kl.item())
            metrics.clip_frac.append(clipfrac.item())

        # Log progress
        if epoch % 1 == 0:
            reward_rate = rewards[:, 0].mean().item()  # All positions have same reward
            print(
                f"Epoch {epoch}: reward_rate={reward_rate:.3f}, "
                f"policy_loss={policy_loss.item():.4f}, "
                f"value_loss={value_loss.item():.4f}, "
                f"clipfrac={clipfrac.item():.3f}, "
                f"entropy={entropy.item():.3f}"
            )

    return metrics


if __name__ == "__main__":
    # Hyperparameters
    seq_len = 5
    batch_size = 32
    vocab_size = 10
    hidden_dim = 64
    lr = 1e-3
    training_epochs = 100
    ppo_epochs = 4

    # Device selection (industry standard pattern)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input: position indices (same for all batches)
    sample = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Create actor-critic and frozen reference
    actor_critic = ActorCritic(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    reference = ActorCritic(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    reference.load_state_dict(actor_critic.state_dict())  # Copy weights
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False  # Freeze reference

    # Single optimizer for shared backbone
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    # Train!
    print("Starting PPO training...")
    print(f"Goal: Generate sequences of {seq_len} digits (0-9) that sum to 10")
    print("-" * 60)

    metrics = train_mini_ppo(
        sample=sample,
        actor_critic=actor_critic,
        reference=reference,
        optimizer=optimizer,
        training_epochs=training_epochs,
        ppo_epochs=ppo_epochs,
    )

    # Test final policy
    print("-" * 60)
    print("Testing final policy:")
    with torch.inference_mode():
        logits = actor_critic.forward_policy(sample[:5])
        probs = logits.softmax(dim=-1)
        sequences = torch.distributions.Categorical(logits=logits).sample()
        rewards = reward_function(sequences)

        for i in range(5):
            seq = sequences[i].tolist()
            total = sum(seq)
            success = "✓" if total == 10 else "✗"
            print(f"  Sequence: {seq}, Sum: {total} {success}")

    success_rate = rewards[:, 0].float().mean().item()
    print(f"\nFinal success rate: {success_rate:.1%}")

# **Success criteria**:
# - Policy learns to generate sequences summing to ~10
# - clipfrac is reasonable (not 0, not 1)
# - Entropy decreases over time but doesn't collapse to 0
# - You understand every line of code
