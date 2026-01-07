import torch


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implement the PPO clipped surrogate objective.

    Args:
        old_log_probs (bs, output_len): Log probs from behavior policy
        new_log_probs (bs, output_len): Log probs from current policy
        advantages (bs, output_len): Advantage estimates
        clip_ratio: Clip parameter (ε in the paper)

    Returns:
        loss: PPO loss to minimize
        clipfrac: Fraction of samples that got clipped
    """
    # TODO: Implement this
    # Hint:
    # 1. Compute ratio = exp(new_log_probs - old_log_probs)
    # 2. Compute two losses: ratio * advantages and clipped_ratio * advantages
    # 3. Take max (worst case) of the two
    # 4. Average and negate (we minimize in PyTorch)

    # log(pi_new) - log(pi_old)
    # unclipped_loss = A * ratio; clipped_loss = A * clip(r, 1-e, 1+ e)
    # (bs, output_len)
    ratio = (new_log_probs - old_log_probs).exp()
    # (bs, output_len)
    unclipped_loss = advantages * ratio
    clipped_loss = advantages * torch.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
    loss = -torch.min(clipped_loss, unclipped_loss).mean()
    clip_fraction = (torch.abs(ratio - 1) > clip_ratio).float().mean()
    return loss, clip_fraction


def value_loss(
    predicted_values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_value: bool = True,
    clip_range: float = 0.5,
):
    """
    Implement PPO's value function loss.

    Args:
        predicted_values: V(s) from current value network
        returns: Target returns (from GAE)
        old_values: V(s) from old value network (if clipping)
        clip_value: Whether to clip value function updates
        clip_range: Clip range for value function

    Returns:
        loss: Value function loss
    """
    # TODO: Implement this
    # With clipping: max((V_new - G)^2, (V_clipped - G)^2)
    # Without clipping: (V_new - G)^2
    if clip_value:
        assert old_values is not None, (
            "`old_values` cannot be none when using `clip_values`"
        )
        V_clipped = old_values + torch.clip(
            predicted_values - old_values, -clip_range, clip_range
        )
        loss = torch.max(
            (predicted_values - returns) ** 2, (V_clipped - returns) ** 2
        ).mean()
    else:
        loss = ((predicted_values - returns) ** 2).mean()
    return loss


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits.

    Args:
        logits: Shape (batch, vocab_size) or (batch, seq_len, vocab_size)

    Returns:
        entropy: Shape (batch,) or (batch, seq_len)
    """
    # TODO: Use log_softmax for stability
    return -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(
        dim=-1
    )


def entropy_loss(logits: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked average entropy.

    Args:
        logits: Shape (batch, seq_len, vocab_size)
        eos_mask: Shape (batch, seq_len)

    Returns:
        entropy: Scalar (averaged over valid tokens)
    """
    # TODO: Compute entropy at each position, then masked average
    entropy = entropy_from_logits(logits)
    masked_entropy = entropy * eos_mask
    return masked_entropy.sum() / eos_mask.sum()
