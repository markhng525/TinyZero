from collections import defaultdict

import torch


def grpo_advantages(
    outcome_rewards: torch.Tensor, prompt_indices: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Implement GRPO's advantage computation.

    Key idea: Sample multiple responses per prompt, then normalize
    advantages within each prompt group.

    Args:
        outcome_rewards: (batch,) - scalar reward per sequence
        prompt_indices: (batch,) - which prompt each sequence came from

    Returns:
        advantages: (batch,) - normalized advantages

    Algorithm:
    1. Group sequences by prompt_index
    2. For each group, compute mean and std of rewards
    3. Normalize: adv[i] = (reward[i] - mean[group]) / (std[group] + eps)
    """
    batch_size = prompt_indices.shape[0]
    index = defaultdict(list[int])
    advantages = torch.zeros_like(outcome_rewards)
    for i in range(batch_size):
        index[prompt_indices[i].item()].append(i)

    for _, group_index in index.items():
        mean = outcome_rewards[group_index].mean() if len(group_index) > 1 else 0
        std = outcome_rewards[group_index].std() if len(group_index) > 1 else 1
        advantages[group_index] = (outcome_rewards[group_index] - mean) / (std + eps)

    return advantages


# **Questions**:
# - Why sample multiple responses per prompt?
# - How does this avoid length bias?
# - What if we only have 1 response per prompt? (Check TinyZero line 143-145)
