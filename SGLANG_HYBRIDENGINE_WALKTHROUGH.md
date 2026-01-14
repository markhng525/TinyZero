# Building a HybridEngine for RLHF: A Complete Technical Walkthrough

**Purpose**: A comprehensive educational guide for implementing an inference engine integration into an RLHF training framework. This document captures lessons learned from attempting SGLang integration into veRL on DGX Spark (GB10), structured as a lesson plan for building deep technical understanding.

**Target Audience**: ML engineers familiar with HuggingFace Transformers APIs who want to build production-quality RLHF infrastructure from scratch.

**How to Use This Document**: Each section poses motivating questions, provides mathematical foundations, shows the connection between math and code, discusses design trade-offs, and provides references for deeper study. Work through sequentially, implementing as you go.

---

## Table of Contents

1. [Conceptual Foundation: What Are We Building?](#1-conceptual-foundation-what-are-we-building)
2. [The Mathematics of Policy Gradient RL](#2-the-mathematics-of-policy-gradient-rl)
3. [Architecture Deep Dive: How Training Frameworks Work](#3-architecture-deep-dive-how-training-frameworks-work)
4. [The HybridEngine Pattern](#4-the-hybridengine-pattern)
5. [Weight Synchronization: The Critical Interface](#5-weight-synchronization-the-critical-interface)
6. [Memory Management: The Hidden Complexity](#6-memory-management-the-hidden-complexity)
7. [The SGLang Challenge: A Case Study in Systems Integration](#7-the-sglang-challenge-a-case-study-in-systems-integration)
8. [Implementation Reference: Code Patterns That Work](#8-implementation-reference-code-patterns-that-work)
9. [Lessons Learned and Debugging Strategies](#9-lessons-learned-and-debugging-strategies)
10. [Building From Scratch: A Roadmap](#10-building-from-scratch-a-roadmap)

---

# 1. Conceptual Foundation: What Are We Building?

## Motivating Questions

Before diving into implementation, ask yourself:

- **Why can't I just call `model.generate()` in my training loop?** (You can, but it's 10-100x slower than optimized inference engines)
- **Why do I need two different systems for the same model?** (Training and inference have fundamentally different computational patterns)
- **What makes RLHF different from supervised fine-tuning?** (The training signal comes from generated outputs, not static labels)

## 1.1 The RLHF Training Objective

At its core, RLHF optimizes a language model to maximize expected reward while staying close to a reference policy. The objective function is:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ R(x, y) - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref}) \right]$$

Where:
- $\pi_\theta(y|x)$ is the policy (language model) we're training
- $\pi_{ref}(y|x)$ is the frozen reference policy (usually the SFT checkpoint)
- $R(x, y)$ is the reward for response $y$ given prompt $x$
- $\beta$ is the KL penalty coefficient (typically 0.01-0.1)
- $D_{KL}$ is the KL divergence preventing the policy from diverging too far

**Design Decision Point**: The KL penalty $\beta$ controls exploration vs. stability. Too high, and the model barely changes. Too low, and training becomes unstable (reward hacking). This is a hyperparameter you'll tune.

> **Reference**: The foundational RLHF paper is [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155). For the KL penalty formulation, see [Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)](https://arxiv.org/abs/1909.08593).

## 1.2 GRPO: A Critic-Free Alternative

GRPO (Group Relative Policy Optimization) simplifies RLHF by eliminating the critic/value network. Instead of estimating advantages with a learned value function, GRPO uses **relative rewards within a group**:

$$A_i = \frac{R_i - \text{mean}(R_{1:n})}{\text{std}(R_{1:n}) + \epsilon}$$

For each prompt, generate $n$ responses and compute advantages relative to the group. This is the approach used in DeepSeek-R1 and the TinyZero project.

**Pseudocode**:
```
for each prompt x:
    # Generate n responses from current policy
    responses = [sample(π_θ, x) for _ in range(n)]

    # Score all responses
    rewards = [reward_fn(x, y) for y in responses]

    # Compute relative advantages (no critic needed!)
    advantages = (rewards - mean(rewards)) / (std(rewards) + ε)

    # Policy gradient update
    for y, A in zip(responses, advantages):
        log_prob = π_θ.log_prob(y | x)
        loss -= A * log_prob  # Gradient ascent on advantage-weighted log probs
```

**Why this matters for systems design**: GRPO requires generating multiple responses per prompt (typically n=4 to 8). This amplifies the importance of fast generation—if generation takes 10 seconds and you need 8 responses per prompt, that's 80 seconds just for one training sample!

> **Reference**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) introduces GRPO. [TinyZero](https://github.com/Jiayi-Pan/TinyZero) provides a minimal implementation.

## 1.3 The Four Phases of an RLHF Training Step

Every RLHF training step consists of four distinct computational phases, each with different requirements:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RLHF TRAINING STEP PHASES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: GENERATION (Autoregressive Sampling)                              │
│  ─────────────────────────────────────────────                              │
│  Computational Pattern: Sequential token generation, memory-bound           │
│  Key Resource: KV-cache for efficient attention                             │
│  Optimization: Continuous batching, CUDA graphs, speculative decoding       │
│  Typical Time: 60-80% of total step time                                    │
│                                                                             │
│  Mathematical Operation:                                                     │
│    for t = 1 to max_length:                                                 │
│      logits_t = model(y_{<t}, x)  # Forward pass for ONE token              │
│      y_t ~ Categorical(softmax(logits_t / temperature))                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 2: LOG PROBABILITY COMPUTATION (Actor)                               │
│  ────────────────────────────────────────────                               │
│  Computational Pattern: Single forward pass over full sequence              │
│  Key Resource: Activation memory for all positions                          │
│  Optimization: Chunked softmax, fused kernels                               │
│                                                                             │
│  Mathematical Operation:                                                     │
│    logits = model(y, x)           # Shape: [batch, seq_len, vocab_size]     │
│    log_probs = log_softmax(logits, dim=-1)                                  │
│    token_log_probs = gather(log_probs, y)  # Shape: [batch, seq_len]        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 3: REFERENCE LOG PROBABILITIES                                       │
│  ─────────────────────────────────────                                      │
│  Computational Pattern: Same as Phase 2, but with frozen model              │
│  Key Insight: No gradients needed, can use inference optimizations          │
│  Memory Note: Reference model weights must be in memory                     │
│                                                                             │
│  Mathematical Operation:                                                     │
│    with torch.no_grad():                                                    │
│      ref_log_probs = ref_model.log_prob(y | x)                              │
│    kl_div = actor_log_probs - ref_log_probs  # Per-token KL                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 4: POLICY GRADIENT UPDATE                                            │
│  ───────────────────────────────                                            │
│  Computational Pattern: Forward + Backward + Optimizer step                 │
│  Key Resources: Gradients, optimizer states (2x model size for AdamW)       │
│  Optimization: Gradient checkpointing, mixed precision, FSDP                │
│                                                                             │
│  Mathematical Operation (PPO-style clipped objective):                      │
│    ratio = exp(log_prob_new - log_prob_old)                                 │
│    clipped = clip(ratio, 1-ε, 1+ε)                                          │
│    loss = -min(ratio * A, clipped * A) + β * KL                             │
│    loss.backward()                                                           │
│    optimizer.step()                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Phase 1 (generation) is fundamentally different from Phases 2-4. Generation is **memory-bandwidth bound** (reading weights repeatedly for each token), while training is **compute-bound** (matrix multiplications). This is why we need different systems.

## 1.4 Why Two Systems? The Training vs. Inference Dichotomy

**Training Frameworks (PyTorch, FSDP, DeepSpeed)** are optimized for:

| Concern | How It's Addressed |
|---------|-------------------|
| Gradient computation | Autograd graph construction |
| Memory efficiency | Activation checkpointing, gradient accumulation |
| Distributed training | Parameter sharding (FSDP), pipeline parallelism |
| Numerical stability | Loss scaling, gradient clipping |

**Inference Engines (vLLM, SGLang, TensorRT-LLM)** are optimized for:

| Concern | How It's Addressed |
|---------|-------------------|
| Token generation speed | KV-cache reuse, continuous batching |
| Throughput | CUDA graphs, kernel fusion |
| Memory efficiency | PagedAttention, memory pools |
| Latency | Speculative decoding, chunked prefill |

**Concrete Example**: Consider generating 1024 tokens with a 7B model.

*Naive PyTorch (model.generate())*:
- Each token requires a full forward pass: ~15ms per token
- Total time: 1024 × 15ms = **15.4 seconds**
- No KV-cache reuse between calls in some implementations

*vLLM/SGLang*:
- Efficient KV-cache management: ~2ms per token after prefill
- CUDA graph capture: reduces kernel launch overhead
- Continuous batching: process multiple sequences simultaneously
- Total time: **~2-3 seconds** for the same generation

This 5-10x speedup is why inference engines exist. When training requires generating thousands of sequences per epoch, this difference becomes hours vs. days.

> **Reference**: [Efficiently Scaling Transformer Inference (Pope et al., 2022)](https://arxiv.org/abs/2211.05102) explains the memory-bandwidth bound nature of autoregressive generation. [vLLM: Easy, Fast, and Cheap LLM Serving (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) introduces PagedAttention.

## 1.5 The Naive Approach and Why It Fails

Before understanding the solution, let's deeply understand the problem:

```python
# THE NAIVE APPROACH - DO NOT USE
class NaiveRLHFTrainer:
    def __init__(self, model_path):
        # Load model for training
        self.actor = AutoModelForCausalLM.from_pretrained(model_path)  # 3GB
        self.actor = FSDP(self.actor)  # Wrap for distributed training
        self.optimizer = AdamW(self.actor.parameters())  # +12GB optimizer states

        # Load SAME model again for fast generation
        self.generator = vllm.LLM(model_path)  # Another 3GB!

        # Load SAME model again for reference
        self.reference = AutoModelForCausalLM.from_pretrained(model_path)  # Another 3GB!

        # Total model weights: 9GB (3x duplication!)
        # Plus optimizer: 12GB
        # Plus KV-cache for generation: 20-40GB
        # Total before any activations: 41-61GB minimum

    def train_step(self, prompts):
        # Generate responses using vLLM (fast!)
        responses = self.generator.generate(prompts)  # Uses original weights!

        # PROBLEM 1: self.generator has ORIGINAL weights, not trained weights!
        # As training progresses, generation quality doesn't improve!

        # Compute rewards
        rewards = self.reward_fn(prompts, responses)

        # Compute log probs for policy gradient
        actor_log_probs = self.compute_log_probs(self.actor, responses)
        ref_log_probs = self.compute_log_probs(self.reference, responses)

        # PROBLEM 2: At this point we have:
        # - Actor weights (3GB) + gradients (3GB) + optimizer (12GB)
        # - Generator weights (3GB) + KV-cache (20-40GB)
        # - Reference weights (3GB)
        # Total: 44-64GB just for weights + optimizer!
        # No room for activations on a 40GB or even 80GB GPU!

        # Update
        loss = self.compute_loss(actor_log_probs, ref_log_probs, rewards)
        loss.backward()
        self.optimizer.step()
```

**Three Critical Failures**:

1. **Memory Explosion**: Three copies of model weights plus optimizer states exceeds most GPU memory
2. **Stale Weights**: Inference engine never sees updated weights, so generated text quality never improves
3. **Resource Contention**: No coordination between systems competing for the same GPU

---

# 2. The Mathematics of Policy Gradient RL

## Motivating Questions

- **How do we take gradients through sampling?** (The REINFORCE trick / score function estimator)
- **Why does the KL penalty appear in the objective?** (Constrained optimization as regularization)
- **What makes PPO more stable than vanilla policy gradient?** (Clipped objectives bound the update size)

## 2.1 The Policy Gradient Theorem

The fundamental result that makes RL training possible. Given a parameterized policy $\pi_\theta$, the gradient of expected reward is:

$$\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[R(y)] = \mathbb{E}_{y \sim \pi_\theta}[R(y) \cdot \nabla_\theta \log \pi_\theta(y)]$$

**Why this is remarkable**: We can estimate gradients of the expected reward without differentiating through the reward function or the sampling process. We only need:
1. Samples from the policy (generation)
2. Rewards for those samples (scoring)
3. Gradients of log probabilities (differentiable)

**In code**:
```python
def policy_gradient_loss(log_probs, rewards):
    """
    log_probs: [batch_size, seq_len] - log π_θ(y_t | y_{<t}, x)
    rewards: [batch_size] - R(x, y) for each sequence

    Returns scalar loss for gradient descent (negative for ascent)
    """
    # Sum log probs over tokens to get log π_θ(y | x)
    sequence_log_probs = log_probs.sum(dim=-1)  # [batch_size]

    # Policy gradient: ∇ E[R] ≈ E[R · ∇ log π]
    # For gradient descent, we minimize the negative
    loss = -(rewards * sequence_log_probs).mean()

    return loss
```

> **Reference**: [Policy Gradient Methods for Reinforcement Learning (Sutton et al., 2000)](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) is the foundational paper. [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) provides excellent intuition.

## 2.2 Variance Reduction with Baselines

Vanilla policy gradient has high variance. The key insight: we can subtract any baseline $b$ that doesn't depend on actions without biasing the gradient:

$$\nabla_\theta \mathbb{E}[R] = \mathbb{E}[(R - b) \cdot \nabla_\theta \log \pi_\theta]$$

**Common baselines**:
- **Constant baseline**: $b = \mathbb{E}[R]$ (mean reward)
- **State-dependent baseline**: $b(s) = V(s)$ (value function)
- **GRPO baseline**: $b_i = \text{mean}(R_{1:n})$ within the response group

**GRPO Advantage Computation** (what TinyZero uses):
```python
def compute_grpo_advantages(rewards, n_samples_per_prompt):
    """
    rewards: [batch_size] where batch_size = num_prompts * n_samples_per_prompt

    For GRPO, advantages are relative within each prompt's response group.
    """
    # Reshape to [num_prompts, n_samples_per_prompt]
    rewards = rewards.view(-1, n_samples_per_prompt)

    # Compute mean and std within each group
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)

    # Normalize (z-score within group)
    advantages = (rewards - mean) / (std + 1e-8)

    return advantages.view(-1)  # Flatten back
```

**Design Decision Point**: The number of samples per prompt ($n$) trades off variance reduction vs. computational cost. With $n=1$, you have no baseline (high variance). With $n=8$, you get good variance reduction but 8x generation cost.

| n | Variance | Compute Cost | When to Use |
|---|----------|--------------|-------------|
| 1 | Very high | 1x | Never for GRPO (need comparison) |
| 2 | High | 2x | Fast experiments, high reward signal |
| 4 | Medium | 4x | Good default |
| 8 | Low | 8x | Sparse/noisy rewards |
| 16+ | Very low | 16x+ | Only if generation is cheap |

## 2.3 The KL Divergence Constraint

RLHF adds a KL penalty to prevent the policy from deviating too far from the reference:

$$\mathcal{L} = -\mathbb{E}[R] + \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

For language models, we compute KL per-token:

$$D_{KL}(\pi_\theta || \pi_{ref}) = \sum_{t=1}^{T} \left[ \log \pi_\theta(y_t | y_{<t}) - \log \pi_{ref}(y_t | y_{<t}) \right]$$

**In code**:
```python
def compute_kl_divergence(actor_log_probs, ref_log_probs, mask):
    """
    Compute KL divergence between actor and reference policies.

    actor_log_probs: [batch, seq_len] - log π_θ(y_t | ...)
    ref_log_probs: [batch, seq_len] - log π_ref(y_t | ...)
    mask: [batch, seq_len] - 1 for real tokens, 0 for padding

    Returns per-sequence KL divergence.
    """
    # Per-token KL
    per_token_kl = actor_log_probs - ref_log_probs  # [batch, seq_len]

    # Mask out padding and sum over sequence
    per_token_kl = per_token_kl * mask
    sequence_kl = per_token_kl.sum(dim=-1)  # [batch]

    return sequence_kl
```

**Why KL matters for systems design**: Computing KL requires log probabilities from BOTH the actor AND reference model. This means:
1. We need reference model weights in memory during this computation
2. We need to run forward passes through both models
3. This doubles the memory pressure compared to single-model training

> **Reference**: [Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)](https://arxiv.org/abs/1909.08593) introduces the KL penalty formulation for LLMs.

## 2.4 PPO: Bounded Policy Updates

PPO (Proximal Policy Optimization) clips the policy update to prevent destructively large changes:

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

**Implementation**:
```python
def ppo_loss(new_log_probs, old_log_probs, advantages, clip_epsilon=0.2):
    """
    PPO clipped objective.

    new_log_probs: log probs from CURRENT policy (requires grad)
    old_log_probs: log probs from policy at START of update (detached)
    advantages: computed advantages (detached)
    """
    # Probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # PPO objective: min of clipped and unclipped
    # We want to maximize advantage, so we take min to be conservative
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    # Take min (pessimistic bound) and negate for gradient descent
    loss = -torch.min(surrogate1, surrogate2).mean()

    return loss
```

**Key insight for implementation**: PPO requires log probs from the "old" policy (policy at start of this update iteration) AND the current policy. In practice:
1. Generate responses with policy $\pi_{\theta_{old}}$
2. Store `old_log_probs` (detached, no gradient)
3. For each PPO epoch (typically 1-4):
   - Compute `new_log_probs` with current $\pi_\theta$
   - Compute clipped loss
   - Update $\theta$
   - Repeat

> **Reference**: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

## 2.5 Putting It Together: The Complete GRPO Objective

The loss function used in TinyZero/DeepSeek-style training:

$$\mathcal{L}(\theta) = -\mathbb{E}\left[\sum_{t} \hat{A} \cdot \log \pi_\theta(y_t | y_{<t}, x)\right] + \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

**Complete implementation**:
```python
def grpo_loss(
    actor_log_probs,      # [batch, seq_len] - current policy
    ref_log_probs,        # [batch, seq_len] - reference policy
    rewards,              # [batch] - per-sequence rewards
    response_mask,        # [batch, seq_len] - 1 for response tokens
    n_samples_per_prompt, # int - number of responses per prompt
    kl_coef=0.01,         # β - KL penalty coefficient
):
    """
    Compute GRPO loss with KL penalty.
    """
    # 1. Compute GRPO advantages (relative within group)
    advantages = compute_grpo_advantages(rewards, n_samples_per_prompt)

    # 2. Compute policy gradient loss
    # Sum log probs over response tokens only
    response_log_probs = (actor_log_probs * response_mask).sum(dim=-1)
    pg_loss = -(advantages * response_log_probs).mean()

    # 3. Compute KL divergence
    kl = compute_kl_divergence(actor_log_probs, ref_log_probs, response_mask)
    kl_loss = kl.mean()

    # 4. Total loss
    total_loss = pg_loss + kl_coef * kl_loss

    return total_loss, {
        'pg_loss': pg_loss.item(),
        'kl_loss': kl_loss.item(),
        'kl_mean': kl.mean().item(),
        'advantages_mean': advantages.mean().item(),
        'advantages_std': advantages.std().item(),
    }
```

---

# 3. Architecture Deep Dive: How Training Frameworks Work

## Motivating Questions

- **Why do we need FSDP/DeepSpeed for a 1.5B model that fits on one GPU?** (Optimizer states are 2-4x model size!)
- **How does gradient checkpointing trade compute for memory?** (Recompute activations during backward)
- **What's the difference between data parallelism and model parallelism?** (Split data vs. split model)

## 3.1 Memory Anatomy of a Training Step

Before optimizing, you must understand where memory goes. For a transformer with $P$ parameters:

| Component | Size (bytes) | For 1.5B Model (bf16) | Notes |
|-----------|--------------|----------------------|-------|
| Model Parameters | $2P$ | 3 GB | bf16 = 2 bytes/param |
| Gradients | $2P$ | 3 GB | Same size as parameters |
| Optimizer State (AdamW) | $8P$ | 12 GB | m, v in fp32 = 4+4 bytes |
| **Subtotal (static)** | $12P$ | **18 GB** | Always in memory |
| Activations | $O(B \cdot S \cdot H \cdot L)$ | 10-50 GB | Depends on batch, sequence |
| **Total** | - | **28-68 GB** | Without any optimizations |

**Key insight**: Optimizer states dominate! AdamW keeps two fp32 copies (momentum, variance) per parameter. This is why a 1.5B model that "should" need 3GB actually needs 18GB minimum.

```python
# Memory calculation helper
def estimate_training_memory(
    num_params: int,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    bytes_per_param: int = 2,  # bf16
    gradient_checkpointing: bool = True,
):
    """
    Estimate peak GPU memory for training.
    """
    # Static memory
    params_memory = num_params * bytes_per_param
    gradients_memory = num_params * bytes_per_param
    optimizer_memory = num_params * 8  # AdamW: m and v in fp32

    static_memory = params_memory + gradients_memory + optimizer_memory

    # Activation memory (rough estimate)
    if gradient_checkpointing:
        # Only store activations at layer boundaries
        activation_memory = batch_size * seq_length * hidden_size * num_layers * bytes_per_param * 0.3
    else:
        # Store all activations
        activation_memory = batch_size * seq_length * hidden_size * num_layers * bytes_per_param * 4

    total = static_memory + activation_memory

    return {
        'params_gb': params_memory / 1e9,
        'gradients_gb': gradients_memory / 1e9,
        'optimizer_gb': optimizer_memory / 1e9,
        'activations_gb': activation_memory / 1e9,
        'total_gb': total / 1e9,
    }

# Example: Qwen2.5-1.5B
memory = estimate_training_memory(
    num_params=1.5e9,
    batch_size=4,
    seq_length=1280,  # 256 prompt + 1024 response
    hidden_size=1536,
    num_layers=28,
    gradient_checkpointing=True,
)
print(memory)
# {'params_gb': 3.0, 'gradients_gb': 3.0, 'optimizer_gb': 12.0,
#  'activations_gb': ~9.2, 'total_gb': ~27.2}
```

> **Reference**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054) provides detailed memory analysis.

## 3.2 Gradient Checkpointing: Trading Compute for Memory

**The problem**: During backpropagation, we need activations from the forward pass to compute gradients. Naively, we store all activations, which is $O(L)$ memory for $L$ layers.

**The solution**: Don't store all activations. Instead, store only at "checkpoints" (e.g., every layer boundary) and recompute the rest during backward.

```
Standard Backprop (memory hungry):
Forward:  Save A1 → Save A2 → Save A3 → Save A4 → output
Backward: Use A4   Use A3   Use A2   Use A1
Memory: O(L) activations

Gradient Checkpointing (compute hungry):
Forward:  Save A1 → (discard) → (discard) → Save A4 → output
Backward: Use A4   Recompute A3   Recompute A2   Use A1
Memory: O(√L) activations (with optimal checkpointing)
Extra Compute: ~33% more forward passes
```

**Implementation in practice**:
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask):
        # Checkpoint this layer - activations will be recomputed during backward
        return checkpoint(
            self.layer,
            hidden_states,
            attention_mask,
            use_reentrant=False,  # Recommended for modern PyTorch
        )

# In HuggingFace, just set:
model.gradient_checkpointing_enable()
```

**Design Decision Point**: Gradient checkpointing is almost always worth it for RLHF because:
1. Memory is typically the bottleneck, not compute
2. The 33% compute overhead is dwarfed by generation time
3. Enables larger batch sizes → better gradient estimates

## 3.3 FSDP: Fully Sharded Data Parallelism

FSDP shards model parameters, gradients, and optimizer states across GPUs. Even on a single GPU, it provides useful abstractions.

**Core concept**: Instead of each GPU holding the full model, each GPU holds a shard. Before computation, gather the shards. After computation, release.

```
Traditional Data Parallel (each GPU has full model):
GPU 0: [Full Model] [Full Optimizer] [Gradients] [Activations for batch 0]
GPU 1: [Full Model] [Full Optimizer] [Gradients] [Activations for batch 1]
Memory per GPU: Full model + optimizer + gradients + activations

FSDP (sharded):
GPU 0: [Shard 0] [Opt Shard 0] [Grad Shard 0] [Activations for batch 0]
GPU 1: [Shard 1] [Opt Shard 1] [Grad Shard 1] [Activations for batch 1]
Memory per GPU: (Model + optimizer + gradients) / N + activations
```

**Key FSDP concepts for single-GPU**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# Even on single GPU, FSDP provides:
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # Shard gradients and optimizer
    # Options:
    # - FULL_SHARD: Shard params, grads, optimizer (most memory efficient)
    # - SHARD_GRAD_OP: Shard grads and optimizer only
    # - NO_SHARD: Like regular DDP

    cpu_offload=CPUOffload(offload_params=False),  # Can offload to CPU

    # Mixed precision
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,  # Reduce in fp32 for stability
        buffer_dtype=torch.bfloat16,
    ),
)
```

**Critical for HybridEngine**: FSDP wraps tensors in `DTensor` for sharding. When you call `model.state_dict()`, you get DTensors, not regular tensors. The inference engine doesn't understand DTensors, so you need conversion.

```python
# Getting weights out of FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig

# Option 1: Get sharded state dict (memory efficient)
FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT)
sharded_state = model.state_dict()  # Contains DTensors

# Option 2: Get full state dict (gathers to rank 0)
FSDP.set_state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
)
full_state = model.state_dict()  # Contains regular tensors
# Warning: This gathers ALL parameters to rank 0, causing memory spike!
```

> **Reference**: [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [Introducing PyTorch Fully Sharded Data Parallel](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

## 3.4 veRL's Worker Architecture

veRL uses Ray for distributed orchestration. Understanding the worker pattern is essential:

```python
# Simplified from verl/workers/fsdp_workers.py

class ActorRolloutRefWorker:
    """
    A SINGLE Ray actor that handles ALL roles:
    - Actor (policy being trained)
    - Rollout (generation for data collection)
    - Reference (frozen policy for KL computation)

    Why colocate? Memory sharing! These roles can share:
    - The underlying model weights (with careful synchronization)
    - GPU memory (time-multiplexed between phases)
    """

    def __init__(self, config):
        self.config = config

        # Determine which roles this worker handles
        self._is_actor = 'actor' in self.role
        self._is_rollout = 'rollout' in self.role
        self._is_ref = 'ref' in self.role

    def init_model(self):
        # 1. Build the actor (FSDP-wrapped model + optimizer)
        if self._is_actor:
            self.actor_module, self.optimizer = self._build_actor()

        # 2. Build the inference engine (vLLM/SGLang)
        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        # 3. Build reference model (can share weights with actor!)
        if self._is_ref:
            self.ref_module = self._build_reference()

    def _build_rollout(self):
        """
        This is where HybridEngine magic happens.
        """
        if self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager import FSDPVLLMShardingManager

            rollout = vLLMRollout(
                actor_module=self.actor_module,  # Pass FSDP module!
                config=self.config.rollout,
                tokenizer=self.tokenizer,
            )

            sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module,
                inference_engine=rollout.inference_engine,
                model_config=self.model_config,
            )

            return rollout, sharding_manager
```

**The key insight**: The `actor_module` is passed to both the `Rollout` and the `ShardingManager`. This enables weight sharing—the inference engine can access the actor's weights directly.

## 3.5 The Training Loop Orchestration

```python
# Simplified from verl/trainer/ppo/ray_trainer.py

class PPOTrainer:
    def training_step(self, batch):
        # ╔══════════════════════════════════════════════════════════════╗
        # ║  PHASE 1: GENERATION                                         ║
        # ║  - Sync actor weights to inference engine                    ║
        # ║  - Generate responses                                        ║
        # ║  - Release inference engine memory                           ║
        # ╚══════════════════════════════════════════════════════════════╝

        with self.rollout_sharding_manager:  # <-- CRITICAL!
            # Inside this context:
            # 1. __enter__: Actor weights synced to inference engine
            # 2. Generation happens with synced weights
            # 3. __exit__: Inference engine weights offloaded

            rollout_data = self.rollout_worker.generate_sequences(
                prompts=batch['prompts'],
                sampling_params=self.sampling_params,
            )

        # Now outside the context:
        # - Inference engine memory is freed
        # - Actor is back in training mode

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  PHASE 2: COMPUTE LOG PROBABILITIES                          ║
        # ║  - Run forward pass through actor for all generated tokens   ║
        # ║  - This gives us log π_θ(y|x) for policy gradient            ║
        # ╚══════════════════════════════════════════════════════════════╝

        old_log_probs = self.actor_worker.compute_log_probs(rollout_data)

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  PHASE 3: COMPUTE REFERENCE LOG PROBABILITIES                ║
        # ║  - Forward pass through frozen reference model               ║
        # ║  - Gives us log π_ref(y|x) for KL penalty                    ║
        # ╚══════════════════════════════════════════════════════════════╝

        with torch.no_grad():
            ref_log_probs = self.ref_worker.compute_log_probs(rollout_data)

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  PHASE 4: COMPUTE REWARDS AND ADVANTAGES                     ║
        # ║  - Score responses with reward function                      ║
        # ║  - Compute GRPO advantages (relative within group)           ║
        # ╚══════════════════════════════════════════════════════════════╝

        rewards = self.reward_fn(rollout_data)
        advantages = self.compute_advantages(rewards, rollout_data)

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  PHASE 5: POLICY GRADIENT UPDATE                             ║
        # ║  - Forward + backward through actor                          ║
        # ║  - Optimizer step                                            ║
        # ╚══════════════════════════════════════════════════════════════╝

        loss = self.actor_worker.update(
            rollout_data=rollout_data,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
        )

        return loss
```

---

# 4. The HybridEngine Pattern

## Motivating Questions

- **How can two systems share one GPU without running out of memory?** (Time-multiplexing + weight synchronization)
- **When should weights be synchronized?** (Before every generation, not continuously)
- **What happens to the inference engine's memory during training?** (It gets released!)

## 4.1 The Core Insight: Time-Multiplexing

The HybridEngine pattern recognizes that training and generation don't happen simultaneously. We can use the same GPU memory for different purposes at different times:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    TIME-MULTIPLEXED GPU MEMORY                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIME ──────────────────────────────────────────────────────────────────►  │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   TRAINING      │  │   GENERATION    │  │   TRAINING      │            │
│  │                 │  │                 │  │                 │            │
│  │  FSDP Model     │  │  InfEng Model   │  │  FSDP Model     │            │
│  │  Optimizer      │  │  KV Cache       │  │  Optimizer      │            │
│  │  Gradients      │  │  CUDA Graphs    │  │  Gradients      │            │
│  │  Activations    │  │                 │  │  Activations    │            │
│  │                 │  │                 │  │                 │            │
│  │  (InfEng: 0GB)  │  │  (Grads: 0GB)   │  │  (InfEng: 0GB)  │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│          │                    │                    │                       │
│          └──── sync ─────────┘                    │                       │
│                weights        └──── offload ──────┘                       │
│                                    weights                                 │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: The same 40GB of GPU memory can hold:
- During training: Activations, gradients, optimizer states
- During generation: KV cache, CUDA graphs, inference engine buffers

## 4.2 The ShardingManager: The Key Abstraction

The `ShardingManager` is a context manager that orchestrates the phase transitions:

```python
class BaseShardingManager:
    """
    Abstract base class for sharding managers.

    The sharding manager handles the critical transitions between
    training and inference phases in a HybridEngine setup.
    """

    def __enter__(self):
        """
        Called when entering generation phase.

        Responsibilities:
        1. Extract weights from training framework (FSDP)
        2. Transfer weights to inference engine
        3. Initialize inference engine memory (KV cache)
        4. Set model to eval mode
        """
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting generation phase.

        Responsibilities:
        1. Release inference engine memory
        2. Offload inference engine weights (optional)
        3. Set model back to train mode
        4. Clear CUDA cache
        """
        raise NotImplementedError
```

**Why a context manager?** The context manager pattern guarantees cleanup even if an exception occurs:

```python
# This is safe - __exit__ always runs
with sharding_manager:
    responses = inference_engine.generate(prompts)
    # Even if generate() raises an exception,
    # __exit__ will still clean up memory

# After the with block, we're guaranteed:
# - Inference engine memory is freed
# - Model is back in training mode
```

## 4.3 Memory Flow: A Detailed Walkthrough

Let's trace memory through a complete training step for a 1.5B model on a 120GB unified memory system:

```
INITIAL STATE (after model loading):
├── FSDP Actor: 3GB (model weights)
├── Optimizer: 12GB (AdamW states, lazily initialized)
├── Reference: 3GB (frozen weights)
├── InfEngine: 0GB (offloaded after init)
├── FREE: ~102GB
└── TOTAL USED: ~18GB

STEP 1: Enter ShardingManager.__enter__()
├── Action: Extract state_dict from FSDP
├── Temporary: +3GB (state_dict copy)
├── FREE: ~99GB
└── TOTAL USED: ~21GB

STEP 2: Sync weights to inference engine
├── Action: Copy state_dict to inference engine
├── InfEngine weights: +3GB
├── Delete state_dict: -3GB
├── FREE: ~99GB
└── TOTAL USED: ~21GB

STEP 3: Allocate KV cache
├── Action: inference_engine.init_cache_engine()
├── KV Cache: +40GB (at 35% utilization)
├── FREE: ~59GB
└── TOTAL USED: ~61GB

STEP 4: Generate responses
├── Action: inference_engine.generate()
├── Temporary buffers: +5GB (sampling, logits)
├── Peak during generation: ~66GB
├── After generation complete: -5GB (buffers freed)
└── TOTAL USED: ~61GB

STEP 5: Exit ShardingManager.__exit__()
├── Action: Free KV cache
├── KV Cache: -40GB
├── Action: Offload inference engine weights
├── InfEngine weights: -3GB (moved to CPU or released)
├── FREE: ~102GB
└── TOTAL USED: ~18GB (back to initial!)

STEP 6: Compute log probabilities (actor forward pass)
├── Action: Forward pass through FSDP actor
├── Activations: +20GB (with gradient checkpointing)
├── FREE: ~82GB
└── TOTAL USED: ~38GB

STEP 7: Compute reference log probabilities
├── Action: Forward pass through reference (no_grad)
├── Activations: +10GB (smaller without grad tracking)
├── Peak: ~48GB
├── After forward: -10GB (activations freed)
└── TOTAL USED: ~38GB

STEP 8: Policy gradient update (first PPO epoch)
├── Action: Forward + Backward + Optimizer step
├── First step loads optimizer states: +12GB
├── Gradients during backward: +3GB (peak)
├── After optimizer.step(): gradients freed
├── Peak during backward: ~63GB
└── TOTAL USED after step: ~30GB (optimizer now loaded)

STEP 9: Subsequent PPO epochs
├── Optimizer already loaded, no additional memory
├── Peak during backward: ~53GB
└── TOTAL USED: ~30GB
```

**Critical observations**:

1. **Peak memory occurs during training backward pass**, not generation
2. **Optimizer states load lazily** - first training step uses more memory
3. **KV cache is the largest temporary allocation** - tune `gpu_memory_utilization`
4. **State dict copy causes a brief spike** - unavoidable with current FSDP design

## 4.4 The vLLM HybridEngine Implementation

vLLM's integration with veRL is the reference implementation. Let's examine it in detail:

```python
# verl/workers/sharding_manager/fsdp_vllm.py

class FSDPVLLMShardingManager(BaseShardingManager):
    """
    Manages weight synchronization between FSDP and vLLM.

    Design decisions:
    1. Uses DTensor format for efficient weight transfer (no gather to rank 0)
    2. Saves/restores RNG state for reproducibility across TP ranks
    3. Optionally frees KV cache between generations
    """

    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config: PretrainedConfig,
        full_params: bool = False,  # If True, gather full state dict
        device_mesh: Optional[DeviceMesh] = None,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.full_params = full_params
        self.device_mesh = device_mesh

        # For deterministic generation across TP ranks
        self._rng_states: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        # 1. Save RNG state (for tensor parallel reproducibility)
        self._save_rng_states()

        # 2. Extract weights from FSDP
        # This is where the magic happens - FSDP can give us either
        # sharded (DTensor) or full state dicts
        params = self.module.state_dict()

        # 3. Determine the load format based on whether we have full params
        # - 'hf': Full tensors, standard HuggingFace format
        # - 'dtensor': Sharded tensors, PyTorch DTensor format
        load_format = 'hf' if self.full_params else 'dtensor'

        # 4. Sync weights to vLLM
        # This calls vLLM's weight loader, which handles the conversion
        self.inference_engine.sync_model_weights(params, load_format=load_format)

        # 5. Clean up the temporary state dict
        del params
        torch.cuda.empty_cache()

        # 6. Set model to eval mode (disables dropout, etc.)
        self.module.eval()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. Offload vLLM weights to free GPU memory
        self.inference_engine.offload_model_weights()

        # 2. Optionally free KV cache
        if getattr(self, 'free_cache_engine', False):
            self.inference_engine.free_cache_engine()

        # 3. Restore RNG state
        self._restore_rng_states()

        # 4. Set model back to training mode
        self.module.train()

        # 5. Clear CUDA cache
        torch.cuda.empty_cache()

        # Don't suppress exceptions
        return False

    def _save_rng_states(self):
        """Save RNG states for reproducibility across tensor parallel ranks."""
        self._rng_states = {
            'cuda': torch.cuda.get_rng_state(),
            'cpu': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            self._rng_states['cuda_all'] = torch.cuda.get_rng_state_all()

    def _restore_rng_states(self):
        """Restore saved RNG states."""
        torch.cuda.set_rng_state(self._rng_states['cuda'])
        torch.set_rng_state(self._rng_states['cpu'])
        if 'cuda_all' in self._rng_states:
            torch.cuda.set_rng_state_all(self._rng_states['cuda_all'])
```

> **Reference**: The vLLM integration code is in [verl/workers/sharding_manager/fsdp_vllm.py](https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/fsdp_vllm.py)

## 4.5 Design Decision: When to Sync Weights?

**Option 1: Sync before every generation** (veRL default)
```python
for step in training_steps:
    with sharding_manager:  # Syncs weights here
        responses = generate(prompts)
    loss = compute_and_update(responses)
```
- **Pros**: Responses always reflect latest policy
- **Cons**: Weight sync overhead every step
- **When to use**: Always, unless weight sync is prohibitively expensive

**Option 2: Sync every N steps**
```python
for step in training_steps:
    if step % sync_interval == 0:
        sharding_manager.sync_weights()
    responses = generate(prompts)  # Uses possibly stale weights
    loss = compute_and_update(responses)
```
- **Pros**: Reduced sync overhead
- **Cons**: Generation uses stale weights, may slow convergence
- **When to use**: Only if weight sync takes >10% of step time

**Option 3: Never sync (what broken SGLang does)**
```python
# DON'T DO THIS
for step in training_steps:
    responses = generate(prompts)  # Always uses initial weights!
    loss = compute_and_update(responses)
```
- **Pros**: No sync overhead
- **Cons**: Training doesn't actually improve generation!
- **When to use**: Never (this is a bug)

---

# 5. Weight Synchronization: The Critical Interface

## Motivating Questions

- **Why can't we just share a pointer to the weights?** (Different memory layouts, different processes)
- **What's a DTensor and why does it matter?** (FSDP's distributed tensor format)
- **How do we handle models sharded across multiple GPUs?** (Collective communication)

## 5.1 The Weight Synchronization Challenge

At first glance, weight sync seems simple: copy tensors from A to B. In practice, it's complicated by:

1. **Different tensor formats**: FSDP uses DTensor, inference engines use regular tensors
2. **Different memory layouts**: Training may use different sharding than inference
3. **Different processes**: SGLang runs in a subprocess with its own CUDA context
4. **Large data size**: 1.5B parameters = 3GB of data to transfer
5. **Naming conventions**: Parameter names may differ between frameworks

## 5.2 Understanding DTensor

FSDP (Fully Sharded Data Parallel) represents distributed tensors using PyTorch's DTensor abstraction:

```python
# Regular tensor
regular_tensor = torch.randn(1000, 1000)  # All data on one device

# DTensor: same logical tensor, but sharded across devices
from torch.distributed._tensor import DTensor, DeviceMesh, Shard

mesh = DeviceMesh("cuda", [0, 1, 2, 3])  # 4 GPUs
dtensor = DTensor.from_local(
    local_tensor,  # Each GPU has 1/4 of the data
    device_mesh=mesh,
    placements=[Shard(0)],  # Sharded along dimension 0
)

# To get the full tensor (EXPENSIVE - requires all-gather)
full_tensor = dtensor.full_tensor()  # Gathers all shards to one device
```

**Why this matters for HybridEngine**: When you call `fsdp_model.state_dict()`, you get DTensors. The inference engine doesn't understand DTensors, so you need to convert.

```python
def convert_dtensor_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict containing DTensors to regular tensors.

    WARNING: This gathers all shards to the current device!
    Memory usage will spike temporarily.
    """
    converted = {}
    for name, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            # Gather all shards - this is the expensive part
            converted[name] = tensor.full_tensor()
        else:
            converted[name] = tensor
    return converted
```

## 5.3 Weight Loading Strategies

Different strategies for loading weights into the inference engine:

### Strategy 1: Full State Dict (Simple but Memory-Intensive)

```python
def sync_weights_full(fsdp_model, inference_engine):
    """
    Gather full state dict and load into inference engine.

    Memory: Requires 2x model weights temporarily
    Speed: One all-gather, then sequential copy
    """
    # Configure FSDP to return full state dict
    FSDP.set_state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
    )

    # Get full state dict (triggers all-gather)
    state_dict = fsdp_model.state_dict()  # +3GB temporary

    # Load into inference engine
    inference_engine.model.load_state_dict(state_dict)

    # Clean up
    del state_dict
    torch.cuda.empty_cache()
```

**When to use**: Single GPU, or when simplicity matters more than memory

### Strategy 2: DTensor Direct Load (Memory-Efficient)

```python
def sync_weights_dtensor(fsdp_model, inference_engine):
    """
    Load weights directly from DTensors without gathering.

    Memory: Minimal overhead (no full copy)
    Speed: Faster for multi-GPU (parallel loads)
    """
    # Configure FSDP to return sharded state dict
    FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)
    state_dict = fsdp_model.state_dict()

    # vLLM's DTensor loader handles the conversion internally
    from verl.third_party.vllm import load_dtensor_weights
    load_dtensor_weights(
        state_dict,
        inference_engine.model,
        inference_engine.config,
    )

    del state_dict
    torch.cuda.empty_cache()
```

**When to use**: Multi-GPU setups, memory-constrained environments

### Strategy 3: Disk-Based Transfer (For Process Isolation)

```python
def sync_weights_via_disk(fsdp_model, inference_engine, temp_dir="/tmp"):
    """
    Save weights to disk, then load in inference engine.

    Memory: Only one copy in memory at a time
    Speed: Slowest (disk I/O)
    Benefit: Works across process boundaries!
    """
    import tempfile
    from safetensors.torch import save_file, load_file

    # Save FSDP weights to disk
    state_dict = fsdp_model.state_dict()
    weights_path = f"{temp_dir}/weights.safetensors"
    save_file(state_dict, weights_path)
    del state_dict
    torch.cuda.empty_cache()

    # Load into inference engine (can be in different process!)
    inference_engine.update_weights_from_disk(temp_dir)
```

**When to use**: When training and inference are in different processes (like SGLang!)

## 5.4 Parameter Name Mapping

Different frameworks use different naming conventions:

```python
# HuggingFace naming
"model.layers.0.self_attn.q_proj.weight"

# FSDP might add prefixes
"_fsdp_wrapped_module.model.layers.0.self_attn.q_proj.weight"

# vLLM might expect
"model.layers.0.self_attn.qkv_proj.weight"  # Fused QKV!
```

You need a mapping function:

```python
def map_hf_to_vllm_name(hf_name: str) -> str:
    """
    Map HuggingFace parameter names to vLLM names.

    vLLM fuses some layers for efficiency:
    - q_proj, k_proj, v_proj → qkv_proj
    - gate_proj, up_proj → gate_up_proj
    """
    # Remove FSDP prefix if present
    if hf_name.startswith("_fsdp_wrapped_module."):
        hf_name = hf_name[len("_fsdp_wrapped_module."):]

    # Handle fused attention
    for proj in ['q_proj', 'k_proj', 'v_proj']:
        if proj in hf_name:
            return hf_name.replace(proj, 'qkv_proj')

    # Handle fused MLP
    for proj in ['gate_proj', 'up_proj']:
        if proj in hf_name:
            return hf_name.replace(proj, 'gate_up_proj')

    return hf_name
```

> **Reference**: vLLM's weight loading code is in [vllm/model_executor/model_loader/weight_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/weight_utils.py)

## 5.5 Common Weight Sync Bugs

**Bug 1: Shape mismatch due to fused layers**
```
RuntimeError: shape mismatch: param.shape = [4096, 1536],
              loaded.shape = [1536, 1536]
```
**Cause**: vLLM fuses Q, K, V into one tensor; HuggingFace keeps them separate
**Fix**: Handle fused layers in your weight loader

**Bug 2: Missing keys**
```
Missing keys: ['model.layers.0.self_attn.rotary_emb.inv_freq']
```
**Cause**: Some buffers are computed, not loaded
**Fix**: Use `strict=False` or explicitly compute missing buffers

**Bug 3: CUDA context mismatch (SGLang specific)**
```
CUDA error: invalid device context
```
**Cause**: Tensors created in one process, used in another
**Fix**: Transfer via CPU or disk, not direct CUDA copy

---

# 6. Memory Management: The Hidden Complexity

## Motivating Questions

- **Why does PyTorch report 8GB allocated but 76GB reserved?** (Caching allocator)
- **Where does the memory spike come from during training?** (Logits tensor!)
- **Why does my training OOM on step 2 but not step 1?** (Lazy optimizer initialization)

## 6.1 Understanding PyTorch's Memory Allocator

PyTorch uses a caching memory allocator that pools GPU memory:

```python
# What you might expect:
tensor = torch.randn(1000, 1000, device='cuda')  # Allocates memory
del tensor  # Frees memory immediately

# What actually happens:
tensor = torch.randn(1000, 1000, device='cuda')  # Allocates from pool
del tensor  # Returns to pool, but POOL KEEPS THE MEMORY

# Memory stays reserved until:
torch.cuda.empty_cache()  # Returns unused cached memory to CUDA
```

**Two key metrics**:
- `torch.cuda.memory_allocated()`: Memory actually holding tensors
- `torch.cuda.memory_reserved()`: Memory held by the allocator (always >= allocated)

```python
def log_memory(label: str):
    """Log detailed GPU memory status."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{label}:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Cached:    {reserved - allocated:.2f} GB")

# Typical output during training:
# Forward pass:
#   Allocated: 8.66 GB
#   Reserved:  32.00 GB
#   Cached:    23.34 GB
#
# Backward pass (peak):
#   Allocated: 20.18 GB
#   Reserved:  76.00 GB  <-- Allocator reserves extra for fragmentation!
#   Cached:    55.82 GB
```

**Why reserved >> allocated?** The allocator over-provisions to handle:
1. Memory fragmentation
2. Varying tensor sizes during backward pass
3. CUDA alignment requirements

## 6.2 The Logits Memory Problem

The single largest memory consumer during RLHF training is often the **logits tensor**:

```python
# Model forward pass
output = model(input_ids)  # input_ids: [batch, seq_len]
logits = output.logits     # logits: [batch, seq_len, vocab_size]

# For Qwen2.5-1.5B:
# vocab_size = 151,936 (HUGE!)
# batch = 4, seq_len = 1280
# Memory: 4 × 1280 × 151,936 × 2 bytes (bf16) = 1.56 GB

# And that's just ONE tensor!
# During log_softmax:
log_probs = F.log_softmax(logits, dim=-1)  # Another 1.56 GB!
# During entropy computation:
probs = F.softmax(logits, dim=-1)  # Another 1.56 GB!
# Peak: ~4.7 GB just for these three tensors!
```

**The math**:
$$\text{Logits Memory} = B \times S \times V \times \text{bytes\_per\_element}$$

Where $V$ (vocab size) is the killer:
| Model | Vocab Size | Logits for batch=4, seq=1024 |
|-------|------------|------------------------------|
| LLaMA-2 | 32,000 | 0.26 GB |
| Mistral | 32,768 | 0.27 GB |
| Qwen2.5 | 151,936 | **1.25 GB** |

## 6.3 Memory-Efficient Log Probability Computation

The standard approach materializes the full logits tensor:

```python
# MEMORY INEFFICIENT - Standard approach
def compute_log_probs_naive(logits, labels):
    """
    Compute log probabilities for each token.

    logits: [batch, seq, vocab_size]
    labels: [batch, seq]
    """
    # This creates a [batch, seq, vocab_size] tensor!
    log_probs = F.log_softmax(logits, dim=-1)

    # Then we only take the values at label positions
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
    return gathered.squeeze(-1)  # [batch, seq]
```

A memory-efficient approach avoids materializing the full softmax:

```python
# MEMORY EFFICIENT - Gather then softmax
def compute_log_probs_efficient(logits, labels):
    """
    Compute log probabilities without materializing full softmax.

    Key insight: log_softmax(x)[i] = x[i] - logsumexp(x)
    We only need x[i] (the logit at label position), not all of x!
    """
    # Gather the logits at label positions: O(batch × seq) memory
    logits_at_labels = torch.gather(
        logits, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq]

    # Compute logsumexp over vocab dimension: O(batch × seq) output
    # This iterates over vocab internally, but output is small
    logsumexp = torch.logsumexp(logits, dim=-1)  # [batch, seq]

    # log_prob = logit - logsumexp
    return logits_at_labels - logsumexp
```

**Memory comparison** for batch=4, seq=1280, vocab=151,936:

| Approach | Peak Memory | Explanation |
|----------|-------------|-------------|
| Naive | 3.12 GB | logits + log_probs |
| Efficient | 0.02 GB | Only gathered values + logsumexp |

## 6.4 The Fused Kernel Solution

The ultimate solution fuses the linear layer (lm_head) with the cross-entropy computation:

```python
# Standard approach (two steps):
hidden = model.transformer(input_ids)        # [batch, seq, hidden]
logits = model.lm_head(hidden)               # [batch, seq, vocab] - HUGE!
loss = F.cross_entropy(logits, labels)

# Fused approach (one kernel):
hidden = model.transformer(input_ids)        # [batch, seq, hidden]
loss = fused_linear_cross_entropy(
    hidden,
    model.lm_head.weight,
    labels
)  # Never materializes [batch, seq, vocab]!
```

**How does it work?** The fused kernel computes:
$$\text{loss}_i = -\log \frac{\exp(h_i \cdot w_{y_i})}{\sum_j \exp(h_i \cdot w_j)}$$

In chunks over the vocabulary dimension, accumulating the logsumexp without storing the full logits.

> **Reference**: [Liger Kernel](https://github.com/linkedin/Liger-Kernel) provides fused kernels. veRL upstream has [fused linear cross-entropy](https://github.com/volcengine/verl/tree/main/verl/utils/kernel) (PR #462).

## 6.5 Memory Budget Planning

Before running training, calculate your memory budget:

```python
def plan_memory_budget(
    model_params: int,
    batch_size: int,
    n_samples: int,  # GRPO samples per prompt
    seq_length: int,
    vocab_size: int,
    total_memory_gb: float,
    safety_margin: float = 0.95,  # Ray OOM threshold
):
    """
    Plan memory budget for RLHF training.
    """
    available = total_memory_gb * safety_margin
    print(f"Available memory: {available:.1f} GB")

    # Static allocations
    model_gb = model_params * 2 / 1e9  # bf16
    optimizer_gb = model_params * 8 / 1e9  # AdamW
    reference_gb = model_params * 2 / 1e9  # Reference model (if separate)
    static = model_gb + optimizer_gb + reference_gb

    print(f"\nStatic allocations:")
    print(f"  Model: {model_gb:.1f} GB")
    print(f"  Optimizer: {optimizer_gb:.1f} GB")
    print(f"  Reference: {reference_gb:.1f} GB")
    print(f"  Total static: {static:.1f} GB")

    remaining = available - static
    print(f"\nRemaining for dynamic: {remaining:.1f} GB")

    # Generation phase
    total_sequences = batch_size * n_samples
    kv_cache_per_seq = 2 * 28 * 1536 * seq_length * 2 / 1e9  # K+V per layer
    kv_cache_total = kv_cache_per_seq * total_sequences

    print(f"\nGeneration phase:")
    print(f"  Sequences: {total_sequences}")
    print(f"  KV cache per seq: {kv_cache_per_seq*1000:.1f} MB")
    print(f"  Total KV cache: {kv_cache_total:.1f} GB")

    # Training phase
    micro_batch = 4  # Typical
    logits_gb = micro_batch * seq_length * vocab_size * 2 / 1e9
    activations_gb = micro_batch * seq_length * 1536 * 28 * 2 * 0.3 / 1e9
    gradients_gb = model_gb

    print(f"\nTraining phase (micro_batch={micro_batch}):")
    print(f"  Logits: {logits_gb:.1f} GB")
    print(f"  Activations: {activations_gb:.1f} GB")
    print(f"  Gradients: {gradients_gb:.1f} GB")
    print(f"  Training total: {logits_gb + activations_gb + gradients_gb:.1f} GB")

    # Verdict
    gen_total = kv_cache_total + model_gb  # Need model during generation
    train_total = logits_gb + activations_gb + gradients_gb

    print(f"\n{'='*50}")
    print(f"Generation peak: {gen_total:.1f} GB {'✓' if gen_total < remaining else '✗ OOM!'}")
    print(f"Training peak: {train_total:.1f} GB {'✓' if train_total < remaining else '✗ OOM!'}")

# Example for Qwen2.5-1.5B on GB10
plan_memory_budget(
    model_params=1.5e9,
    batch_size=8,
    n_samples=4,
    seq_length=1280,
    vocab_size=151936,
    total_memory_gb=120,
)
```

## 6.6 Common Memory Issues and Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| OOM during generation | KV cache too large | Reduce `gpu_memory_utilization` |
| OOM at step 1 | Model + optimizer don't fit | Enable gradient checkpointing, reduce batch |
| OOM at step 2 | Optimizer states lazy load | Account for +12GB at first training step |
| OOM during backward | Activation memory | Reduce micro_batch_size |
| Gradual memory growth | Memory leak/fragmentation | Call `empty_cache()` between steps |
| OOM with long sequences | Quadratic attention | Use flash attention, chunked computation |

---

# 7. The SGLang Challenge: A Case Study in Systems Integration

## Motivating Questions

- **Why does SGLang work standalone but fail in Ray?** (Subprocess architecture conflicts)
- **What makes vLLM's architecture more compatible?** (In-process execution)
- **Can we fix SGLang integration, or is it fundamentally incompatible?** (It's fixable, but requires SGLang changes)

## 7.1 SGLang's Architecture: Designed for Serving

SGLang was designed as a high-performance serving system, not a training component:

```
SGLang Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                        Main Process                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    TokenizerManager                                  ││
│  │  - Handles API requests                                              ││
│  │  - Manages tokenization                                              ││
│  │  - Routes to scheduler                                               ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                              │                                           │
│                              │ IPC (pickle + shared memory)              │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │              Scheduler Subprocess (multiprocessing.spawn)            ││
│  │  - Continuous batching                                               ││
│  │  - Memory management                                                 ││
│  │  - KV cache allocation                                               ││
│  │  - Model execution                                                   ││
│  │                                                                       ││
│  │  ╔═══════════════════════════════════════════════════════════════╗  ││
│  │  ║                    MODEL WEIGHTS LIVE HERE                     ║  ││
│  │  ║                    (Separate CUDA context!)                    ║  ││
│  │  ╚═══════════════════════════════════════════════════════════════╝  ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: The model weights are in the **scheduler subprocess**, not the main process. This is great for serving (isolates the model from request handling) but terrible for HybridEngine (we need to sync weights from a different process!).

## 7.2 The CUDA Context Conflict

When Ray spawns a worker that then spawns SGLang, we get conflicting CUDA contexts:

```
Process hierarchy:
Ray Worker (CUDA Context A)
├── FSDP Actor (uses Context A)
└── SGLang Engine
    └── TokenizerManager (Context A)
        └── Scheduler Subprocess (Context B ← CONFLICT!)
            └── Model Weights (Context B)
```

**What happens when we try to sync weights**:

1. Get state dict from FSDP (Context A tensors)
2. Call `engine.update_weights_from_tensor(state_dict)`
3. SGLang serializes tensors with pickle (includes CUDA device info)
4. Sends to scheduler subprocess via IPC
5. Scheduler deserializes... **SEGFAULT!**

**Why?** CUDA tensors are tied to their creating context. When the subprocess tries to access tensors created in Context A, it fails because it's in Context B.

## 7.3 Weight Sync Methods We Tried

### Method 1: `update_weights_from_tensor` (SEGFAULT)

```python
# What we tried:
state_dict = fsdp_model.state_dict()
engine.update_weights_from_tensor(state_dict)
# Result: SEGFAULT in cuMemcpyDtoDAsync_v2
```

**Why it fails**: Pickle serialization of CUDA tensors doesn't work across CUDA contexts.

### Method 2: `update_weights_from_distributed` (GPU conflict)

```python
# What we tried:
nccl_group = torch.distributed.new_group([0, 1])
engine.update_weights_from_distributed(state_dict, nccl_group)
# Result: "Duplicate GPU detected" error
```

**Why it fails**: On a single GPU, both processes try to use the same device, but NCCL expects different devices.

### Method 3: `update_weights_from_disk` (Works, but slow)

```python
# What we tried:
save_safetensors(state_dict, "/tmp/weights")
success, msg = engine.update_weights_from_disk("/tmp/weights")
# Result: Works standalone, but hangs in Ray
```

**Why it partially works**: Disk-based transfer avoids CUDA context issues. But in Ray, the subprocess still has context conflicts when loading.

### Method 4: HTTP Server Mode (Works, but overhead)

```python
# What we implemented:
# Start SGLang as completely separate process
subprocess.Popen(["python", "-m", "sglang.launch_server", ...])

# Communicate via HTTP
requests.post("http://localhost:30000/generate", ...)
requests.post("http://localhost:30000/update_weights_from_disk", ...)
# Result: Works! But has HTTP overhead and disk I/O
```

**Why it works**: Complete process isolation means no CUDA context conflicts.

## 7.4 Comparing vLLM and SGLang Architectures

| Aspect | vLLM | SGLang |
|--------|------|--------|
| Execution model | In-process | Subprocess |
| CUDA context | Shared with caller | Separate (subprocess) |
| Weight sync | Direct tensor copy | IPC serialization |
| HybridEngine compatibility | Excellent | Poor |
| Serving performance | Great | Great |
| Complexity | Lower | Higher |

**The fundamental difference**:

```python
# vLLM: Model runs in your process
llm = vllm.LLM(model_path)
# llm.model is a regular PyTorch module in YOUR CUDA context
# You can directly manipulate its parameters

# SGLang: Model runs in a subprocess
engine = sglang.Engine(model_path)
# engine doesn't hold the model - the scheduler subprocess does!
# You can only communicate via IPC
```

## 7.5 The `expandable_segments` Incompatibility

An additional complication: SGLang's memory management is incompatible with PyTorch's `expandable_segments`:

```python
# If you set this environment variable:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# SGLang will crash with:
# RuntimeError: expandable_segments is not supported with TorchMemorySaver

# Why? SGLang uses torch_memory_saver which uses CUDAPluggableAllocator
# CUDAPluggableAllocator is incompatible with expandable_segments
```

**From torch_memory_saver source**:
```python
def _sanity_checks():
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in conf:
        raise RuntimeError(
            "TorchMemorySaver is disabled because "
            "expandable_segments is not supported yet."
        )
```

> **Reference**: [PyTorch Issue #147851](https://github.com/pytorch/pytorch/issues/147851)

## 7.6 Lessons from the SGLang Integration Attempt

1. **Process architecture matters**: Subprocess-based systems are harder to integrate with colocated training

2. **CUDA contexts are tricky**: Cross-process CUDA communication requires careful handling

3. **Test integration early**: Run your inference engine inside Ray before building the full system

4. **Have a fallback**: HTTP server mode works even when direct integration doesn't

5. **Read the source**: Understanding `torch_memory_saver` and `CUDAPluggableAllocator` explained the `expandable_segments` issue

6. **Upstream changes may be needed**: True SGLang HybridEngine support requires SGLang to support in-process execution

---

# 8. Implementation Reference: Code Patterns That Work

## Motivating Questions

- **What does production-quality RLHF code actually look like?** (Patterns from veRL that work)
- **How do I structure the training loop for reliability?** (Error handling, checkpointing, logging)
- **What abstractions pay off and which ones don't?** (Workers, managers, rollouts)

## 8.1 The Worker Abstraction: Encapsulating Complexity

veRL uses "workers" as the fundamental abstraction unit. Each worker is a Ray actor that owns GPU resources and encapsulates a specific role:

```python
# Simplified from verl/workers/fsdp_workers.py

import ray
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

@ray.remote(num_gpus=1)
class ActorRolloutRefWorker:
    """
    A colocated worker that handles multiple roles.

    Design decision: Why colocate actor, rollout, and reference?
    - Memory sharing: Same GPU can time-multiplex roles
    - Weight sharing: Actor and reference can share base weights
    - Reduced communication: No cross-worker tensor transfers

    Alternative: Separate workers per role (uses more GPUs, simpler code)
    """

    def __init__(self, config, role):
        """
        Initialize the worker with specified roles.

        Args:
            config: OmegaConf configuration object
            role: String like "actor_rollout_ref" specifying active roles
        """
        self.config = config
        self.role = role

        # Parse which roles this worker handles
        self._is_actor = "actor" in role
        self._is_rollout = "rollout" in role
        self._is_ref = "ref" in role

        # Placeholders - initialized in init_model()
        self.actor_module = None
        self.optimizer = None
        self.rollout = None
        self.sharding_manager = None
        self.reference = None

    def init_model(self):
        """
        Initialize models after worker is placed on GPU.

        Why separate from __init__?
        - Ray actors are created before GPU assignment
        - Model loading needs to happen ON the assigned GPU
        - This pattern is common in distributed ML code
        """
        # 1. Build tokenizer (shared by all roles)
        self.tokenizer = self._build_tokenizer()

        # 2. Build actor (training model + optimizer)
        if self._is_actor:
            self.actor_module = self._build_fsdp_model()
            self.optimizer = self._build_optimizer(self.actor_module)

        # 3. Build rollout (inference engine + sharding manager)
        if self._is_rollout:
            self.rollout = self._build_rollout()
            self.sharding_manager = self._build_sharding_manager()

        # 4. Build reference (can share weights with actor!)
        if self._is_ref:
            if self._is_actor:
                # Share the same module - reference is just actor in eval mode
                self.reference = self.actor_module
            else:
                # Separate reference model (uses more memory)
                self.reference = self._build_fsdp_model()

    def _build_fsdp_model(self):
        """
        Build an FSDP-wrapped model.

        FSDP provides:
        - Sharded optimizer states (memory efficient)
        - Automatic gradient synchronization
        - Mixed precision training
        """
        from transformers import AutoModelForCausalLM
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # Critical for efficiency
        )

        # Enable gradient checkpointing (trade compute for memory)
        if self.config.model.get("enable_gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

        # Wrap with FSDP
        # auto_wrap_policy tells FSDP which modules to shard
        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=torch.cuda.current_device(),
        )

        return model

    def _build_rollout(self):
        """
        Build the inference engine for generation.

        This is where HybridEngine integration happens.
        """
        if self.config.rollout.name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            return vLLMRollout(
                actor_module=self.actor_module,  # Pass FSDP module!
                config=self.config.rollout,
                tokenizer=self.tokenizer,
            )
        elif self.config.rollout.name == "hf":
            # Simple HuggingFace generate() fallback
            from verl.workers.rollout.hf_rollout import HFRollout
            return HFRollout(
                actor_module=self.actor_module,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
            )
        else:
            raise ValueError(f"Unknown rollout: {self.config.rollout.name}")
```

**Why this pattern works**:
1. **Encapsulation**: All model state is owned by one Ray actor
2. **GPU isolation**: Each worker gets exclusive GPU access
3. **Flexibility**: Roles can be combined or separated based on memory
4. **Testability**: Workers can be unit tested in isolation

> **Reference**: veRL worker implementation in [verl/workers/fsdp_workers.py](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py)

## 8.2 The Training Loop: Orchestrating Phases

The training loop coordinates all phases of RLHF:

```python
# Simplified from verl/trainer/ppo/ray_trainer.py

class PPOTrainer:
    """
    Orchestrates the RLHF training loop.

    Design principles:
    1. Single source of truth for training state
    2. Clear phase boundaries
    3. Comprehensive logging for debugging
    4. Checkpoint-able at any point
    """

    def __init__(self, config):
        self.config = config
        self.step = 0
        self.epoch = 0

        # Build workers
        self.workers = self._create_workers()

        # Build data pipeline
        self.dataloader = self._create_dataloader()

        # Metrics tracking
        self.metrics_logger = MetricsLogger()

    def train(self):
        """Main training loop."""

        for epoch in range(self.config.trainer.total_epochs):
            self.epoch = epoch

            for batch in self.dataloader:
                # Training step with comprehensive error handling
                try:
                    metrics = self.training_step(batch)
                    self.log_metrics(metrics)

                except OutOfMemoryError:
                    # Handle OOM gracefully
                    self._handle_oom()
                    continue

                except Exception as e:
                    # Log error but don't crash
                    self.log_error(e)
                    if self.config.trainer.strict:
                        raise
                    continue

                self.step += 1

                # Checkpointing
                if self.step % self.config.trainer.save_freq == 0:
                    self.save_checkpoint()

                # Evaluation
                if self.step % self.config.trainer.eval_freq == 0:
                    self.evaluate()

    def training_step(self, batch):
        """
        Execute one complete training step.

        Memory flow:
        1. Generation: ~60GB peak (KV cache + model)
        2. Log prob compute: ~40GB peak (activations)
        3. Training: ~50GB peak (gradients + optimizer)

        Note: Phases are sequential - memory is reused between phases.
        """
        metrics = {}

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: GENERATION
        # ═══════════════════════════════════════════════════════════════

        with self.timing("generation"):
            # The sharding manager context is CRITICAL:
            # - __enter__: Syncs actor weights to inference engine
            # - __exit__: Frees inference engine memory
            with self.sharding_manager:
                responses = self.rollout_worker.generate_sequences(
                    prompts=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    sampling_params=SamplingParams(
                        temperature=self.config.rollout.temperature,
                        max_new_tokens=self.config.data.max_response_length,
                        n=self.config.rollout.n,  # GRPO samples per prompt
                    ),
                )

        # At this point:
        # - Inference engine memory is freed
        # - responses contains generated token IDs
        metrics["generation_time"] = self.timing.last("generation")
        metrics["num_tokens_generated"] = responses.num_tokens()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: COMPUTE REWARDS
        # ═══════════════════════════════════════════════════════════════

        with self.timing("reward"):
            # Decode responses for reward computation
            decoded = self.tokenizer.batch_decode(
                responses.response_ids,
                skip_special_tokens=True,
            )

            # Compute rewards (typically rule-based or model-based)
            rewards = self.reward_fn(
                prompts=batch["prompts"],
                responses=decoded,
            )

        metrics["reward_mean"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: COMPUTE LOG PROBABILITIES
        # ═══════════════════════════════════════════════════════════════

        with self.timing("log_prob"):
            # Prepare full sequences (prompt + response)
            full_sequences = torch.cat([
                responses.prompt_ids,
                responses.response_ids,
            ], dim=1)

            # Compute log probs from actor (current policy)
            with self.timing("actor_logprob"):
                actor_log_probs = self.compute_log_probs(
                    self.actor_module,
                    full_sequences,
                )

            # Compute log probs from reference (frozen policy)
            with torch.no_grad():
                with self.timing("ref_logprob"):
                    ref_log_probs = self.compute_log_probs(
                        self.reference_module,
                        full_sequences,
                    )

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: COMPUTE ADVANTAGES (GRPO)
        # ═══════════════════════════════════════════════════════════════

        with self.timing("advantage"):
            # GRPO: Relative advantages within prompt groups
            advantages = self.compute_grpo_advantages(
                rewards=rewards,
                n_samples=self.config.rollout.n,
            )

            # Compute KL divergence for penalty
            kl_div = (actor_log_probs - ref_log_probs).sum(dim=-1)

        metrics["kl_mean"] = kl_div.mean().item()
        metrics["advantage_mean"] = advantages.mean().item()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: POLICY GRADIENT UPDATE
        # ═══════════════════════════════════════════════════════════════

        with self.timing("update"):
            # Micro-batching for memory efficiency
            num_micro_batches = len(responses) // self.config.actor.ppo_micro_batch_size

            total_loss = 0.0

            for micro_idx in range(num_micro_batches):
                # Get micro-batch slice
                start = micro_idx * self.config.actor.ppo_micro_batch_size
                end = start + self.config.actor.ppo_micro_batch_size

                # Forward pass
                new_log_probs = self.compute_log_probs(
                    self.actor_module,
                    full_sequences[start:end],
                )

                # Compute loss
                loss = self.compute_policy_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=actor_log_probs[start:end].detach(),
                    advantages=advantages[start:end],
                    kl_div=kl_div[start:end],
                )

                # Scale for gradient accumulation
                scaled_loss = loss / num_micro_batches
                scaled_loss.backward()

                total_loss += loss.item()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(),
                self.config.actor.max_grad_norm,
            )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        metrics["loss"] = total_loss / num_micro_batches
        metrics["update_time"] = self.timing.last("update")

        return metrics
```

## 8.3 Memory-Safe Log Probability Computation

The log probability computation is memory-critical. Here's a production implementation:

```python
def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities for response tokens.

    Memory optimization: Process in chunks to avoid OOM on long sequences.

    Args:
        model: The language model (FSDP wrapped)
        input_ids: [batch, seq_len] token IDs
        attention_mask: [batch, seq_len] attention mask
        response_mask: [batch, seq_len] mask for response tokens only

    Returns:
        log_probs: [batch, seq_len] log probabilities (0 for prompt tokens)
    """
    batch_size, seq_len = input_ids.shape
    vocab_size = model.config.vocab_size

    # Configure chunking based on available memory
    # Rule of thumb: chunk_size * vocab_size * 4 bytes < 1GB
    max_chunk_size = min(256, 1_000_000_000 // (vocab_size * 4))

    all_log_probs = []

    # Process sequence in chunks
    for chunk_start in range(0, seq_len, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, seq_len)

        # Forward pass for chunk
        # Note: We need attention to all previous tokens, not just the chunk
        outputs = model(
            input_ids=input_ids[:, :chunk_end],
            attention_mask=attention_mask[:, :chunk_end],
            use_cache=False,  # Disable KV cache for training
        )

        # Extract logits for this chunk only
        chunk_logits = outputs.logits[:, chunk_start:chunk_end, :]  # [batch, chunk, vocab]

        # Get labels (shifted by 1)
        chunk_labels = input_ids[:, chunk_start+1:chunk_end+1]  # [batch, chunk]

        # Compute log probs efficiently (gather then logsumexp)
        chunk_log_probs = compute_log_probs_efficient(chunk_logits, chunk_labels)

        all_log_probs.append(chunk_log_probs)

        # Free intermediate tensors
        del outputs, chunk_logits

    # Concatenate chunks
    log_probs = torch.cat(all_log_probs, dim=1)

    # Mask out prompt tokens (we only want response log probs)
    log_probs = log_probs * response_mask

    return log_probs


def compute_log_probs_efficient(logits, labels):
    """
    Memory-efficient log probability computation.

    Standard: log_softmax creates [batch, seq, vocab] then gathers
    Efficient: Gather first, compute logsumexp separately

    Memory savings: vocab_size factor (~150k for Qwen)
    """
    # Gather logits at label positions: [batch, seq]
    logits_at_labels = torch.gather(
        logits,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # Compute logsumexp over vocab: [batch, seq]
    # This iterates over vocab but output is small
    logsumexp = torch.logsumexp(logits, dim=-1)

    # log_softmax(x)[i] = x[i] - logsumexp(x)
    return logits_at_labels - logsumexp
```

## 8.4 The Rollout Interface: Abstraction Over Inference Engines

A clean rollout interface allows swapping inference engines:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SamplingParams:
    """Parameters for text generation."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 256
    n: int = 1  # Number of responses per prompt
    stop_sequences: Optional[List[str]] = None


@dataclass
class RolloutOutput:
    """Output from generation."""
    prompt_ids: torch.Tensor      # [batch, prompt_len]
    response_ids: torch.Tensor    # [batch, response_len]
    log_probs: torch.Tensor       # [batch, response_len] - can be zeros

    def num_tokens(self) -> int:
        return self.response_ids.numel()

    def full_ids(self) -> torch.Tensor:
        return torch.cat([self.prompt_ids, self.response_ids], dim=1)


class BaseRollout(ABC):
    """
    Abstract base class for rollout engines.

    Design principle: Hide inference engine complexity behind a simple interface.
    Implementations can use vLLM, SGLang, HuggingFace, or anything else.
    """

    @abstractmethod
    def generate_sequences(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> RolloutOutput:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            sampling_params: Generation parameters

        Returns:
            RolloutOutput containing generated sequences
        """
        raise NotImplementedError


class vLLMRollout(BaseRollout):
    """vLLM-based rollout implementation."""

    def __init__(self, actor_module, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Import veRL's custom vLLM wrapper
        from verl.third_party.vllm import LLM

        # Initialize with HybridEngine support
        self.inference_engine = LLM(
            actor_module=actor_module,  # Receives FSDP module
            tokenizer=tokenizer,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.get("enforce_eager", False),
            dtype="bfloat16",
            load_format="dummy",  # Don't load from disk!
        )

        # Offload weights after initialization
        self.inference_engine.offload_model_weights()

    def generate_sequences(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> RolloutOutput:
        """Generate using vLLM."""

        # Convert to vLLM sampling params
        vllm_params = vllm.SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            max_tokens=sampling_params.max_new_tokens,
            n=sampling_params.n,
        )

        # Generate
        outputs = self.inference_engine.generate(
            prompt_token_ids=prompts.tolist(),
            sampling_params=vllm_params,
        )

        # Convert outputs to tensors
        response_ids = self._collate_outputs(outputs)

        return RolloutOutput(
            prompt_ids=prompts,
            response_ids=response_ids,
            # vLLM can return log_probs, but veRL typically recomputes them
            # for consistency with the training forward pass
            log_probs=torch.zeros_like(response_ids, dtype=torch.float),
        )


class HFRollout(BaseRollout):
    """
    HuggingFace generate() fallback.

    Use this when:
    - Debugging (simpler code path)
    - vLLM/SGLang not available
    - Very small batches where inference engine overhead dominates

    Don't use when:
    - Performance matters (5-10x slower)
    - Large batches (memory inefficient)
    """

    def __init__(self, actor_module, config, tokenizer):
        self.module = actor_module
        self.config = config
        self.tokenizer = tokenizer

    def generate_sequences(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> RolloutOutput:
        """Generate using HuggingFace generate()."""

        # Switch to eval mode
        self.module.eval()

        with torch.no_grad():
            outputs = self.module.generate(
                input_ids=prompts,
                attention_mask=attention_mask,
                max_new_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                do_sample=sampling_params.temperature > 0,
                num_return_sequences=sampling_params.n,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Switch back to train mode
        self.module.train()

        # Split outputs into prompts and responses
        prompt_len = prompts.shape[1]
        response_ids = outputs[:, prompt_len:]

        return RolloutOutput(
            prompt_ids=prompts.repeat_interleave(sampling_params.n, dim=0),
            response_ids=response_ids,
            log_probs=torch.zeros_like(response_ids, dtype=torch.float),
        )
```

## 8.5 Checkpointing: Resumable Training

Production training needs robust checkpointing:

```python
import os
import json
from pathlib import Path

class CheckpointManager:
    """
    Manages training checkpoints for resumability.

    Saves:
    - Model weights (FSDP sharded)
    - Optimizer state
    - Training state (step, epoch, metrics)
    - Configuration
    """

    def __init__(self, save_dir: str, max_checkpoints: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        metrics: dict,
    ):
        """
        Save a checkpoint.

        Uses FSDP's sharded state dict for memory efficiency.
        """
        checkpoint_dir = self.save_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # 1. Save model weights (FSDP sharded)
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        model_state = model.state_dict()
        torch.save(model_state, checkpoint_dir / "model.pt")

        # 2. Save optimizer state
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
        torch.save(optimizer_state, checkpoint_dir / "optimizer.pt")

        # 3. Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f)

        # 4. Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        print(f"Saved checkpoint to {checkpoint_dir}")

    def load(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
    ) -> dict:
        """
        Load a checkpoint.

        If checkpoint_path is None, loads the latest checkpoint.
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None:
            print("No checkpoint found, starting from scratch")
            return {"step": 0, "epoch": 0, "metrics": {}}

        checkpoint_dir = Path(checkpoint_path)

        # 1. Load model weights
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        model_state = torch.load(checkpoint_dir / "model.pt")
        model.load_state_dict(model_state)

        # 2. Load optimizer state
        optimizer_state = torch.load(checkpoint_dir / "optimizer.pt")
        FSDP.optim_state_dict_to_load(model, optimizer, optimizer_state)

        # 3. Load training state
        with open(checkpoint_dir / "training_state.json") as f:
            training_state = json.load(f)

        print(f"Loaded checkpoint from {checkpoint_dir}")
        return training_state

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint."""
        checkpoints = list(self.save_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.name.split("-")[1]))
        return str(checkpoints[-1])

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints."""
        checkpoints = sorted(
            self.save_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )

        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            print(f"Removed old checkpoint: {oldest}")
```

---

# 9. Lessons Learned and Debugging Strategies

## Motivating Questions

- **How do I debug OOM errors that don't give stack traces?** (Memory profiling techniques)
- **Why did training work yesterday but not today?** (Reproducibility and determinism)
- **What should I check first when something breaks?** (Systematic debugging)

## 9.1 The Memory Debugging Toolkit

OOM errors are cryptic. Here's how to diagnose them:

```python
import torch
import gc
from contextlib import contextmanager

@contextmanager
def memory_snapshot(label: str):
    """
    Context manager that tracks memory changes.

    Usage:
        with memory_snapshot("my operation"):
            # do something

    Prints memory delta when block exits.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    before = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()

    try:
        yield
    finally:
        after = torch.cuda.memory_allocated()
        after_reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()

        print(f"\n=== Memory Snapshot: {label} ===")
        print(f"Allocated: {before/1e9:.2f} GB → {after/1e9:.2f} GB "
              f"(Δ {(after-before)/1e9:+.2f} GB)")
        print(f"Reserved:  {before_reserved/1e9:.2f} GB → {after_reserved/1e9:.2f} GB "
              f"(Δ {(after_reserved-before_reserved)/1e9:+.2f} GB)")
        print(f"Peak:      {peak/1e9:.2f} GB")


def memory_summary():
    """
    Print a detailed memory summary.

    Useful for understanding where memory is going.
    """
    print("\n=== GPU Memory Summary ===")
    print(f"Allocated:   {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved:    {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"Max Alloc:   {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"Max Reserved:{torch.cuda.max_memory_reserved()/1e9:.2f} GB")

    # Memory breakdown by tensor
    print("\n=== Largest Tensors ===")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    size_mb = obj.element_size() * obj.nelement() / 1e6
                    if size_mb > 100:  # Only show tensors > 100MB
                        print(f"  {obj.shape} {obj.dtype}: {size_mb:.1f} MB")
        except:
            pass


def find_memory_leak():
    """
    Find tensors that might be leaking.

    Run this at the end of a training step -
    any tensors still alive might be leaks.
    """
    gc.collect()
    torch.cuda.empty_cache()

    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append({
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                    'size_mb': obj.element_size() * obj.nelement() / 1e6,
                    'requires_grad': obj.requires_grad,
                })
        except:
            pass

    # Sort by size
    cuda_tensors.sort(key=lambda x: x['size_mb'], reverse=True)

    print("\n=== CUDA Tensors Still Alive ===")
    for i, t in enumerate(cuda_tensors[:20]):
        print(f"{i+1}. {t['shape']} {t['dtype']}: {t['size_mb']:.1f} MB "
              f"(grad={t['requires_grad']})")
```

**Using the toolkit**:

```python
# During training step
with memory_snapshot("Generation"):
    with sharding_manager:
        responses = rollout.generate_sequences(...)

with memory_snapshot("Log prob computation"):
    log_probs = compute_log_probs(model, responses)

with memory_snapshot("Backward pass"):
    loss.backward()

# At end of step
find_memory_leak()
```

## 9.2 Common Bugs and Their Symptoms

### Bug 1: Gradients Not Flowing

**Symptom**: Loss stays constant, model doesn't learn

**Diagnosis**:
```python
# Check if gradients exist
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient: {name}")
    elif param.grad.abs().max() == 0:
        print(f"Zero gradient: {name}")
```

**Common causes**:
1. `torch.no_grad()` left on
2. Detached tensors in loss computation
3. Frozen parameters (check `requires_grad`)

### Bug 2: Stale Weights in Generation

**Symptom**: Generated text quality doesn't improve despite loss decreasing

**Diagnosis**:
```python
# Compare weights before and after sync
def check_weight_sync(fsdp_model, inference_engine):
    fsdp_state = fsdp_model.state_dict()
    ie_state = inference_engine.model.state_dict()

    for key in fsdp_state:
        if key in ie_state:
            fsdp_tensor = fsdp_state[key].float().cpu()
            ie_tensor = ie_state[key].float().cpu()

            diff = (fsdp_tensor - ie_tensor).abs().max()
            if diff > 1e-5:
                print(f"MISMATCH: {key}, diff={diff}")
```

**Common causes**:
1. ShardingManager not being used
2. Weight sync failing silently
3. Inference engine using cached weights

### Bug 3: KL Divergence Explosion

**Symptom**: KL penalty grows unboundedly, training diverges

**Diagnosis**:
```python
# Monitor KL per step
def diagnose_kl(actor_log_probs, ref_log_probs):
    per_token_kl = actor_log_probs - ref_log_probs

    print(f"KL mean: {per_token_kl.mean():.4f}")
    print(f"KL std:  {per_token_kl.std():.4f}")
    print(f"KL max:  {per_token_kl.max():.4f}")
    print(f"KL min:  {per_token_kl.min():.4f}")

    # Very negative values suggest probability underflow
    if per_token_kl.min() < -10:
        print("WARNING: Very negative KL values - check for underflow")
```

**Common causes**:
1. Learning rate too high
2. KL coefficient too low
3. Reference model not frozen properly

### Bug 4: NaN Loss

**Symptom**: Loss becomes NaN, training crashes

**Diagnosis**:
```python
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Num NaN: {torch.isnan(tensor).sum()}")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        return True
    return False

# In training loop
check_for_nan(log_probs, "log_probs")
check_for_nan(advantages, "advantages")
check_for_nan(loss, "loss")
```

**Common causes**:
1. Log of zero (add epsilon to probabilities)
2. Division by zero in advantage normalization
3. Gradient overflow (use gradient clipping)

## 9.3 Reproducibility Checklist

When debugging, reproducibility is essential:

```python
def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.

    Note: Full reproducibility also requires:
    - CUBLAS_WORKSPACE_CONFIG=:4096:8
    - torch.use_deterministic_algorithms(True)
    - Same hardware, CUDA version, PyTorch version
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility in CUDA ops
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment():
    """Log environment for debugging."""
    print("=== Environment ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    import transformers
    print(f"Transformers: {transformers.__version__}")

    try:
        import vllm
        print(f"vLLM: {vllm.__version__}")
    except:
        print("vLLM: Not installed")
```

## 9.4 Systematic Debugging Workflow

When something breaks, follow this systematic approach:

```
Step 1: ISOLATE THE PROBLEM
├── Does it fail on a minimal example?
├── Does it fail with batch_size=1?
├── Does it fail with HuggingFace generate() instead of vLLM?
├── Does it fail outside of Ray?
└── Which phase fails: generation, log_prob, or training?

Step 2: GATHER INFORMATION
├── What's the exact error message?
├── What's the stack trace (if any)?
├── What are the memory stats before crash?
├── What were the last successful steps?
└── What changed since it last worked?

Step 3: FORM HYPOTHESIS
├── Memory: Is peak memory exceeding capacity?
├── Weights: Are weights synced correctly?
├── Gradients: Are gradients flowing?
├── Numerics: Are there NaN/Inf values?
└── Configuration: Is config valid?

Step 4: TEST HYPOTHESIS
├── Add memory profiling
├── Add tensor shape/dtype logging
├── Test with reduced batch size
├── Test each component in isolation
└── Compare with known working configuration

Step 5: FIX AND VERIFY
├── Make ONE change at a time
├── Verify fix doesn't break other things
├── Document the root cause
└── Add regression test if possible
```

## 9.5 Performance Profiling

Once correctness is verified, optimize performance:

```python
import torch.profiler

def profile_training_step(trainer, batch):
    """
    Profile a training step to find bottlenecks.

    Creates a Chrome trace viewable at chrome://tracing
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,  # Skip first iteration (warmup)
            warmup=1,
            active=3,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            trainer.training_step(batch)
            prof.step()

    # Print summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**Interpreting profiler output**:

| What to Look For | Indicates | Action |
|------------------|-----------|--------|
| Long "aten::mm" | Matrix mult dominates | Normal for transformers |
| Long "aten::copy_" | Memory transfers | Reduce host-device transfers |
| Long "cudaDeviceSynchronize" | Sync overhead | Batch operations |
| Memory increasing over time | Memory leak | Find leaked tensors |
| Low GPU utilization | CPU bottleneck | Profile data loading |

---

# 10. Building From Scratch: A Roadmap

## Motivating Questions

- **If I were starting from zero, what order should I build things?** (Dependencies and milestones)
- **What's the minimum viable RLHF system?** (Core components only)
- **How do I validate each component works before moving on?** (Testing strategy)

## 10.1 Development Phases

### Phase 1: Supervised Fine-Tuning Foundation (Week 1-2)

Build the training infrastructure first, without RL complexity:

```
Deliverables:
├── Data loading pipeline
│   ├── Tokenization
│   ├── Batching and padding
│   └── Attention mask handling
│
├── FSDP training loop
│   ├── Model loading
│   ├── Optimizer setup
│   ├── Forward/backward pass
│   └── Gradient checkpointing
│
├── Checkpointing
│   ├── Save/load model weights
│   ├── Save/load optimizer state
│   └── Resume training
│
└── Metrics and logging
    ├── Loss tracking
    ├── Learning rate scheduling
    └── Validation evaluation

Validation:
- Train on 1000 examples, verify loss decreases
- Save checkpoint, load checkpoint, verify training continues
- Compare loss curve with reference implementation
```

**Why start here?** SFT shares 80% of the infrastructure with RLHF. Get this working first.

### Phase 2: Simple Generation (Week 2-3)

Add generation without inference engine optimization:

```
Deliverables:
├── HuggingFace generate() integration
│   ├── Sampling parameters
│   ├── Batch generation
│   └── Decode to text
│
├── Generation quality testing
│   ├── Manual inspection
│   ├── Perplexity measurement
│   └── Format verification
│
└── Reference model
    ├── Weight sharing with actor
    ├── Eval mode enforcement
    └── No gradient computation

Validation:
- Generate 100 samples, verify they're coherent
- Compute perplexity, verify it matches model quality
- Verify reference model gives consistent outputs
```

### Phase 3: Reward Function (Week 3-4)

Implement and validate your reward function:

```
Deliverables:
├── Reward computation
│   ├── Rule-based rewards (e.g., format checking)
│   ├── Model-based rewards (e.g., classifier)
│   └── Combined reward functions
│
├── Reward statistics
│   ├── Mean/std tracking
│   ├── Distribution visualization
│   └── Outlier detection
│
└── Unit tests
    ├── Known-good examples get high reward
    ├── Known-bad examples get low reward
    └── Edge cases handled

Validation:
- Test on 100 hand-labeled examples
- Verify reward ranking matches human preference
- Verify no obvious failure modes
```

### Phase 4: Basic RLHF Loop (Week 4-6)

Combine everything into a minimal RLHF loop:

```
Deliverables:
├── Log probability computation
│   ├── Efficient implementation
│   ├── Response masking
│   └── Memory management
│
├── Advantage computation
│   ├── GRPO formula
│   ├── Normalization
│   └── Clipping
│
├── Policy gradient loss
│   ├── Loss computation
│   ├── KL penalty
│   └── Entropy bonus (optional)
│
└── Full training loop
    ├── Phase orchestration
    ├── Metrics logging
    └── Debugging tools

Validation:
- Train on 1000 steps, verify reward increases
- Monitor KL divergence stays bounded
- Compare with known working implementation
```

**Milestone**: At this point you have a working RLHF system, just slow.

### Phase 5: Inference Engine Integration (Week 6-8)

Add vLLM/SGLang for fast generation:

```
Deliverables:
├── Inference engine wrapper
│   ├── Engine initialization
│   ├── Weight loading
│   └── Generate interface
│
├── Sharding manager
│   ├── Weight sync
│   ├── Memory management
│   └── Phase transitions
│
├── HybridEngine integration
│   ├── Time-multiplexing
│   ├── Cache management
│   └── Error handling
│
└── Performance benchmarking
    ├── Generation throughput
    ├── Memory utilization
    └── End-to-end step time

Validation:
- Verify generation output matches HF generate()
- Verify weight sync works (check tensor equality)
- Measure speedup (should be 5-10x)
```

### Phase 6: Production Hardening (Week 8-10)

Make it reliable for long training runs:

```
Deliverables:
├── Error handling
│   ├── OOM recovery
│   ├── Timeout handling
│   └── Graceful degradation
│
├── Monitoring
│   ├── Real-time metrics
│   ├── Alerting
│   └── Visualization
│
├── Distributed training
│   ├── Multi-GPU support
│   ├── Multi-node support
│   └── Fault tolerance
│
└── Optimization
    ├── Memory optimization
    ├── Compute optimization
    └── IO optimization

Validation:
- Run for 24 hours without crash
- Inject failures, verify recovery
- Scale to multiple GPUs
```

## 10.2 Testing Strategy

Each component needs verification:

```python
# Test 1: Data Loading
def test_data_loading():
    dataloader = build_dataloader(config)
    batch = next(iter(dataloader))

    assert batch["input_ids"].shape[0] == config.batch_size
    assert batch["input_ids"].dtype == torch.long
    assert (batch["attention_mask"].sum(-1) > 0).all()  # No empty sequences


# Test 2: Forward Pass
def test_forward_pass():
    model = build_model(config)
    batch = get_test_batch()

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    assert output.logits.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(output.logits).any()


# Test 3: Gradient Flow
def test_gradient_flow():
    model = build_model(config)
    optimizer = build_optimizer(model)
    batch = get_test_batch()

    # Forward
    output = model(**batch)
    loss = output.logits.mean()

    # Backward
    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# Test 4: Weight Sync
def test_weight_sync():
    actor = build_fsdp_model()
    inference_engine = build_inference_engine(actor)
    sharding_manager = build_sharding_manager(actor, inference_engine)

    # Modify actor weights
    with torch.no_grad():
        for param in actor.parameters():
            param.add_(1.0)

    # Sync weights
    with sharding_manager:
        # Check inference engine has updated weights
        for (name, actor_param), ie_param in zip(
            actor.named_parameters(),
            inference_engine.model.parameters()
        ):
            diff = (actor_param - ie_param).abs().max()
            assert diff < 1e-5, f"Weight mismatch: {name}, diff={diff}"


# Test 5: Generation Quality
def test_generation_quality():
    model = build_model(config)
    rollout = build_rollout(model)

    prompts = ["Hello, my name is", "The capital of France is"]
    outputs = rollout.generate(prompts, max_new_tokens=50)

    for prompt, output in zip(prompts, outputs):
        # Check generation is coherent (basic check)
        assert len(output) > len(prompt)
        assert output.startswith(prompt)


# Test 6: Reward Consistency
def test_reward_consistency():
    reward_fn = build_reward_fn()

    # Good responses should get high reward
    good_response = "The answer is 42."
    assert reward_fn("What is the answer?", good_response) > 0.5

    # Bad responses should get low reward
    bad_response = "I don't know anything."
    assert reward_fn("What is the answer?", bad_response) < 0.5


# Test 7: Full Training Step
def test_full_training_step():
    trainer = build_trainer(config)
    batch = get_test_batch()

    # Run one step
    metrics = trainer.training_step(batch)

    # Check metrics are reasonable
    assert "loss" in metrics
    assert not math.isnan(metrics["loss"])
    assert metrics["loss"] > 0  # Cross-entropy is positive
```

## 10.3 Reference Architecture

Here's the final architecture you're building toward:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RLHF Training System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Ray Cluster Orchestration                        │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐│ │
│  │  │                        PPOTrainer                                  ││ │
│  │  │  - Training loop coordination                                      ││ │
│  │  │  - Phase sequencing                                                ││ │
│  │  │  - Checkpoint management                                           ││ │
│  │  │  - Metrics aggregation                                             ││ │
│  │  └────────────────────────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         │                          │                          │             │
│         ▼                          ▼                          ▼             │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │   Worker 1  │           │   Worker 2  │           │   Worker N  │       │
│  │  (GPU 0)    │           │  (GPU 1)    │           │  (GPU N-1)  │       │
│  │             │           │             │           │             │       │
│  │  ┌───────┐  │           │  ┌───────┐  │           │  ┌───────┐  │       │
│  │  │ Actor │  │           │  │ Actor │  │           │  │ Actor │  │       │
│  │  │(FSDP) │  │           │  │(FSDP) │  │           │  │(FSDP) │  │       │
│  │  └───┬───┘  │           │  └───┬───┘  │           │  └───┬───┘  │       │
│  │      │      │           │      │      │           │      │      │       │
│  │  ┌───▼───┐  │           │  ┌───▼───┐  │           │  ┌───▼───┐  │       │
│  │  │Shardi-│  │           │  │Shardi-│  │           │  │Shardi-│  │       │
│  │  │  ng   │  │           │  │  ng   │  │           │  │  ng   │  │       │
│  │  │Manager│  │           │  │Manager│  │           │  │Manager│  │       │
│  │  └───┬───┘  │           │  └───┬───┘  │           │  └───┬───┘  │       │
│  │      │      │           │      │      │           │      │      │       │
│  │  ┌───▼───┐  │           │  ┌───▼───┐  │           │  ┌───▼───┐  │       │
│  │  │Rollout│  │           │  │Rollout│  │           │  │Rollout│  │       │
│  │  │(vLLM) │  │           │  │(vLLM) │  │           │  │(vLLM) │  │       │
│  │  └───────┘  │           │  └───────┘  │           │  └───────┘  │       │
│  │             │           │             │           │             │       │
│  │  ┌───────┐  │           │  ┌───────┐  │           │  ┌───────┐  │       │
│  │  │  Ref  │  │           │  │  Ref  │  │           │  │  Ref  │  │       │
│  │  │ Model │  │           │  │ Model │  │           │  │ Model │  │       │
│  │  └───────┘  │           │  └───────┘  │           │  └───────┘  │       │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Shared Components                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │  Data Loader │  │  Reward Fn   │  │  Tokenizer   │                  │ │
│  │  │  (Ray Data)  │  │  (Remote)    │  │  (Shared)    │                  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 10.4 Essential Reading List

Before and during implementation, study these resources:

### Foundational Papers

1. **RLHF Origins**
   - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - OpenAI InstructGPT
   - [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) - RLHF formulation

2. **Algorithms**
   - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO
   - [DeepSeekMath](https://arxiv.org/abs/2402.03300) - GRPO introduction

3. **Infrastructure**
   - [vLLM: Easy, Fast, and Cheap LLM Serving](https://arxiv.org/abs/2309.06180) - PagedAttention
   - [ZeRO: Memory Optimizations](https://arxiv.org/abs/1910.02054) - FSDP foundations

### Code References

1. **veRL** - Production RLHF framework
   - [GitHub](https://github.com/volcengine/verl)
   - Study: Worker architecture, sharding managers, training loop

2. **TRL** - HuggingFace RLHF library
   - [GitHub](https://github.com/huggingface/trl)
   - Study: PPO implementation, reward modeling

3. **DeepSpeed-Chat** - Microsoft RLHF
   - [GitHub](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
   - Study: Hybrid engine, multi-node training

4. **TinyZero** - Minimal GRPO implementation
   - [GitHub](https://github.com/Jiayi-Pan/TinyZero)
   - Study: Minimal GRPO, debugging

### Tutorials and Guides

1. **Spinning Up in Deep RL** - Policy gradient intuition
   - [Website](https://spinningup.openai.com/en/latest/)

2. **PyTorch FSDP Tutorial** - Distributed training
   - [Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

3. **vLLM Documentation** - Inference engine
   - [Docs](https://docs.vllm.ai/)

## 10.5 Final Checklist

Before declaring your RLHF system production-ready:

```
□ CORRECTNESS
  □ Training loss decreases
  □ Reward increases over time
  □ KL divergence stays bounded
  □ Generated text quality improves (manual inspection)
  □ Weights sync correctly (verified via tensor comparison)
  □ Gradients flow through all parameters
  □ No NaN/Inf values during training

□ PERFORMANCE
  □ Generation is 5-10x faster than HF generate()
  □ GPU utilization > 80% during training
  □ No memory leaks over 1000 steps
  □ Step time is consistent (low variance)

□ RELIABILITY
  □ Can resume from checkpoint
  □ Handles OOM gracefully
  □ Handles network failures (for distributed)
  □ Runs for 24+ hours without crash
  □ Metrics are logged and viewable

□ DOCUMENTATION
  □ Setup instructions work on fresh machine
  □ Configuration options documented
  □ Debugging guide for common issues
  □ Performance expectations documented
```

---

# Conclusion

Building an RLHF training system is a challenging but rewarding endeavor. The key insights from this walkthrough:

1. **Training and inference are fundamentally different** - accept this and use specialized systems for each

2. **Memory is the bottleneck** - plan your memory budget carefully, especially for large vocabularies

3. **The HybridEngine pattern works** - time-multiplex GPU memory between training and inference

4. **Weight synchronization is critical** - get this right or your model won't learn

5. **Process architecture matters** - prefer in-process inference engines (vLLM) over subprocess-based ones (SGLang)

6. **Build incrementally** - start with SFT, add generation, add RL, add optimization

7. **Test everything** - OOM errors and silent failures are common; catch them early

The code in this walkthrough represents patterns that work in production. Use them as starting points, but always verify correctness on your specific setup.

Good luck building your RLHF system!

---

## Appendix A: Quick Reference Cards

### A.1 Memory Estimation

```
Model Memory (bf16):       params × 2 bytes
Optimizer Memory (AdamW):  params × 8 bytes
Gradient Memory:           params × 2 bytes
Activation Memory:         batch × seq × hidden × layers × 2 bytes × factor
                          (factor: 0.3 with checkpointing, 4 without)
Logits Memory:             batch × seq × vocab_size × 2 bytes
KV Cache Memory:           2 × batch × seq × hidden × layers × 2 bytes
```

### A.2 Key Config Parameters

```yaml
# Memory-related
batch_size: 8                    # Reduce if OOM during generation
micro_batch_size: 4              # Reduce if OOM during training
gpu_memory_utilization: 0.35     # Reduce if OOM during generation
gradient_checkpointing: true     # Enable if OOM during training

# Performance-related
enforce_eager: true              # Set false for faster generation (if memory allows)
n_samples: 4                     # GRPO samples per prompt

# Stability-related
kl_coef: 0.01                    # Increase if KL explodes
max_grad_norm: 1.0               # Gradient clipping
learning_rate: 1e-6              # Lower for stability
```

### A.3 Debugging Commands

```python
# Memory status
torch.cuda.memory_allocated()     # Active tensors
torch.cuda.memory_reserved()      # Pool reserved
torch.cuda.max_memory_allocated() # Peak active

# Clear memory
torch.cuda.empty_cache()          # Return pool to CUDA
gc.collect()                      # Garbage collect

# Check gradients
for n, p in model.named_parameters():
    print(n, p.grad is not None, p.grad.abs().max() if p.grad is not None else 0)

# Check for NaN
torch.isnan(tensor).any()
torch.isinf(tensor).any()
```

---

*Document Version: 1.0*
*Last Updated: Based on SGLang integration attempt on DGX Spark (GB10)*
*Author: Generated through collaborative debugging and research*
