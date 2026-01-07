# TinyZero Deep Dive Tutorial
## Learning RL for LLMs by Building from Scratch

**Philosophy**: The best way to understand TinyZero is to implement the core algorithms yourself, then see how the production code handles the same problems at scale.

---

## Week 1: From Scratch to TinyZero

### Day 1: Policy Gradients from First Principles

**Morning: Theory & Minimal Implementation (3-4 hours)**

#### Learning Objectives
- Understand why we need policy gradients for LLMs
- Implement REINFORCE from scratch on a toy problem
- Derive the advantage function intuitively

#### Exercises

**Exercise 1.1: REINFORCE on Coin Flip**
Implement a simple REINFORCE agent that learns to predict a biased coin (70% heads).

```python
# Your task: Fill in the TODOs
import torch
import torch.nn as nn

class TinyPolicy(nn.Module):
    """A 1-layer network that outputs probability of 'heads'"""
    def __init__(self):
        super().__init__()
        # TODO: Define a simple linear layer
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
```

**Questions to answer**:
- Why do we multiply log_prob by returns instead of just using returns?
- What happens if you don't discount (gamma=1.0)?
- Why is this algorithm high variance?

---

**Exercise 1.2: Understanding Baselines**
Modify your REINFORCE implementation to subtract a baseline.

```python
def compute_advantages(rewards, values, gamma=0.99):
    """
    Compute advantages using a simple baseline.

    Args:
        rewards: List of rewards
        values: Baseline predictions (e.g., average return)
        gamma: Discount factor

    Returns:
        advantages: returns - baseline
    """
    # TODO: Implement this
    # First compute returns, then subtract baseline
    pass
```

**Questions to answer**:
- Why does subtracting a baseline reduce variance?
- Does it introduce bias into the gradient estimate?
- What's a good choice for the baseline?

---

**Afternoon: Connect to TinyZero (2-3 hours)**

#### Exercise 1.3: Read TinyZero's Advantage Computation

Now open `verl/trainer/ppo/core_algos.py` and study these functions:

1. **`compute_gae_advantage_return`** (line 70-107)
2. **`compute_grpo_outcome_advantage`** (line 111-155)

**Guided Reading Questions**:
```python
# Read the GAE implementation and answer:

# Q1: What are the inputs to compute_gae_advantage_return?
# - token_level_rewards: What shape? What does it represent?
# - values: What are these? Where do they come from?
# - eos_mask: Why do we need this?
# - gamma and lam: What's the difference between these two?

# Q2: Walk through the loop (lines 98-102)
# - Why does it iterate in reverse?
# - What is delta? Why is it called TD error?
# - What is lastgaelam accumulating?
# - Derive: why is this formula correct?

# Q3: GRPO vs GAE (lines 111-155)
# - GRPO doesn't use values. Why not?
# - What does "outcome supervision" mean?
# - How does GRPO normalize advantages? (hint: look at id2mean/id2std)
# - Why group by index when computing mean/std?
```

**Implementation Challenge**:
```python
def my_simple_gae(rewards, values, gamma, lam):
    """
    Your minimal implementation of GAE.

    Try to implement this WITHOUT looking at TinyZero's code first.
    Then compare your solution.

    Hint: GAE(γ, λ) = Σ(γλ)^l * δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    # TODO: Implement GAE from the formula
    pass

# After implementing, compare with TinyZero's version
# - Is your implementation equivalent?
# - Which is more efficient?
# - What edge cases does TinyZero handle that you didn't?
```

---

**Evening: Reflection (30 min)**

Write answers to:
1. In your own words, what is the purpose of GAE's λ parameter?
2. Why does GRPO not need a value function?
3. What are the tradeoffs between PPO (with GAE) and GRPO?

---

### Day 2: PPO from Scratch

**Morning: Clipped Surrogate Objective (3-4 hours)**

#### Learning Objectives
- Understand why we need the PPO clip
- Implement PPO loss from scratch
- Visualize the clipping behavior

#### Exercise 2.1: Implement PPO Policy Loss

```python
def ppo_loss(old_log_probs, new_log_probs, advantages, clip_ratio=0.2):
    """
    Implement the PPO clipped surrogate objective.

    Args:
        old_log_probs: Log probs from behavior policy
        new_log_probs: Log probs from current policy
        advantages: Advantage estimates
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
    pass
```

**Visualization Exercise**:
```python
# TODO: Create a plot showing:
# - X-axis: ratio = π_new / π_old (range from 0 to 3)
# - Y-axis: loss contribution
# - Three lines:
#   1. Unclipped: ratio * A
#   2. Clipped: clamp(ratio, 1-ε, 1+ε) * A
#   3. PPO objective: max of the two
#
# Do this for both positive and negative advantages

import matplotlib.pyplot as plt
import numpy as np

def plot_ppo_objective():
    ratios = np.linspace(0, 3, 100)
    epsilon = 0.2

    # TODO: Implement plotting
    # Case 1: Positive advantage
    # Case 2: Negative advantage
    pass
```

**Questions to answer**:
- When does clipping activate for positive advantages? Negative advantages?
- Why does clipping prevent policy from changing too fast?
- What happens if clip_ratio is too small? Too large?

---

#### Exercise 2.2: Value Function Loss

```python
def value_loss(predicted_values, returns, old_values=None, clip_value=True, clip_range=0.5):
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
    pass
```

**Questions to answer**:
- Why might we want to clip value function updates?
- Is value clipping as important as policy clipping?
- What are the tradeoffs?

---

#### Exercise 2.3: Entropy Bonus

```python
def entropy_from_logits(logits):
    """
    Compute categorical entropy from logits.

    Args:
        logits: Shape (batch, vocab_size)

    Returns:
        entropy: Shape (batch,)

    Formula: H(π) = -Σ π(a) log π(a)
    """
    # TODO: Implement this
    # Hint: Use log_softmax for numerical stability
    pass

def entropy_loss(logits, eos_mask):
    """
    Compute average entropy across valid tokens.

    Args:
        logits: Shape (batch, seq_len, vocab_size)
        eos_mask: Shape (batch, seq_len) - masks out tokens after EOS

    Returns:
        entropy: Scalar
    """
    # TODO: Implement this
    # Need to handle variable-length sequences
    pass
```

**Questions to answer**:
- Why do we want to maximize entropy?
- What happens if entropy bonus is too large? Too small?
- How does this relate to exploration?

---

**Afternoon: Study TinyZero's PPO Implementation (2-3 hours)**

#### Exercise 2.4: Code Reading with Intent

Open `verl/trainer/ppo/core_algos.py` and compare your implementations:

**Part A: Policy Loss** (lines 163-194)
```python
# After reading compute_policy_loss():

# Q1: How does TinyZero handle variable-length sequences?
# - What is eos_mask used for?
# - Why use masked_mean instead of regular mean?

# Q2: What metrics are returned besides loss?
# - pg_clipfrac: What does this tell you?
# - ppo_kl: Why is this useful to track?

# Q3: Compare with your implementation
# - What's different?
# - What did TinyZero handle that you didn't?
# - Is the core logic the same?
```

**Part B: Value Loss** (lines 216-239)
```python
# After reading compute_value_loss():

# Q1: Does TinyZero clip value function by default?
# - Look at the config: ppo_trainer.yaml:119
# - What is cliprange_value set to?

# Q2: Compare with your implementation
# - Same formula?
# - How does masking work?
```

**Part C: Entropy Loss** (lines 197-213)
```python
# After reading compute_entropy_loss():

# Q1: Where is entropy_from_logits defined?
# - Search for it: look in verl/utils/torch_functional.py
# - Is it the same as yours?

# Q2: Why pass entire logits instead of probabilities?
# - Hint: numerical stability
```

---

**Evening: Implement Mini-PPO (2 hours)**

#### Exercise 2.5: Put It All Together

Create a minimal PPO implementation for a simple task (e.g., CartPole or a simple text task).

```python
"""
Mini-PPO: A minimal implementation using what you've learned.

Task: Train a small LM to generate high-reward sequences.
For example: Generate numbers that sum to 10.
"""

# TODO: Implement a complete training loop with:
# 1. Rollout phase (generate sequences)
# 2. Advantage computation (use your GAE)
# 3. PPO update phase (multiple epochs over the batch)
# 4. Logging (loss, clipfrac, entropy, KL)

# Starter code:
import torch
import torch.nn as nn

class TinyLM(nn.Module):
    """A very small language model"""
    def __init__(self, vocab_size=10, hidden_dim=64):
        super().__init__()
        # TODO: Define your architecture
        pass

    def forward(self, input_ids):
        # TODO: Return logits
        pass

def reward_function(sequences):
    """
    Simple reward: +1 if sequence sums to 10, else 0
    sequences: (batch, seq_len) of integers
    """
    # TODO: Implement this
    pass

def train_mini_ppo():
    # TODO: Implement training loop
    # 1. Generate sequences
    # 2. Compute rewards
    # 3. Compute advantages with GAE
    # 4. Update policy with PPO loss
    # 5. Update value function
    # 6. Log metrics
    pass

if __name__ == "__main__":
    train_mini_ppo()
```

**Success criteria**:
- Policy learns to generate sequences summing to ~10
- clipfrac is reasonable (not 0, not 1)
- Entropy decreases over time but doesn't collapse to 0
- You understand every line of code

---

### Day 3: GRPO and the Length Problem

**Morning: Understanding GRPO (3 hours)**

#### Learning Objectives
- Understand how GRPO differs from PPO
- Discover why length normalization matters
- Implement GRPO advantage computation

#### Exercise 3.1: The Length Normalization Problem

**Experiment**:
```python
"""
Discover the length bias problem empirically.
"""

def simple_ppo_reward_computation(log_probs, reward_scalar, kl_penalty):
    """
    Standard PPO: Add up log_probs for the sequence.

    Problem: Longer sequences have more terms in the sum!
    """
    # TODO: Implement this and observe the bias
    pass

def demonstrate_length_bias():
    """
    Generate sequences of different lengths with same per-token reward.
    Show that PPO objective favors longer sequences.
    """
    # TODO:
    # 1. Create sequences of length 5, 10, 20
    # 2. Give each token the same reward
    # 3. Compute PPO objective for each
    # 4. Show that objective is higher for longer sequences
    # 5. Think: Why is this bad?
    pass
```

**Questions**:
- Why does summing over tokens create length bias?
- How would this affect RL training on reasoning tasks?
- What would the model learn to do?

---

#### Exercise 3.2: Implement GRPO Advantage

```python
def grpo_advantages(outcome_rewards, prompt_indices):
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
    # TODO: Implement this
    # Hint: Use a dictionary to group by prompt_index
    pass
```

**Questions**:
- Why sample multiple responses per prompt?
- How does this avoid length bias?
- What if we only have 1 response per prompt? (Check TinyZero line 143-145)

---

**Afternoon: Compare GAE vs GRPO in TinyZero (2-3 hours)**

#### Exercise 3.3: Deep Dive into GRPO Implementation

Read `verl/trainer/ppo/core_algos.py` lines 111-155.

**Guided questions**:
```python
# Q1: Input format
# - Why is input "token_level_rewards" but we only use the sum?
# - What does "outcome supervision" mean?
# - How does the code extract the scalar reward? (line 131-132)

# Q2: The grouping logic (lines 134-150)
# - What is id2score storing?
# - Why use defaultdict(list)?
# - What happens when len(id2score[idx]) == 1? (line 143-145)
#   Why set mean=0, std=1?

# Q3: Broadcasting (line 153)
# - Why .unsqueeze(-1).tile([1, response_length])?
# - Why multiply by eos_mask?
# - Think: What shape is the final advantage?

# Q4: Returns (line 155)
# - Why does GRPO return (scores, scores)?
# - What would GAE return instead?
# - Look at where this function is called - what expects two outputs?
```

**Implementation comparison**:
```python
# TODO: Compare your GRPO implementation with TinyZero's
# - Same algorithm?
# - What edge cases does TinyZero handle?
# - Why the special case for len==1?

# TODO: Implement the token-level version (like TinyZero)
def grpo_token_level_advantages(token_level_rewards, eos_mask, index):
    """
    TinyZero's version that works with token-level tensors.

    Even though GRPO only uses scalar rewards, we need to broadcast
    to token-level for compatibility with the training loop.
    """
    # TODO: Implement this to match TinyZero
    pass
```

---

#### Exercise 3.4: GRPO vs Dr. GRPO

**Reading**: Now read about Dr. GRPO (you mentioned this in your agenda).

Dr. GRPO removes length normalization that GRPO uses. The question is: which normalization?

**Investigation**:
```python
# TODO: Search TinyZero codebase for length normalization
# Hint: grep for "length" in the GRPO-related files

# Questions to answer:
# 1. Does GRPO in TinyZero normalize by length?
# 2. Where would you add/remove length normalization?
# 3. How would you implement Dr. GRPO modification?
```

**Hypothesis formation**:
```
TODO: Before implementing, write down your hypothesis:

1. What will happen to response length if we remove length normalization?
2. Will the model learn to write longer or shorter responses?
3. Why would this affect reasoning performance?

Keep these hypotheses - you'll test them on Day 4!
```

---

### Day 4: Ablation - GRPO vs Dr. GRPO

**Morning: Implement Dr. GRPO (2 hours)**

#### Exercise 4.1: Make the Modification

Based on your Day 3 investigation:

```python
# TODO: Modify TinyZero to implement Dr. GRPO
#
# This might involve:
# 1. Changing advantage computation
# 2. Changing reward computation
# 3. Changing config parameters
#
# Document what you changed and why.
```

**Setup experiment**:
```python
# TODO: Set up two training runs:
# Run 1: Standard GRPO (baseline)
# Run 2: Dr. GRPO (your modification)
#
# Make sure to:
# - Use same random seed
# - Same hyperparameters (except the modification)
# - Same dataset
# - Log to WandB with clear names
```

---

**Afternoon: Run Experiments (4 hours)**

Let the experiments run while you study the literature.

#### Exercise 4.2: Predict the Outcomes

Before looking at results, write down predictions:

```
TODO: Prediction exercise

1. Response length over time:
   - GRPO will: _______________
   - Dr. GRPO will: ___________
   - Because: _________________

2. Reward over time:
   - GRPO will: _______________
   - Dr. GRPO will: ___________
   - Because: _________________

3. KL divergence:
   - GRPO will: _______________
   - Dr. GRPO will: ___________
   - Because: _________________

4. Final performance:
   - Which will perform better: ___________
   - Why: _____________________________
```

---

**Evening: Analyze Results (2 hours)**

#### Exercise 4.3: Scientific Analysis

```python
# TODO: Create analysis notebook

import wandb
import matplotlib.pyplot as plt

# 1. Load runs from WandB
grpo_run = wandb.Api().run("...")
dr_grpo_run = wandb.Api().run("...")

# 2. Plot comparisons:
#    - Reward vs. step (with error bars if multiple seeds)
#    - Response length vs. step
#    - KL divergence vs. step
#    - Entropy vs. step

# 3. Statistical tests:
#    - Is the difference in final reward significant?
#    - Is the difference in response length significant?

# 4. Qualitative analysis:
#    - Sample 10 responses from each model
#    - Do you see different reasoning patterns?
#    - Which looks more like "genuine" reasoning?
```

**Write-up**:
```
TODO: Write 1-page analysis

Title: GRPO vs Dr. GRPO on [Your Task]

1. Hypothesis (what you predicted)
2. Methods (what you changed)
3. Results (with plots)
4. Discussion:
   - Were your predictions correct?
   - What surprised you?
   - What did you learn about length bias?
   - Implications for RL training
```

---

### Day 5-6: Literature Deep Dive

**Goal**: Read criticism papers with newfound context from your implementations.

#### Exercise 5.1: Targeted Reading

Now that you've implemented everything, read these papers with specific questions:

**Paper 1: "Understanding R1-Zero-Like Training: A Critical Perspective"**
- TODO: Find and read this paper
- Focus questions:
  - What do they say about length bias?
  - How does it relate to what you observed?
  - What solutions do they propose?
  - Would they work with your implementation?

**Paper 2: DAPO Paper**
- TODO: Find and read
- Focus questions:
  - What modifications to GRPO do they propose?
  - How do these compare to Dr. GRPO?
  - Could you implement their approach in TinyZero?

**Paper 3: Sebastian Raschka's "State of RL for LLM Reasoning"**
- TODO: Find and read
- Focus questions:
  - What are the current open problems?
  - Which ones did you encounter in your experiments?
  - What's the state of the art?

**Paper 4: Nathan Lambert's "Recent reasoning research: GRPO tweaks"**
- TODO: Find and read
- Focus questions:
  - What tweaks are people trying?
  - Which seem most promising?
  - Could you implement them as Day 7 experiments?

---

#### Exercise 5.2: Create Concept Map

```
TODO: Draw a concept map connecting:

- PPO ←→ GRPO ←→ Dr. GRPO
- Value function ←→ GAE ←→ Advantages
- Length bias ←→ Normalization ←→ Outcome rewards
- Exploration ←→ Entropy ←→ KL divergence
- Policy clipping ←→ Trust region ←→ Stability

Add notes from papers to each connection.
```

---

### Day 7: Synthesis and Next Steps

**Morning: Write Summary (2 hours)**

#### Exercise 7.1: "Three Things I Didn't Understand Until I Implemented It"

Write a blog post / document covering:

```markdown
# Three Things I Didn't Understand About GRPO Until I Implemented It

## Thing 1: [e.g., Why GAE's λ parameter matters]
- What I thought before: ...
- What I learned: ...
- The aha moment: ...
- Code snippet that made it click: ...

## Thing 2: [e.g., The subtle difference between GRPO and PPO advantages]
- What I thought before: ...
- What I learned: ...
- The aha moment: ...
- Code snippet that made it click: ...

## Thing 3: [e.g., Why length bias is so pernicious]
- What I thought before: ...
- What I learned: ...
- The aha moment: ...
- Code snippet that made it click: ...

## Bonus insights:
- ...
```

---

**Afternoon: Future Experiments (2 hours)**

#### Exercise 7.2: Design Next Experiments

Based on everything you've learned, design 3 experiments you'd want to run:

```
Experiment 1: [Name]
- Hypothesis: ...
- Modification to code: ...
- Expected outcome: ...
- Why it matters: ...

Experiment 2: [Name]
- Hypothesis: ...
- Modification to code: ...
- Expected outcome: ...
- Why it matters: ...

Experiment 3: [Name]
- Hypothesis: ...
- Modification to code: ...
- Expected outcome: ...
- Why it matters: ...
```

---

## Success Criteria for Week 1

By the end of the week, you should be able to:

- [ ] Implement GAE from memory (without looking at code)
- [ ] Explain PPO clipping to someone else (with a diagram you draw)
- [ ] Describe the exact difference between GRPO and PPO advantages
- [ ] Explain why length bias happens and how to fix it
- [ ] Have run at least one successful ablation experiment
- [ ] Critique current RL approaches with specific examples
- [ ] Know where to look in TinyZero's codebase for any RL concept

---

## Compute Requirements

- **Days 1-2**: CPU only (toy implementations)
- **Day 3**: 1 GPU for testing (a few hours)
- **Day 4**: 1-2 GPUs for ablations (4-8 hours)
- **Days 5-7**: No compute needed

---

## Tips for Success

1. **Don't skip the from-scratch implementations** - they build intuition
2. **Always predict before you observe** - makes learning stick
3. **Compare your code with TinyZero's** - learn production patterns
4. **Write down your confusions** - they often lead to insights
5. **If stuck for >30min** - ask for hints (but not solutions!)
6. **Teach what you learn** - write it down as if explaining to someone

---

## When You Get Stuck

**Level 1 Hint**: Review the math/theory
**Level 2 Hint**: Look at similar code in TinyZero
**Level 3 Hint**: Ask for pseudocode
**Level 4 Hint**: Ask for code with TODOs
**Level 5**: "Cut to the chase!" - get full solution

Remember: Struggling is part of learning. The confusion you feel means you're learning something real.

---

## Next Steps (Week 2+)

After Week 1, you'll be ready for:
- Implementing process reward models (PRM)
- Scaling experiments to larger models
- Trying novel algorithmic modifications
- Contributing to TinyZero or veRL
- Designing your own RL approaches

But first - master the fundamentals. Good luck!
