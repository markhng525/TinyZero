"""
PPO Clipping Visualization

Shows how the PPO clipped surrogate objective behaves for different
probability ratios and advantage signs.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_ppo_objective(clip_ratio: float = 0.2, save_path: str = "ppo_clipping.png"):
    """
    Plot the PPO objective showing clipping behavior.

    Creates a 2-panel figure:
    - Left: Positive advantage (A = +1)
    - Right: Negative advantage (A = -1)
    """
    ratios = np.linspace(0.0, 2.5, 500)
    epsilon = clip_ratio

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Left panel: Positive Advantage (A > 0) ===
    A_pos = 1.0

    unclipped_pos = ratios * A_pos
    clipped_ratios = np.clip(ratios, 1 - epsilon, 1 + epsilon)
    clipped_pos = clipped_ratios * A_pos
    ppo_objective_pos = np.minimum(unclipped_pos, clipped_pos)

    ax1 = axes[0]
    ax1.plot(ratios, unclipped_pos, 'b--', label='Unclipped: r × A', linewidth=2)
    ax1.plot(ratios, clipped_pos, 'r--', label=f'Clipped: clip(r, {1-epsilon}, {1+epsilon}) × A', linewidth=2)
    ax1.plot(ratios, ppo_objective_pos, 'g-', label='PPO: min(unclipped, clipped)', linewidth=3)

    # Highlight clipping regions
    ax1.axvspan(0, 1 - epsilon, alpha=0.1, color='orange', label='Clip region (r < 1-ε)')
    ax1.axvspan(1 + epsilon, 2.5, alpha=0.1, color='orange')
    ax1.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='r = 1 (no change)')
    ax1.axvline(x=1 - epsilon, color='orange', linestyle=':', alpha=0.5)
    ax1.axvline(x=1 + epsilon, color='orange', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Probability Ratio r = π_new / π_old', fontsize=12)
    ax1.set_ylabel('Objective Value', fontsize=12)
    ax1.set_title('Positive Advantage (A = +1)\n"Good action - want to increase probability"', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.grid(True, alpha=0.3)

    # Annotations
    ax1.annotate('Clipped!\nPrevents over-optimization',
                xy=(1.8, 1.2), xytext=(2.0, 1.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate('Not clipped\nGradient corrects mistake',
                xy=(0.4, 0.4), xytext=(0.2, 1.2),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))

    # === Right panel: Negative Advantage (A < 0) ===
    A_neg = -1.0

    unclipped_neg = ratios * A_neg
    clipped_neg = clipped_ratios * A_neg
    ppo_objective_neg = np.minimum(unclipped_neg, clipped_neg)

    ax2 = axes[1]
    ax2.plot(ratios, unclipped_neg, 'b--', label='Unclipped: r × A', linewidth=2)
    ax2.plot(ratios, clipped_neg, 'r--', label=f'Clipped: clip(r, {1-epsilon}, {1+epsilon}) × A', linewidth=2)
    ax2.plot(ratios, ppo_objective_neg, 'g-', label='PPO: min(unclipped, clipped)', linewidth=3)

    # Highlight clipping regions
    ax2.axvspan(0, 1 - epsilon, alpha=0.1, color='orange')
    ax2.axvspan(1 + epsilon, 2.5, alpha=0.1, color='orange', label='Clip region')
    ax2.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='r = 1 (no change)')
    ax2.axvline(x=1 - epsilon, color='orange', linestyle=':', alpha=0.5)
    ax2.axvline(x=1 + epsilon, color='orange', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Probability Ratio r = π_new / π_old', fontsize=12)
    ax2.set_ylabel('Objective Value', fontsize=12)
    ax2.set_title('Negative Advantage (A = -1)\n"Bad action - want to decrease probability"', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(-2.5, 0.5)
    ax2.grid(True, alpha=0.3)

    # Annotations
    ax2.annotate('Clipped!\nPrevents over-punishment',
                xy=(0.4, -0.8), xytext=(0.6, -0.2),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate('Not clipped\nGradient corrects mistake',
                xy=(1.8, -1.8), xytext=(2.0, -0.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved PPO clipping visualization to: {save_path}")
    return save_path


def plot_ppo_loss_surface(clip_ratio: float = 0.2, save_path: str = "ppo_loss_surface.png"):
    """
    Plot PPO loss as a 2D surface over (ratio, advantage) space.
    """
    ratios = np.linspace(0.1, 2.5, 100)
    advantages = np.linspace(-2, 2, 100)
    R, A = np.meshgrid(ratios, advantages)

    epsilon = clip_ratio
    clipped_R = np.clip(R, 1 - epsilon, 1 + epsilon)

    unclipped = R * A
    clipped = clipped_R * A
    ppo_objective = np.minimum(unclipped, clipped)

    # Loss is negative of objective (we minimize)
    ppo_loss = -ppo_objective

    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(R, A, ppo_loss, levels=50, cmap='RdYlGn_r')
    plt.colorbar(contour, ax=ax, label='PPO Loss (to minimize)')

    # Mark clip boundaries
    ax.axvline(x=1 - epsilon, color='white', linestyle='--', linewidth=2, label=f'r = {1-epsilon}')
    ax.axvline(x=1 + epsilon, color='white', linestyle='--', linewidth=2, label=f'r = {1+epsilon}')
    ax.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=1.0, color='white', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Probability Ratio r = π_new / π_old', fontsize=12)
    ax.set_ylabel('Advantage A', fontsize=12)
    ax.set_title('PPO Loss Surface\n(Green = low loss = good, Red = high loss = bad)', fontsize=12)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved PPO loss surface to: {save_path}")
    return save_path


if __name__ == "__main__":
    plot_ppo_objective()
    plot_ppo_loss_surface()
    print("\nVisualization complete! Open the PNG files to view.")
