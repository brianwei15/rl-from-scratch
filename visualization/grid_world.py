import torch
import matplotlib.pyplot as plt
import numpy as np

@torch.no_grad()
def print_policy(policy, env):
    width = env.x_bound + 1
    height = env.y_bound + 1
    device = next(policy.parameters()).device

    # run inference on all width*height states at once. 
    # (width*height) x 2 -> policyNN -> (width*height) x 4
    states = [(x, y) for x in range(width) for y in range(height)]
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)

    logits = policy(states_tensor / (1.0 * env.x_bound)) # run inference on normalized i/p
    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()


    # visualize plot
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    fig, ax = plt.subplots(figsize=(8, 8))

    for (x, y), action_probs in zip(states, probabilities):
        if (x, y) == env.goal_pos:
            ax.text(x, y, "GOAL", ha="center", va="center", color="green", weight="bold")
            continue

        if (x, y) in env.mines:
            ax.text(x, y, "MINE", ha="center", va="center", color="red", weight="bold")
            continue

        for probability, (dx, dy) in zip(action_probs, directions):
            ax.annotate(
                "",
                xy=(x + 0.38 * dx, y + 0.38 * dy),
                xytext=(x, y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "tab:blue",
                    "alpha": probability,
                    "linewidth": 0.5 + 3.0 * probability,
                    "mutation_scale": 12,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    ax.set_xticks(range(width))
    ax.set_yticks(range(height))

    # cell boundaries lie halfway between integer coordinates.
    ax.set_xticks([i - 0.5 for i in range(width + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(height + 1)], minor=True)
    ax.grid(which="minor", color="lightgray")

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Policy: darker, thicker arrows = higher probability")

    fig.tight_layout()
    return fig, ax


def plot_success_heatmap(success_rates, env):
    """Show the frozen policy's sampled success rate from each starting cell."""
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    cells = ax.imshow(
        np.ma.masked_invalid(success_rates), origin="lower",
        vmin=0, vmax=100, cmap=cmap,
    )
    for x, y in env.mines:
        ax.text(x, y, "MINE", ha="center", va="center", fontsize=8)
    x, y = env.goal_pos
    ax.text(x, y, "GOAL", ha="center", va="center", fontsize=8)

    ax.set_xticks(range(env.x_bound + 1))
    ax.set_yticks(range(env.y_bound + 1))
    ax.set_xticks([i - 0.5 for i in range(env.x_bound + 2)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(env.y_bound + 2)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("REINFORCE success by starting cell")
    fig.colorbar(cells, ax=ax, label="Success rate (%)", ticks=[0, 25, 50, 75, 100])
    fig.tight_layout()
    return fig, ax
