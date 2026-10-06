import matplotlib.pyplot as plt


def plot_learning_curve(history):
    """Plot (training environment steps, evaluation success percentage)."""
    steps, success_rates = zip(*history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, success_rates, color="tab:blue", linewidth=2)
    ax.set_title("Learning curve")
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Evaluation success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(left=0)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax
