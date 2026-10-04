import numpy as np
import torch


@torch.no_grad()
def evaluate_policy(policy, env, episodes_per_start=20, max_steps=200, seed=1):
    """Estimate success from each starting cell without updating the policy.

    Pass a separate evaluation environment. The local random generator keeps
    evaluation from consuming the random numbers used during training.
    """
    rng = np.random.default_rng(seed)
    device = next(policy.parameters()).device
    states = env.start_positions
    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)

    # The policy is frozen, so compute its probabilities once for each cell.
    was_training = policy.training
    policy.eval()
    try:
        probabilities = torch.softmax(
            policy(state_tensor / float(env.x_bound)), dim=-1
        ).cpu().numpy().astype(float)
    finally:
        policy.train(was_training)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities_by_state = dict(zip(states, probabilities))

    # Index by [y, x] to match the plot. Terminal cells remain blank.
    success_rates = np.full((env.y_bound + 1, env.x_bound + 1), np.nan)
    for start in states:
        successes = 0
        for _ in range(episodes_per_start):
            state = env.reset(start=start)
            for _ in range(max_steps):
                action = int(rng.choice(4, p=probabilities_by_state[state]))
                state, _, terminated = env.step_forward(action)
                if terminated:
                    successes += int(env.at_goal())
                    break

        x, y = start
        success_rates[y, x] = 100.0 * successes / episodes_per_start

    return success_rates
