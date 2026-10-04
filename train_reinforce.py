from policy import Policy
from grid_world import GridWorld
from evaluate import evaluate_policy
from torch.distributions import Categorical
import torch
import numpy as np
from pathlib import Path
from visualization.grid_world import print_policy, plot_success_heatmap
from visualization.learning_curve import plot_learning_curve
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using: {device}")
# if device.type == "cuda":
#     print(torch.cuda.get_device_name(0))

def main():
    seed = 1
    torch.manual_seed(seed)
    output_dir = Path(__file__).resolve().parent / "outputs" / "reinforce"
    output_dir.mkdir(parents=True, exist_ok=True)

    # initialize grid environment
    mine_locations = {(3, 3), (6, 7), (8, 2), (3, 9)}
    grid_world = GridWorld(mines=mine_locations, seed=seed)
    evaluation_world = GridWorld(mines=mine_locations, seed=seed)

    # initialize policy
    policy = Policy()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    log_freq = 100
    max_steps = 200
    eval_every_steps = 5000
    next_eval_step = eval_every_steps
    environment_steps = 0

    # evaluate before training, then every 'eval_every_steps' training interactions
    # every evaluation checkpoint uses 20 sampled rollouts from each eligible cell.
    success_rates = evaluate_policy(policy, evaluation_world, max_steps=max_steps, seed=seed)
    learning_history = [(0, float(np.nanmean(success_rates)))]

    num_successes = 0
    num_failures = 0
    num_timeouts = 0
    steps_per_success = []

    # start train epoch
    episodes = 5000
    for episode in range(episodes):
        terminated = False
        G = []
        log_probs = []
        state = grid_world.reset() # reset state randomly
        optimizer.zero_grad()

        for step in range(max_steps):
            # run state through policy to get action (1x4 tensor where B=batch_size)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_logits = policy(state_tensor / (1.0 * grid_world.x_bound)) # normalized

            # sample action based on distribution of actions from policy
            dist = Categorical(logits=action_logits)
            action = dist.sample()

            # this is the log(pi(a_t|s_t)) that we need for the loss function
            log_prob = dist.log_prob(action) 
            log_probs.append(log_prob)

            # step once in environment using action
            next_state, reward, terminated = grid_world.step_forward(action.item())
            environment_steps += 1
            state = next_state

            # r = environment reward from action a at state t
            G.append(reward)
            # we want G_i = r_i + gamma*G_{i+1}
            # currently, we are just setting G_i = r_i

            if terminated:
                break
        
        # calculate all of the G_i's
        # G[-1] contains the final reward r
        # G[i-1] = r[i-1] + gamma * G[i]
        # since we already assigned each G[k] to r[k] in the forward pass, 
        # we just need to iterate backwards and add the gamma * G[k+1] term
        gamma = 0.99
        for i in range(len(G) - 2, -1, -1):
            G[i] += gamma * G[i+1]

        loss = 0
        # we want loss = -sum_{t=0}^{len(G)-1} (gamma**t G[t] * log_probs[t])
        # where log_probs[t] = log(pi(a_t|s_t)), the log prob of our policy choosing action a_t under state s_t
        for t in range(len(G)):
            loss -= gamma ** t * G[t] * log_probs[t]
        # gamma ** t * 

        loss.backward()
        optimizer.step()


        # tallying metrics
        if grid_world.at_goal():
            num_successes += 1
            steps_per_success.append(step + 1)
        elif terminated:
            num_failures += 1
        else:
            num_timeouts += 1
        
        if (episode + 1) % log_freq == 0:
            goal_percentage = 100.0 * num_successes / log_freq
            mine_percentage = 100.0 * num_failures / log_freq
            timeout_percentage = 100.0 * num_timeouts / log_freq
            success_steps = np.mean(steps_per_success)
            print(f"Episodes {episode - 99}-{episode + 1} | Goal: {goal_percentage}% | Mine: {mine_percentage}% | Timeout: {timeout_percentage}% | Success steps: {success_steps}")
            num_successes = 0
            num_failures = 0
            num_timeouts = 0
            steps_per_success = []

        if environment_steps >= next_eval_step or episode == episodes - 1:
            success_rates = evaluate_policy(
                policy, evaluation_world, max_steps=max_steps, seed=seed
            )
            success_rate = float(np.nanmean(success_rates))
            learning_history.append((environment_steps, success_rate))
            print(f"Evaluation at {environment_steps:,} steps | Goal: {success_rate:.1f}%")
            next_eval_step += eval_every_steps

    # Save the two README figures and the values behind the learning curve.
    fig, ax = plot_learning_curve(learning_history)
    fig.savefig(output_dir / "learning_curve.png", dpi=200)
    fig, ax = plot_success_heatmap(success_rates, evaluation_world)
    fig.savefig(output_dir / "success_heatmap.png", dpi=200)
    np.savetxt(
        output_dir / "learning_curve.csv", learning_history, delimiter=",",
        header="training_environment_steps,evaluation_success_percent",
        comments="", fmt=["%d", "%.4f"],
    )

    # Plot policy results
    fig, ax = print_policy(policy, grid_world)
    fig.savefig(output_dir / "policy_final.png", dpi=200)
    print(f"Saved figures to {output_dir}")
    plt.show()



if __name__ == "__main__":
    main()
