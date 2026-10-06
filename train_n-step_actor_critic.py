import torch
from grid_world import GridWorld
from pathlib import Path
from policy import Policy
from critic import Critic
from evaluate import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from visualization.learning_curve import plot_learning_curve
from visualization.grid_world import print_policy, plot_success_heatmap


def main():

    seed = 1
    torch.manual_seed(seed)
    output_dir = Path(__file__).resolve().parent / "outputs" / "n-step_actor_critic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # initialize grid environment
    mine_locations = {(3, 3), (6, 7), (8, 2), (3, 9)}
    grid_world = GridWorld(mines=mine_locations, seed=seed)
    evaluation_world = GridWorld(mines=mine_locations, seed=seed)

    # initialize policy and critic
    policy = Policy()
    critic = Critic()
    
    all_params = list(policy.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
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
    n = 5 # rollout length
    for episode in range(episodes):
        terminated = False
        state = grid_world.reset() # reset state randomly

        rewards = []
        log_probs = []
        values = []
        

        gamma = 0.99

        for step in range(max_steps):
            optimizer.zero_grad()

            # run state through policy to get action (1x4 tensor where B=batch_size)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_logits = policy(state_tensor / float(grid_world.x_bound)) # normalized

            # sample action based on distribution of actions from policy
            dist = Categorical(logits=action_logits)
            action = dist.sample()

            # this is the log(pi(a_t|s_t)) that we need for the loss function
            log_prob = dist.log_prob(action) 

            # step once in environment using action
            next_state, reward, terminated = grid_world.step_forward(action.item())

            # keep track of the following for weight update at end of rollout
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(critic(state_tensor / float(grid_world.x_bound)))

            # step forward
            environment_steps += 1
            state = next_state

            # update weights after a n-length rollout
            if len(rewards) == n or terminated or step + 1 == max_steps:
                actor_loss = 0.0
                critic_loss = 0.0

                with torch.no_grad():
                    if terminated:
                        G = 0.0
                    else:
                        final_state_tensor = torch.tensor(state, dtype=torch.float32)
                        G = critic(final_state_tensor / float(grid_world.x_bound))
                
                for t in range(len(rewards) - 1, -1, -1):
                    # G_t = r_t + gamma * G_{t+1} 
                    G = rewards[t] + gamma * G

                    # Advantage_t = Q_t - V(s_t)   [here, G is technically Q_hat because we add V(s_n) above]
                    td_error = G - values[t]

                    critic_loss += 0.5 * td_error**2
                    actor_loss -= td_error.detach() * log_probs[t] # negative because we want to maximize reward

                # gradient update
                loss = (critic_loss + actor_loss) / len(rewards)
                loss.backward()
                optimizer.step()

                rewards.clear()
                log_probs.clear()
                values.clear()


            if terminated:
                break


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