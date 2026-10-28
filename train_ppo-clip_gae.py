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
    output_dir = Path(__file__).resolve().parent / "outputs" / "ppo-clip_gae"
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
    episodes = 10000
    n = 5 # rollout length
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    update_epochs = 4 # for each rollout, we run 4 epochs, each updating the weights once


    for episode in range(episodes):
        terminated = False
        state = grid_world.reset() # reset state randomly

        states = []
        actions = []
        rewards = []
        old_log_probs = []
        old_values = []
        
        for step in range(max_steps):
            # data collection
            state_tensor = torch.tensor(state, dtype=torch.float32) / float(grid_world.x_bound)

            # no grad because we are just collecting rollout data and we dont need to backprop through
            with torch.no_grad():
                # run state through policy to get action (1x4 tensor where B=batch_size)
                action_logits = policy(state_tensor)

                # sample action based on distribution of actions from policy
                dist = Categorical(logits=action_logits)
                action = dist.sample()

                # this is the log(pi(a_t|s_t)) that we need for the loss function
                old_log_prob = dist.log_prob(action) 

                old_value = critic(state_tensor)

            # step once in environment using action
            next_state, reward, terminated = grid_world.step_forward(action.item())

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(old_log_prob)
            old_values.append(old_value)

            # step forward
            environment_steps += 1
            state = next_state

            # run ppo after n-length rollout
            rollout_length = len(rewards)
            if rollout_length == n or terminated or step + 1 == max_steps:
                with torch.no_grad():
                    if terminated:
                        next_value = 0.0
                    else:
                        final_state_tensor = torch.tensor(state, dtype=torch.float32) / float(grid_world.x_bound)
                        next_value = critic(final_state_tensor)

                advantages = [0.0] * rollout_length
                targets = [0.0] * rollout_length

                advantage = 0.0
                for t in reversed(range(rollout_length)):
                    current_value = old_values[t].item()
                    td_error = rewards[t] + gamma * next_value - current_value

                    advantage = td_error + gae_lambda * gamma * advantage
                    advantages[t] = advantage
                    targets[t] = advantage + current_value

                    next_value = current_value


                # gradient updates with ppo-clip
                for epoch in range(update_epochs):
                    actor_loss = 0.0
                    critic_loss = 0.0

                    for t in reversed(range(rollout_length)):
                        state_tensor = states[t]
                        action_logits = policy(state_tensor)

                        dist = Categorical(logits=action_logits)
                        new_log_prob = dist.log_prob(actions[t])
                        current_value = critic(state_tensor)

                        # ratio = pi_new(a_t|s_t) / pi_old(a_t|s_t)
                        ratio = torch.exp(new_log_prob - old_log_probs[t])

                        new_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

                        actor_loss -= torch.minimum(ratio * advantages[t], new_ratio * advantages[t])

                        critic_loss += 0.5 * (targets[t] - current_value) ** 2

                    # gradient update
                    loss = (critic_loss + actor_loss) / rollout_length
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                states.clear()
                actions.clear()
                rewards.clear()
                old_log_probs.clear()
                old_values.clear()


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