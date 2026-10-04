from policy import Policy
from grid_world import GridWorld
from torch.distributions import Categorical
import torch

def main():

    # initialize 6x6 grid environment
    grid_world = GridWorld()

    # initialize policy
    policy = Policy()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # start train epoch
    episodes = 1
    for episode in range(episodes):
        terminated = False
        G = []
        log_probs = []
        state = grid_world.reset() # reset state to (0,0)
        max_steps = 100
        optimizer.zero_grad()

        for step in range(max_steps):
            # run state through policy to get action (1x4 tensor where B=batch_size)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_logits = policy(state_tensor / 5.0) # normalized

            # sample action based on distribution of actions from policy
            dist = Categorical(logits=action_logits)
            action = dist.sample()

            # this is the log(pi(a_t|s_t)) that we need for the loss function
            log_prob = dist.log_prob(action) 
            log_probs.append(log_prob)

            # step once in environment using action
            next_state, reward, terminated = grid_world.step_forward(action.item())
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
        gamma = 0.9
        for i in range(len(G) - 2, -1, -1):
            G[i] += gamma * G[i+1]

        loss = 0
        # we want loss = -sum_{t=0}^{len(G)-1} (gamma**t G[t] * log_probs[t])
        # where log_probs[t] = log(pi(a_t|s_t)), the log prob of our policy choosing action a_t under state s_t
        for t in range(len(G)):
            loss -= gamma ** t * G[t] * log_probs[t]

        loss.backward()
        optimizer.step()






if __name__ == "__main__":
    main()