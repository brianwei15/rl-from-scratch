

def main():

    # initialize environment

    # start train epoch
    episodes = 1
    for episode in range(episodes):
        G = []
        t = 0
        # initialize state
        while True:
            # run state through policy to get action
            # step once in environment using state and action
            # calculate reward based on curr state and prev action
            # r = environment reward from action a at state t
            G.append(reward)
            # we want G_i = r_i + gamma*G_{i+1}
            # currently, G_i = r_i


            t += 1





if __name__ == "__main__":
    main()