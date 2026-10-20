import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make("CartPole-v1", render_mode="rgb_array")

try:
    play(
        env,
        keys_to_action={"a": 0, "d": 1},
        noop=0,
        fps=25,
    )
finally:
    env.close()