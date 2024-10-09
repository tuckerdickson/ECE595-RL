import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def run(episodes):

    for _ in range(episodes):

        env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8), map_name="8x8", is_slippery=False, render_mode="human")

        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            
            action = env.action_space.sample()
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state

    env.close()
                   
if __name__ == '__main__':
    run(1000)
