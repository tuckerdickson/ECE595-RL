import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt

class PartiallyObservableFrozenLake(gym.Env):
    """This class implements a "partially observable" frozen lake.
    That is, it implements a frozen lake simulation where the agent
    can only "see" the 3x3 grid of cells at which it is centered. """
    
    def __init__(self, lake_size=8, is_slippery=False, render_mode=None):
        """Initializes the simulation environment.

        Args:
            lake_size (int): The height and width of the square frozen lake simulation environment.
            is_slippery (bool): Whether to use stochastic movements (True) or not (False).
            render_mode (string): For our purposes, either "human" (render a visible window) or None (don't render a window).
        """

        # initialize the simulation environment using the arguments
        self.env = gym.make("FrozenLake-v1", map_name=f"{lake_size}x{lake_size}", is_slippery=is_slippery, render_mode=render_mode)

        # save arguments for later
        self.lake_size = lake_size
        self.render_mode = render_mode
        self.agent_position = (0, 0)

    def get_observation(self):
        """Observes the 3x3 grid surrounding the agent and returns it.

        Returns:
            [[int]]: a 3x3 array representing the observed state of the environment surrounding the agent.
        """

        # get the agent's current position
        x, y = self.agent_position

        # 3x3 observation 
        observation = []

        # go from -1 to 1 (3 rows)
        for i in range(-1, 2):
            row = []

            # go from -1 to 1 (3 columns)
            for j in range(-1, 2):

                # compute the x and y offsets
                nx, ny = x+i, y+j

                # make sure the offsets are within the bounds of the frozen lake
                if (0 <= nx < self.lake_size) and (0 <= ny < self.lake_size):
                    # get the type of tile at this particular location (either hole, goal, or ice)
                    state = self.env.unwrapped.desc[nx][ny]

                    # append to the current row
                    if state == b'H': row.append(1)    # 1 for hole
                    elif state == b'G': row.append(2)  # 2 for goal
                    else: row.append(0)                # 0 otherwise

                # if this location is out-of-bounds, append None to the row
                else:
                    row.append(None)

            # append the row to the observation
            observation.append(row)

        return observation
        
    def step(self, action):
        """Advances the agent one "step" (outcome given an action), returning the new state, reward, and termination status.

        Args:
            action (int): 0-left, 1-down, 2-right, 3-up

        Returns:
            [[int]]: a 3x3 array representing the observed state of the environment surrounding the agent.
            float: the reward for the action taken
            done: whether or not the episode terminates as a result of the action (i.e., the agent falls in a hole)
        """

        # take the step
        state, reward, done, _, _ = self.env.step(action)

        # update agent position based on state
        self.agent_position = divmod(state, self.lake_size)
    
        # render window with the new state, if applicable
        if self.render_mode is not None:
            self.render()

        return self.get_observation(), reward, done
        
    def reset(self):
        """Resets the simulation environment to its starting state.

        Returns:
            [[int]]: a 3x3 array representing the observed state of the environment surrounding the agent.
        """

        # save agent's position as top left
        self.agent_position = (0, 0)

        # reset environment
        self.env.reset()
        return self.get_observation()
        
    def render(self):
        """Renders the simulation window."""
        self.env.render()

    def close(self):
        """Closes the simulation window."""
        self.env.close()

def run(episodes, max_steps_per_episode=100, is_training=True, render=False):
    env = PartiallyObservableFrozenLake(render_mode='human' if render else None)
    initial_observation = env.reset()

    q = np.zeros((env.env.observation_space.n, env.env.action_space.n))  

    learning_rate_a = 0.9  # Alpha or learning rate
    discount_factor_g = 0.9  # Gamma or discount rate
    epsilon = 1  # 100% random actions initially
    epsilon_decay_rate = 0.0001  # Epsilon decay rate
    rng = np.random.default_rng()  # Random number generator

    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)  

    for i in range(episodes):
        print(f"Episode {i+1}/{episodes}") 
        observation = env.reset()
        state = env.env.unwrapped.s  
        terminated = False
        total_reward = 0
        steps = 0

        while not terminated and steps < max_steps_per_episode:
           # print(f"Observation at step {steps}:\n{np.array(observation)}\n")
            if is_training and rng.random() < epsilon:
                action = env.env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_observation, reward, terminated = env.step(action)

            new_state = env.env.unwrapped.s  

            if is_training:
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state
            observation = new_observation
            total_reward += reward
            steps += 1

        #print(f"Observation at final step {steps}:\n{np.array(observation)}")
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = total_reward
        steps_per_episode[i] = steps  
        print(f"Reward for Episode {i+1}: {total_reward}")

    print("\nQ-table:")
    print("Actions: [Left, Down, Right, Up]")
    for state in range(q.shape[0]):
        print(f"State {state}: {q[state]}")  

    env.close()

   
    sum_rewards = np.zeros(episodes)
    avg_steps = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
        avg_steps[t] = np.mean(steps_per_episode[max(0, t - 100):(t + 1)])  

    plt.figure()
    plt.plot(sum_rewards)
    plt.title("Sum of Rewards over 100 Episodes vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of Rewards in the last 100 Episodes")
    plt.savefig('frozen_lake_partial_observation_rewards.png')

    plt.figure()
    plt.plot(avg_steps)
    plt.title("Average Steps over 100 Episodes vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average Steps in the last 100 Episodes")
    plt.savefig('frozen_lake_partial_observation_steps.png')


if __name__ == '__main__':
    run(episodes=10000, max_steps_per_episode=100, is_training=True, render=False)
