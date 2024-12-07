import numpy as np
import gymnasium as gym
import time
import random

from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class PartiallyObservableFrozenLake(gym.Env):
    """This class implements a "partially observable" frozen lake.
    That is, it implements a frozen lake simulation where the agent
    can only "see" the 3x3 grid of cells at which it is centered. """
    
    def __init__(self, lake_size=8, desc=None, is_slippery=False, render_mode=None, custom_reward=False):
        """Initializes the simulation environment.

        Args:
            lake_size (int): The height and width of the square frozen lake simulation environment.
            is_slippery (bool): Whether to use stochastic movements (True) or not (False).
            render_mode (string): For our purposes, either "human" (render a visible window) or None (don't render a window).
        """

        # save arguments for later
        self.lake_size = lake_size
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        self.custom_reward = custom_reward
        
        self.agent_position = (0, 0)
        self.goal_state = (lake_size-1, lake_size-1)
        self.n_steps = 0

        # initialize the simulation environment using the arguments
        self.env = gym.make("FrozenLake-v1", desc=desc, map_name=None, is_slippery=is_slippery, render_mode=render_mode)

    def initialize_random_layout(self):
        """Initializes frozen lake environment with random layout."""
        random_map = generate_random_map(size=self.lake_size, p=0.9)
        self.env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=self.is_slippery, render_mode=self.render_mode)

    def isCloser(self, current_position, prev_position):
        goal_x, goal_y = self.goal_state
        curr_x, curr_y = self.agent_position
        prev_x, prev_y = prev_position

        distance_to_goal = abs(goal_x - curr_x) + abs(goal_y - curr_y)
        prev_distance_to_goal = abs(goal_x - prev_x) + abs(goal_y - prev_y)

        return distance_to_goal < prev_distance_to_goal

    def customize_reward(self, prev_position, reward, done):
        """Augments Frozen Lake's default reward (1 for goal, 0 for everything else). This custom reward
        returns 1 if the goal is reached, -1 if the agent falls in a hole, a distance-based pentalty otherwise.

        Args:
            reward (int): The reward returned by default in Frozen Lake (either 0 or 1).
            done (boolean): True if the agent has fallen in a hole, False otherwise.
            
        Returns:
            float: The augmented reward.
        """
        
        # goal state
        if reward == 1:
            new_reward = 10

        # hole states
        elif reward == 0 and done:
            new_reward = -5

        # if the agent gets closer to the goal
        elif self.isCloser(self.agent_position, prev_position):
            # use manhattan distance to encourage movement towards goal
            goal_x, goal_y = self.goal_state
            curr_x, curr_y = self.agent_position

            distance_to_goal = abs(goal_x - curr_x) + abs(goal_y - curr_y)
            new_reward = 1 / (distance_to_goal + 0.1)

        # the agent gets farther away from the goal
        else:
            new_reward = -0.1 * self.n_steps
                
        return new_reward           
               
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
                    elif state == b'G': row.append(2)  # 2 for goal (treasure)
                    else: row.append(0)                # 0 otherwise

                # if this location is out-of-bounds, append None to the row
                else:
                    row.append(-1)

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
        self.n_steps += 1
        
        # update agent position based on state
        prev_position = self.agent_position
        self.agent_position = divmod(state, self.lake_size)

        # modify reward if custom_reward is True
        if self.custom_reward:
            reward = self.customize_reward(prev_position, reward, done)
            
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

        # reset number of steps
        self.n_steps = 0

        return self.get_observation()
        
    def render(self):
        """Renders the simulation window."""
        self.env.render()

    def close(self):
        """Closes the simulation window."""
        self.env.close()

if __name__ == "__main__":
    """Main function for playing around with the environment. Not really used for anything in the project."""
    
    env = PartiallyObservableFrozenLake(render_mode="human", custom_reward=True)
    for lo in range(10):
        env.initialize_random_layout()
        env.reset()

        done = False
        
        while not done:
            action = random.randint(0,3)
            obs, reward, done = env.step(action)
            env.render()

            print(env.agent_position, reward)
        
    env.close()


    
