import numpy as np
import gymnasium as gym
import time

from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class PartiallyObservableFrozenLake(gym.Env):
    """This class implements a "partially observable" frozen lake.
    That is, it implements a frozen lake simulation where the agent
    can only "see" the 3x3 grid of cells at which it is centered. """
    
    def __init__(self, lake_size=8, is_slippery=False, render_mode=None, randomize=False):
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
        self.randomize = randomize
        
        self.agent_position = (0, 0)
        self.goal_state = (lake_size-1, lake_size-1)

        # initialize the simulation environment using the arguments
        if randomize:
            self.initialize_random_layout()
        else:
            self.env = gym.make("FrozenLake-v1", map_name=f"{lake_size}x{lake_size}", is_slippery=is_slippery, render_mode=render_mode)

    def initialize_random_layout(self):
        """Initializes frozen lake environment with random layout. NOTE: randomize must be True to use this function!"""

        # ensure randomize is True
        assert self.randomize, "Cannot initialize random layout in environment with randomize=False!"

        # create random map and initialize environment using the map
        random_map = generate_random_map(size=self.lake_size, p=0.8)
        self.env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=self.is_slippery, render_mode=self.render_mode)
        
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

if __name__ == "__main__":
    """Main function for playing around with the environment. Not really used for anything in the project."""
    
    env = PartiallyObservableFrozenLake(render_mode="human", randomize=True)

    for i in range(5):
        env.reset()
        env.render()
        time.sleep(5)
        env.initialize_random_layout()
    
    env.close()


    
