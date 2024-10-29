import numpy as np
import gymnasium as gym
import time

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

        # 3x3 observation array
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
            ...
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
        """Resets the simulation environment to its starting state."""

        # save agent's position as top left
        self.agent_position = (0, 0)

        # reset environment
        self.env.reset()
        
    def render(self):
        """Renders the simulation window."""
        self.env.render()

    def close(self):
        """Closes the simulation window."""
        self.env.close()
        
if __name__ == "__main__":
    env = PartiallyObservableFrozenLake(render_mode="human")
    env.reset()
    env.render()
    
    for action in [1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2]:
        observation, reward, done = env.step(action)
        print(f"Action: {action}, Observation:\n{np.array(observation)}\n")
        
        time.sleep(2)

    time.sleep(30)
    env.close()
    
