import numpy as np
import gymnasium as gym

class PartiallyObservableFrozenLake(gym.Env):
    def __init__(self, lake_size=8, is_slippery=False, render_mode=None):
        self.lake_size = lake_size
        self.env = gym.make("FrozenLake-v1", map_name=f"{lake_size}x{lake_size}", is_slippery=is_slippery, render_mode=render_mode)

    def reset(self):
        self.env.reset()
        
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        
if __name__ == "__main__":
    env = PartiallyObservableFrozenLake(render_mode="human")
    env.reset()
    env.render()
    
    while True:
        pass

    env.close()
