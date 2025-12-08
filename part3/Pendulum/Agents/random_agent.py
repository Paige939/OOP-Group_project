import numpy as np
from base_agent import Agent

class RandomAgent(Agent):
    """
    A random agent that output random actions within the allowed range
    """
    def act(self, observation: np.ndarray)->np.ndarray:
        return np.random.uniform(-self.max_action, self.max_action, size=self.action)
    
    def reset(self):
        pass