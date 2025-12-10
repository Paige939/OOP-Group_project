import numpy as np
from base_agent import Agent

class RandomAgent(Agent):
    """
    A random agent that output random actions within the allowed range
    """
    def __init__(self, action: int, max_action: float):
        super().__init__(action, max_action)

    def act(self, observation: np.ndarray)->np.ndarray:
        return np.random.uniform(-self.max_action, self.max_action, size=(self.action,))
    
    def reset(self):
        pass