import numpy as np
from base_agent import Agent
import os

class CEM_Agent(Agent):
    """
    Cross Entropy Method for Pendulum-v1
    Policy: linear model(action=w^T s)
    """
    def __init__(self, action: int, max_action: float, num_samples: int, elite_frac: float, save_path="CEM_weights.npy"):
        super().__init__(action, max_action)
        #Sample number
        self.num_samples=num_samples
        #Elite percentage
        self.elite_frac=elite_frac
        #Elite number
        self.num_elite=max(1, int(self.num_samples*self.elite_frac))
        #Initialization of distribution params
        self.weights_mean=None
        self.weights_std=None
        self.num_features=None
        self.best_weight=None
        self.save_path=save_path

    def pre_episode(self, env, episode_len):
        """
        Train the agent if no saved weights exist, otherwise load from file
        """
        if hasattr(env,'state'):
            obs_dim=env.state
        else:
            obs_dim=3

        self.num_features=obs_dim+1
        if os.path.exists(self.save_path):
            self.best_weights=np.load(self.save_path)
            print(f"Loaded trained weights from {self.save_path}")
            return
        
        #Initialize mean and standard
        self.weights_mean=np.zeros((self.action, self.num_features))
        self.weights_std=np.ones((self.action, self.num_features))*10.0

        #Start training
        n_iters=40
        for iter in range(n_iters):
            samples=np.random.randn(self.num_samples, self.action, self.num_features)*self.weights_std+self.weights_mean
            rewards=np.zeros(self.num_samples)
            for i in range(self.num_samples):
                rewards[i]=self.evaluate(env, samples[i], episode_len)
            elite_index=rewards.argsort()[-self.num_elite:]
            elite_weights=samples[elite_index]
            self.weights_mean=elite_weights.mean(axis=0)
            self.weights_std=elite_weights.std(axis=0)+1e-6
            print(f"Iteration {iter+1} over {n_iters}, elite reward mean: {rewards[elite_index].mean():.2f}")

        self.best_weights=self.weights_mean.copy()
        np.save(self.save_path, self.best_weights)
        print(f"Training completed. Weights saved to {self.save_path}")

    def evaluate(self, env, weights, episode_len):
        """
        Run an episode using giving weights
        """
        obs=env.reset()
        total_reward=0.0
        for _ in range(episode_len):
            obs_feat=np.append(obs, 1.0)
            action=np.dot(weights, obs_feat)
            action=np.clip(action, -self.max_action, self.max_action)
            next_obs, reward, done, _ =env.step(action)
            total_reward+=reward
            obs=next_obs
            if done:
                break
        return total_reward

    def reset(self):
        pass

    def act(self, observation: np.ndarray)->np.ndarray:
        obs_feat=np.append(observation, 1.0)
        action=np.dot(self.best_weights, obs_feat)
        #Clip by max_action
        action=np.clip(action, -self.max_action, self.max_action)
        return action
    
   