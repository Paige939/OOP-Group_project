from manage import PendulumEnvWrapper, Experiment
from Agents import RandomAgent, CEM_Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    episode_len = 200
    n_episodes = 5

    # --- CEM Agent ---
    # create environmemt for CEM
    cem_env_wrapper = PendulumEnvWrapper(render_mode=None) 
    cem_agent = CEM_Agent(action=cem_env_wrapper.action, max_action=cem_env_wrapper.max_action,
                          num_samples=300, elite_frac=0.2, save_path="CEM_weights.npy")
    exper_cem = Experiment(cem_env_wrapper, cem_agent, episode_len)
    cem_rewards = []
    print("=== CEM Agent ===")
    for ep in range(n_episodes):
        reward = exper_cem.run_episode(render=False) #The render mode can be changed from "human", "rgb_array", None
        cem_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")

    # --- Random Agent ---
    #create environment for Random
    rand_env_wrapper = PendulumEnvWrapper(render_mode=None) 
    random_agent = RandomAgent(action=rand_env_wrapper.action, max_action=rand_env_wrapper.max_action)
    exper_rand = Experiment(rand_env_wrapper, random_agent, episode_len)
    rand_rewards = []
    print("\n=== Random Agent ===")
    for ep in range(n_episodes):
        reward = exper_rand.run_episode(render=False)
        rand_rewards.append(reward)
        print(f"Episode {ep+1} total reward: {reward:.2f}")

    cem_env_wrapper.close()
    rand_env_wrapper.close() # close all environments

    # --- Plot comparison ---
    plt.figure(figsize=(8,5))
    plt.plot(range(1,n_episodes+1), cem_rewards, 'o-', label='CEM Agent')
    plt.plot(range(1,n_episodes+1), rand_rewards, 's-', label='Random Agent')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("CEM vs Random Agent on Pendulum-v1")
    plt.legend()
    plt.grid(True)
    plt.savefig("Comparison.png")
    plt.show()
    
