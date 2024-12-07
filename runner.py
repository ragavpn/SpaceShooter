import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_isolate import MultiAgentSpaceShooterEnv
import psutil, csv, time, threading
from datetime import date

# Track resource utilization

process = psutil.Process(os.getpid()) # Current process

util_start_time = time.time()
util_file = open(f"./util_{time.strftime('%Y-%m-%dT%H:%M:%S')}.csv", 'w')
util_writer = csv.writer(util_file)
util_writer.writerow(["Time Elapsed (runner.py)",
                      f"CPU% (max {psutil.cpu_freq().max:.4f} MHz)",
                      f"Mem% (tot. {psutil.virtual_memory().total/((2**10)**3):.4f} GB)",
                      "Comment"])

def util_write(comment=""):
    util_writer.writerow([time.time() - util_start_time,
                          process.cpu_percent(),
                          process.memory_percent(),
                          comment])
    util_file.flush()

def util_write_sched():
    util_write()
    threading.Timer(1, util_write_sched).start()
util_write_sched()

# Define env

def lr_schedule(progress_remaining):
    return 1e-4 * progress_remaining

num_agents = 3
num_enemies = 5
reward_params = {
    'hit_reward': 5, # reward given to an agent when it successfully hits an enemy.
    'survival_bonus': 1,  #  reward given to an agent for surviving each step in the environment.
    'miss_penalty': -1,  # penalty applied when an agent shoots and misses an enemy 
    'enemy_pass_penalty': -2, # penalty applied when an enemy reaches the bottom of the screen, indicating it has passed the agents.
    'health_loss': 0.2  # amount of health lost by each agent when an enemy reaches the bottom of the screen.
}
hyperparams = {
    'agent_speed': 0.1,  # speed of agents moving left and right
    'enemy_speed': 0.01,  # speed of enemies moving towards the bottom
    'end_on_enemy_pass': True # whether the episode ends when an enemy crosses the bottom
}

env = DummyVecEnv([lambda: MultiAgentSpaceShooterEnv(num_agents=num_agents, num_enemies=num_enemies, reward_params=reward_params, hyperparams=hyperparams)])

util_write("env created")

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_file = "ppo_multiagent_space_shooter_env.zip"

if os.path.exists(model_file):
    print("Loading the saved model...")
    util_write("beginning model load")
    model = PPO.load(model_file, env=env, device=device, weights_only=True)
    util_write("finished model load")
else:
    print("No saved model found. Creating a new model...")
    util_write("beginning model creation")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=lr_schedule, n_steps=2048, batch_size=64, device=device)
    model.learn(total_timesteps=10000)
    model.save(model_file)
    util_write("finished model creation")


for episode in range(100):
    util_write(f"episode {episode}")
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        env.render(mode='None')
    
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")