# test/test_render.py
import sys
import os
import gym
from stable_baselines3 import DQN
import torch
import time

# Add the project root directory to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Ensure the same environment as used in the main code
from src.Common.game_env import create_env
from src.DQN.training_utils import load_model


def main():
    env_name = 'Breakout-v4'
    # Path to model file
    model_path = '/home/veysel/University_of_London/stable_baselines3_DQN_Agent_Application/data/models/DoubleDQN_CustomCNN_Breakout_Updated_small_buffer_size_CPU.zip' 

    # Create environment
    env = create_env(env_name, n_envs=1, n_stack=4)
    
  
    # Load or create model
    model = load_model(env=env, model_path=model_path)

    obs = env.reset()
    print("Monitoring agent started.")
    while True:
        action, _states = model.predict(obs, deterministic=True)  # Use deterministic=True for evaluation
        print("Action taken:", action)
        obs, rewards, dones, info = env.step(action)
        env.render(mode='human')  # Explicitly use 'human' mode for rendering
        time.sleep(0.1)  # Add delay to slow down the rendering
        if True in dones:
            print("Episode finished.")
            break
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
