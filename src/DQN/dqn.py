# src/DQN/dqn.py

import os
import datetime
from DQN.cnn_architecture import CustomCNN
from Common.game_env import create_env, create_eval_env
from DQN.training_utils import setup_eval_callback, create_model, train_model, save_model

def main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count):
    # Policy with custom CNN feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )   

    # Create environment
    env = create_env(environment_name=env_name, n_envs=env_count, n_stack=vectorstacks)
    
    # Create evaluation environment with the same setup
    eval_env = create_eval_env(environment_name=env_name, n_envs=env_count, n_stack=vectorstacks)

    # Load or create model
    model = create_model(env, policy_kwargs=policy_kwargs, log_dir=log_dir)
    
    # Setup evaluation callback
    eval_callback = setup_eval_callback(eval_env, log_dir=log_dir)
    
    # Train the model
    model = train_model(model, total_timesteps=timesteps, eval_callback=eval_callback)
    
    # Save the updated model
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_model_path = os.path.join(save_dir, f'CustomCNN_DQN_{current_time}')
    save_model(model, updated_model_path)

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    env_name = args[0] if len(args) > 0 else 'Breakout-v4'
    save_dir = args[1] if len(args) > 1 else './Training/Saved_Models/'
    log_dir = args[2] if len(args) > 2 else './Training/Logs/'
    timesteps = int(args[3]) if len(args) > 3 else 50000
    vectorstacks = int(args[4]) if len(args) > 4 else 4
    env_count = int(args[5]) if len(args) > 5 else 4
    main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count)
