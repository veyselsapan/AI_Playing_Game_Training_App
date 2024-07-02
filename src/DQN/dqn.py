# src/DQN/dqn.py

import os
import datetime
from DQN.cnn_architecture import CustomCNN
from Common.game_env import create_env
from DQN.training_utils import setup_eval_callback, create_model, train_model, save_model

def main():
    # Environment and model paths
    environment_name = 'Breakout-v4'
    log_dir = '/home/veysel/University_of_London/CM3070_FP/data/performance_logs/logs'
    save_dir = '/home/veysel/University_of_London/CM3070_FP/data/training/Saved_Models/'
    
    # Policy with custom CNN feature extractor
    policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    )   
    # Create environment
    env = create_env(environment_name=environment_name)

    # Load or create model
    model = create_model(env, policy_kwargs=policy_kwargs)
    
    # Setup evaluation callback
    eval_callback = setup_eval_callback(env, log_dir=log_dir)
    
    # Train the model
    model = train_model(model, total_timesteps=500, eval_callback=eval_callback)
    
    # Save the updated model
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_model_path = os.path.join(save_dir, f'CustomCNN_DQN_{current_time}')
    save_model(model, updated_model_path)

if __name__ == '__main__':
    main()
