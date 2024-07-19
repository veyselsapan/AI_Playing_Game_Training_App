import sys
import os
import time

# Add the project root directory to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

# Ensure the same environment as used in the main code
from src.Common.game_env import create_single_env
from src.FSM.fsm import FSMAgent

def main():
    env_name = 'Breakout-v4'
    fsm_model_path = '/home/veysel/University_of_London/stable_baselines3_DQN_Agent_Application/data/models/FSM.pkl'  # Path to FSM model file

    # Create environment
    env = create_single_env(env_name)

    # Load FSM model
    fsm_agent = FSMAgent.load(fsm_model_path)
    counter = 0

    obs = env.reset()
    print("Monitoring FSM agent started.")
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        action = fsm_agent.act(frame)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')  # Explicitly use 'human' mode for rendering
        time.sleep(0.1)  # Add delay to slow down the rendering
        if done:
            counter += 1
            print("Episode finished.")
            obs = env.reset()
            done = False
            if counter > 10:
                break
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
