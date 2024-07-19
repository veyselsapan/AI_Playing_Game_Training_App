# src/FSM/fsm_utils.py
from Common.game_env import create_single_env

def get_ball_position(state):
    # Extract ball position from the state (frame)
    for i in range(90, 185):
        for j in range(10, 150):
            if state[i][j][0] == 200 and state[i][j][1] == 72 and state[i][j][2] == 72:
                return i, j
    return None, None

def get_paddle_position(state):
    # Extract paddle position from the state (frame)
    for i in range(189, 194):
        for j in range(10, 150):
            if state[i][j][0] == 200 and state[i][j][1] == 72 and state[i][j][2] == 72:
                return j
    return None

def run_fsm_performance(fsm_file_path, env_name, episodes=5):
    """
    Function to run the Finite State Machine (FSM) agent performance in a specified game environment for a given number of episodes.
    
    Parameters:
    fsm_file_path (str): Path to the FSM agent file.
    env_name (str): Name of the game environment to be used.
    episodes (int): Number of episodes to run for evaluating the FSM agent's performance. Default is 5.

    Returns:
    float: Average score achieved by the FSM agent over the specified number of episodes.
    """
    from FSM.fsm import FSMAgent  # Moved import here to avoid circular dependency
    fsm_agent = FSMAgent.load(fsm_file_path)
    env = create_single_env(env_name)
    total_score = 0
    # Loop through each episode to evaluate the FSM agent
    for episode in range(episodes):
        fsm_agent.ball_released = False
        state = env.reset()
        done = False
        score = 0
        # Loop until the episode is complete
        while not done:
            frame = env.render(mode='rgb_array') # Render the current frame of the environment
            action = fsm_agent.act(frame)
            state, reward, done, info = env.step(action) # Take the action in the environment and observe the results
            score += reward
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

        total_score += score # Accumulate the score of the current episode to the total score
        print(f"Episode {episode + 1}/{episodes} - Score: {score}")

    env.close()
    average_score = total_score / episodes # Calculate the average score over the specified number of episodes
    print(f"Average Score over {episodes} episodes: {average_score}")
    return average_score