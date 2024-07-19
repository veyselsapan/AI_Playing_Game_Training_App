# Technical Documentation for AI Playing Games Training Interface

## Introduction

This document provides a detailed technical overview of the AI Training Interface application. The application is designed to train, monitor, and evaluate AI agents playing Atari 2600 game using different machine learning models, specifically a custom CNN-based Deep Q-Network (DQN) and a Finite State Machine (FSM).

## Application Structure

The application is structured into several key components:

1. **Graphical User Interface (GUI)**
2. **DQN Training**
3. **DQN Monitoring**
4. **FSM Monitoring**
5. **TensorBoard Visualization**

### 1. Graphical User Interface (GUI)

The GUI is implemented using the Tkinter library and is designed to facilitate user interaction with the application. The GUI components include sections for training DQN agents, monitoring DQN agents, monitoring FSM agents, and visualizing training results with TensorBoard.

#### Key Files

- **src/gui.py**: Contains the implementation of the Tkinter-based GUI.

#### GUI Components

- **DQN Training Section**
  - Allows users to select the game environment, specify training parameters (timesteps, vector stacks, number of environments), choose directories for saving models and logging tensorboard data, and initiate the training process.
- **DQN Monitoring Section**
  - Allows users to load a trained DQN model, specify monitoring parameters (game environment, vector stacks, number of environments, recording time), and monitor the agent's performance in the game environment, saving the performance as a video.
- **FSM Monitoring Section**
  - Allows users to select and load FSM agents, specify the game environment, and monitor the FSM agent's performance in the game environment, saving the performance as a video. Additionally, users can measure the FSM agent's performance over multiple episodes.
- **TensorBoard Visualization Section**
  - Allows users to select the log directory and visualize the training results using TensorBoard by clicking the "Show TensorBoard" button.

### 2. DQN Training

The DQN training process involves training a custom CNN-based DQN model to play the game. The training parameters such as timesteps, vector stacks, and number of environments are specified by the user through the GUI.

#### Key Files

- **src/DQN/dqn.py**: Contains the main function for initiating the DQN training process.
- **src/DQN/training_utils.py**: Provides utility functions for setting up evaluation callbacks, loading models, creating models, training models, and saving models.
- **src/DQN/cnn_architecture.py**: Defines the custom CNN architecture used by the DQN model.

#### Training Process

1. **Environment Creation**: The game environment is created with the specified parameters.
2. **Model Creation**: A new DQN model with the custom CNN architecture is created.
3. **Evaluation Callback Setup**: An evaluation callback is set up to evaluate the model during training.
4. **Model Training**: The model is trained for the specified number of timesteps.
5. **Model Saving**: The trained model is saved to the specified directory.

### 3. DQN Monitoring

The DQN monitoring process involves loading a trained DQN model and monitoring its performance in the game environment. The monitoring parameters such as game environment, vector stacks, number of environments, and recording time are specified by the user through the GUI.

#### Key Files

- **src/gui.py**: Contains the GUI implementation for monitoring DQN agents.
- **src/Common/game_env.py**: Provides functions for creating and configuring the game environment, and for recording the agent's performance as a video.

#### Monitoring Process

1. **Model Loading**: The trained DQN model is loaded from the specified file.
2. **Environment Creation**: The game environment is created with the specified parameters.
3. **Agent Monitoring**: The agent's performance is monitored in the game environment, and the performance is recorded as a video.

### 4. FSM Monitoring

The FSM monitoring process involves loading an FSM agent and monitoring its performance in the game environment. The monitoring parameters such as game environment and recording time are specified by the user through the GUI.

#### Key Files

- **src/gui.py**: Contains the GUI implementation for monitoring FSM agents.
- **src/FSM/fsm.py**: Contains the implementation of the FSM agent.
- **src/FSM/fsm_utils.py**: Provides utility functions for FSM agents, including functions for running FSM agents and measuring their performance.
- **src/Common/game_env.py**: Provides functions for creating and configuring the game environment, and for recording the agent's performance as a video.

#### Monitoring Process

1. **FSM Agent Loading**: The FSM agent is loaded from the specified file.
2. **Environment Creation**: The game environment is created with the specified parameters.
3. **Agent Monitoring**: The agent's performance is monitored in the game environment, and the performance is recorded as a video.

### 5. TensorBoard Visualization

The TensorBoard visualization functionality allows users to visualize the training results. The user can choose the log directory and start TensorBoard to visualize the training metrics.

#### Key Files

- **src/gui.py**: Contains the GUI implementation for TensorBoard visualization.

#### Visualization Process

1. **Log Directory Selection**: The user selects the directory where the TensorBoard log files are stored.
2. **TensorBoard Launch**: By clicking the "Show TensorBoard" button, TensorBoard is launched in a new process, allowing the user to visualize the training results.

### 6. Common Utilities

Common utility functions are provided to support the creation and configuration of the game environment, logging performance metrics, and handling file operations.

#### Key Files

- **src/Common/game_env.py**: Provides functions for creating and configuring the game environment.

## Technical Details

### Environment Creation

The game environment is created using the Stable Baselines3 library. The `make_atari_env` function is used to create the environment, and the `VecFrameStack` wrapper is used to stack frames for input to the DQN model.

### DQN Model

The DQN model is created using the Stable Baselines3 library with a custom CNN architecture. The custom CNN architecture is defined in the `CustomCNN` class.

### Model Training

The DQN model is trained using the `learn` method provided by the Stable Baselines3 library. An evaluation callback is used to evaluate the model during training.

### Model Saving

The trained model is saved using the `save` method provided by the Stable Baselines3 library.

### Agent Monitoring

The agent's performance is monitored in the game environment and recorded as a video using OpenCV.

## Conclusion

This document has provided a detailed technical overview of the AI Playing Games Training Interface application. The application enables users to train, monitor, and evaluate AI agents playing the Atari 2600 games using different machine learning models. The key components of the application have been described, along with the technical details of the environment creation, model training, model saving, agent monitoring, and TensorBoard visualization processes.

For further information or assistance, please refer to the user guide provided in the `docs/user_guide.md` file.