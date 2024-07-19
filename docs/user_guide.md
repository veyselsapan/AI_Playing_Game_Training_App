# User Guide for Breakout AI Training Interface

## Introduction

This user guide provides step-by-step instructions on how to use the AI Training Interface application. The application enables users to train, monitor, and evaluate AI agents playing the game Breakout using different machine learning models, specifically a custom CNN-based Deep Q-Network (DQN) and a Finite State Machine (FSM).

## Launching the Application

To start the application, run the following command in your terminal:

```bash
python src/main.py
```

The main window of the application will appear, providing access to various functionalities through different sections.

## DQN Training

### Steps to Train a DQN Model

1. **Choose Game Environment:**
   - In the "Train DQN Model" section, select the desired game environment from the dropdown menu.

2. **Specify Training Parameters:**
   - Enter the number of timesteps in the "Timesteps" field.
   - Enter the number of vector stacks in the "Vector Stacks" field.
   - Enter the number of environments in the "Number of Environments" field.

3. **Select Directories:**
   - Click the "Choose Save Directory" button to select the directory where the trained model will be saved.
   - Click the "Choose Log Directory" button to select the directory where TensorBoard logs will be stored.

4. **Train Model:**
   - Click the "Train Model" button to start training the DQN model. Training progress can be monitored through the terminal output.

## TensorBoard Visualization

### Steps to View DQN Training Results

1. **Choose Log Directory:**
   - In the "View DQN Training Results" section, click the "Choose Log Directory" button to select the directory containing the TensorBoard logs.

2. **Show TensorBoard:**
   - Click the "Show TensorBoard" button to launch TensorBoard. The output will display the URL to access TensorBoard in your web browser.

## Monitoring DQN Agents

### Steps to Monitor a DQN Agent

1. **Choose Model File:**
   - In the "Monitor DQN Agent" section, click the "Choose Model File" button to select the trained DQN model file.

2. **Specify Monitoring Parameters:**
   - Select the game environment from the dropdown menu.
   - Enter the number of environments in the "Number of Environments" field.
   - Enter the number of vector stacks in the "Vector Stacks" field.
   - Enter the recording time in the "Recording Time (seconds)" field.

3. **Monitor Agent:**
   - Click the "Monitor Agent" button to start monitoring the DQN agent. The performance will be recorded as a video.
   - Choose the save location for the video when prompted.

## Monitoring FSM Agents

### Steps to Monitor an FSM Agent

1. **Select FSM Python File:**
   - In the "FSM Agent" section, click the "Select Python FSM File" button to select the Python file containing the FSM agent code.

2. **Save FSM as Pickle:**
   - Click the "Save FSM as Pickle" button to save the FSM agent as a pickle file.

3. **Select Pickle FSM File:**
   - Click the "Select Pickle FSM File" button to select the saved pickle FSM file.

4. **Specify Monitoring Parameters:**
   - Select the game environment from the dropdown menu.
   - Enter the number of environments in the "Number of Environments" field.
   - Enter the recording time in the "Recording Time (seconds)" field.

5. **Monitor FSM:**
   - Click the "Monitor FSM" button to start monitoring the FSM agent. The performance will be recorded as a video.
   - Choose the save location for the video when prompted.

6. **Measure FSM Performance:**
   - Click the "Measure FSM Performance" button to evaluate the FSM agent's performance over multiple episodes. The results will be displayed in the application.

## Troubleshooting

### Common Issues and Solutions

- **Model File Not Selected:**
  - Ensure that a model file is selected before attempting to monitor the DQN agent.
- **Log Directory Not Selected:**
  - Ensure that a log directory is selected before attempting to run TensorBoard.
- **Python FSM File Not Selected:**
  - Ensure that a Python FSM file is selected before saving the FSM agent as a pickle file.
- **Pickle FSM File Not Selected:**
  - Ensure that a pickle FSM file is selected before attempting to monitor or measure the FSM agent's performance.

## Conclusion

This user guide provides detailed instructions on how to use the Breakout AI Training Interface application. The application enables users to train, monitor, and evaluate AI agents playing the game Breakout using different machine learning models. For further assistance or information, please refer to the technical documentation provided in the `docs/technical_docs.md` file.
