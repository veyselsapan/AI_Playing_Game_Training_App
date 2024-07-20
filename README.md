# AI Playing Games Training Interface

## Project Overview

This document provides a detailed technical overview of the AI Playing Games Training Interface application. The application is designed to train, monitor, and evaluate AI agents playing Atari 2600 games using different machine learning models, specifically a custom CNN-based Deep Q-Network (DQN) and a Finite State Machine (FSM).

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [File Structure](#file-structure)
- [Technologies Used](#technologies-used)
- [Contributions](#contributions)
- [License](#license)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/veyselsapan/AI_Playing_Game_Training_App.git
   cd AI_Playing_Game_Training_App
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the application, run the main script:

```bash
python src/main.py
```

The graphical user interface (GUI) will open, allowing you to:

- Train the DQN agent
- Monitor the performance of both DQN and FSM agents
- Compare the results of the two AI approaches

### Training the DQN Agent

1. Select the game environment.
2. Set the training parameters.
3. Choose the directories for saving models and logging data.
4. Click "Start Training".

### Monitoring Agent Performance

- For the DQN agent: Load the trained model and set monitoring parameters.
- For the FSM agent: Load the FSM agent and set monitoring parameters.

## Features

- **DQN Training Section:** Allows users to train the DQN agent with custom parameters.
- **TensorBoard Visualization Section:** Visualizes the training progress.
- **DQN Monitoring Section:** Observes the behavior of the trained DQN agent.
- **FSM Monitoring Section:** Evaluates the FSM agent’s performance.
- **Comparison Functionality:** Compares the performance metrics of the DQN and FSM agents.

## File Structure

The project is organized as follows:

```
AI_Playing_Game_Training_App/
├── data/
│   ├── models/                 # Saved FSM and DQN models
│   ├── performance_logs/       # Logs of model performance metrics
├── docs/
│   ├── user_guide.md           # User guide and documentation
│   ├── technical_docs.md       # Technical documentation for developers
├── src/
│   ├── Common/
│   │   ├── __init__.py         
│   │   ├── game_env.py         # Game environment wrapper
│   ├── DQN/
│   │   ├── __init__.py 
│   │   ├── cnn_architecture.py # Custom CNN architecture for DQN
│   │   ├── dqn.py              # DQN implementation
│   │   ├── training_utils.py   # Training utilities for DQN
│   ├── FSM/
│   │   ├── __init__.py 
│   │   ├── fsm.py              # FSM implementation for Kane
│   │   ├── fsm_utils.py        # Utility functions for FSM
│   ├── __init__.py 
│   ├── gui.py                  # Graphical user interface implementation
│   ├── main.py                 # Main entry point for the application
├── tests/
│   ├── __init__.py 
│   ├── test_fsm.py             # Unit tests for FSM implementation
│   ├── test_dqn.py             # Unit tests for DQN implementation
│   ├── test_gui.py             # Unit tests for GUI
│   ├── test_utils.py           # Unit tests for utility functions
│   ├── test_render.py          # Tests for monitoring agent in the game environment
├── requirements.txt            # List of dependencies
└── README.md                   # Project overview and setup instructions
```

## Technologies Used

- **Python**: Programming language used for implementation.
- **PyTorch**: Deep learning framework for building and training the DQN model.
- **stable-baselines3**: Reinforcement learning library used for DQN implementation.
- **OpenAI Gym**: Toolkit for developing and comparing reinforcement learning algorithms.
- **tkinter**: Python library for creating the GUI.
- **NumPy**: Library for numerical operations.
- **OpenCV**: Library for computer vision tasks.

## Contributions

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or new features. Ensure that your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
