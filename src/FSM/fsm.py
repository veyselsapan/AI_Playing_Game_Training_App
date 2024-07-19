# src/FSM/fsm.py
import pickle
import numpy as np
import random
from FSM.fsm_utils import get_ball_position, get_paddle_position

class FSMAgent:
    def __init__(self):
        self.ball_released = False # Initialize the ball_released flag to False

    def act(self, frame):
        """
        Determine the action to take based on the current frame of the game.
        Parameters:
        frame (numpy array): The current frame of the game.
        Returns:
        list: The action to take.
        """
        ball_y, ball_x = get_ball_position(frame) # Get the position of the ball
        paddle_x = get_paddle_position(frame) # Get the position of the paddle

        if not self.ball_released:
            action = 1  # Fire (Release the ball)
            self.ball_released = True
        elif paddle_x is not None and ball_y is not None:
            if ball_x < paddle_x:
                action = 3  # Move left
            elif ball_x > paddle_x:
                action = 2  # Move right
            else:
                action = 0  # Stay still
        else:
            action = random.choice([0, 1, 2, 3]) # Randomly choose an action

        return [action] 

    def save(self, filepath):
        """
        Save the FSM agent to a file.
        Parameters:
        filepath (str): The path to the file where the FSM agent will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Load the FSM agent from a file.
        Parameters:
        filepath (str): The path to the file from which the FSM agent will be loaded.
        Returns:
        FSMAgent: The loaded FSM agent.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

