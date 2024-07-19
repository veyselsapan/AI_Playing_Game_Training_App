# tests/test_fsm.py

import unittest
import numpy as np
from src.FSM.fsm import FSMAgent
from src.FSM.fsm_utils import get_ball_position, get_paddle_position
import sys
import os

# Add the project root directory to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))
class TestFSMAgent(unittest.TestCase):

    def setUp(self):
        """
        Set up the FSM agent for testing.
        """
        self.agent = FSMAgent()

    def test_initial_action(self):
        """
        Test the initial action of the FSM agent (release the ball).
        """
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        action = self.agent.act(frame)
        self.assertEqual(action, [1], "Initial action should be to release the ball")

    def test_action_with_positions(self):
        """
        Test the actions of the FSM agent when ball and paddle positions are provided.
        """
        # Create a frame with ball and paddle positions
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[100][50] = [200, 72, 72]  # Ball position
        frame[190][80] = [200, 72, 72]  # Paddle position
        self.agent.ball_released = True
        # Test moving left
        action = self.agent.act(frame)
        self.assertEqual(action, [3], "Agent should move left to reach the ball")
        
        # Test moving right
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[100][90] = [200, 72, 72]  # Ball position
        frame[190][80] = [200, 72, 72]  # Paddle position
        action = self.agent.act(frame)
        self.assertEqual(action, [2], "Agent should move right to reach the ball")
        
        # Test staying still
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[100][80] = [200, 72, 72]  # Ball position
        frame[190][80] = [200, 72, 72]  # Paddle position
        action = self.agent.act(frame)
        self.assertEqual(action, [0], "Agent should stay still when aligned with the ball")

    def test_random_action(self):
        """
        Test the action of the FSM agent when no ball or paddle positions are detected.
        """
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        self.agent.ball_released = True

        action = self.agent.act(frame)
        self.assertIn(action[0], [0, 1, 2, 3], "Agent should take a random action when no positions are detected")

    def test_save_load_agent(self):
        """
        Test saving and loading the FSM agent.
        """
        filepath = '/tmp/test_fsm_agent.pkl'
        self.agent.save(filepath)
        loaded_agent = FSMAgent.load(filepath)

        self.assertIsInstance(loaded_agent, FSMAgent, "Loaded object should be an instance of FSMAgent")
        self.assertEqual(self.agent.ball_released, loaded_agent.ball_released, "Agent state should be preserved after loading")

class TestFSMUtils(unittest.TestCase):

    def test_get_ball_position(self):
        """
        Test the function to get the ball position from a frame.
        """
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[100][50] = [200, 72, 72]  # Ball position

        ball_y, ball_x = get_ball_position(frame)
        self.assertEqual((ball_y, ball_x), (100, 50), "Ball position should be correctly identified")

    def test_get_paddle_position(self):
        """
        Test the function to get the paddle position from a frame.
        """
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[190][80] = [200, 72, 72]  # Paddle position

        paddle_x = get_paddle_position(frame)
        self.assertEqual(paddle_x, 80, "Paddle position should be correctly identified")

if __name__ == '__main__':
    unittest.main()
