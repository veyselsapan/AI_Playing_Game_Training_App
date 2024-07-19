import unittest
import os
import shutil
from stable_baselines3 import DQN
from src.DQN.cnn_architecture import CustomCNN
from src.DQN.dqn import main
from src.DQN.training_utils import create_model, train_model, save_model, setup_eval_callback
from src.Common.game_env import create_env
import sys

# Add the project root directory to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

class TestDQNTrainingUtils(unittest.TestCase):
    def setUp(self):
        """
        Set up the environment and directories for testing.
        """
        self.env = create_env(environment_name='Breakout-v4', n_envs=1, n_stack=4)
        self.env.reset()
        self.log_dir = './test_logs'
        self.save_dir = './test_models'
        self.model_path = os.path.join(self.save_dir, 'test_model')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

    def tearDown(self):
        """
        Clean up the directories after testing.
        """
        shutil.rmtree(self.log_dir)
        shutil.rmtree(self.save_dir)

    def test_create_model(self):
        """
        Test the creation of the DQN model with the custom CNN architecture.
        """
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = create_model(self.env, policy_kwargs, self.log_dir)
        self.assertIsInstance(model, DQN, "Model should be an instance of DQN")

    def test_train_model(self):
        """
        Test the training of the DQN model.
        """
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = create_model(self.env, policy_kwargs, self.log_dir)
        eval_callback = setup_eval_callback(self.env, log_dir=self.log_dir)
        trained_model = train_model(model, total_timesteps=100, eval_callback=eval_callback)

        self.assertIsNotNone(trained_model, "Trained model should not be None")

    def test_save_model(self):
        """
        Test the saving of the DQN model.
        """
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = create_model(self.env, policy_kwargs, self.log_dir)
        save_model(model, self.model_path)

        self.assertTrue(os.path.exists(self.model_path + ".zip"), "Model file should exist")

if __name__ == '__main__':
    unittest.main()
